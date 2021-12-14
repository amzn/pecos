import argparse
import os, sys
import logging
import json
import csv
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler
import sentence_transformers as sent_trans
import transformers
from transformers import set_seed
import accelerate
from accelerate import Accelerator
from dataset import SimpleDataset, padding_util
from model import build_encoder, DualEncoderModel
from utils import perform_eval, eval_and_cluster
import pandas as pd
import warnings
from main import parse_args
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
args = parse_args()
distributed_args = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[distributed_args])
device = accelerator.device
# Make one log on every process with the configuration for debugging.
logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
	filename=f'xmc_{args.dataset}_{args.log}_evaluate.log',
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.INFO,
)
logger.info(accelerator.state)

# Setup logging, we only want one process per machine to log things on the screen.
# accelerator.is_local_main_process is only True for one process per machine.
logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
ch = logging.StreamHandler(sys.stdout)
logger.addHandler(ch)
if accelerator.is_local_main_process:
	transformers.utils.logging.set_verbosity_info()
else:
	transformers.utils.logging.set_verbosity_error()

logger.info(sent_trans.__file__)

# If passed along, set the training seed now.
if args.seed is not None:
	set_seed(args.seed)

# Load pretrained model and tokenizer
if args.model_name_or_path == 'bert-base-uncased' or args.model_name_or_path == 'sentence-transformers/paraphrase-mpnet-base-v2':
	label_encoder = build_encoder(
		args.model_name_or_path,
		args.max_label_length,
		args.pooling_mode,
		args.proj_emb_dim,
	)
else:
	label_encoder = sent_trans.SentenceTransformer(args.model_name_or_path)

tokenizer = label_encoder._first_module().tokenizer

instance_encoder = label_encoder

model = DualEncoderModel(
	label_encoder,
	instance_encoder,
)
model = model.to(device)

# the whole label set
data_path = os.path.join(os.path.abspath(os.getcwd()), 'dataset', args.dataset)
all_labels = pd.read_json(os.path.join(data_path, 'lbl.json'),lines=True)
label_list = list(all_labels.title)
label_ids = list(all_labels.uid)
label_data = SimpleDataset(label_list, transform=tokenizer.encode)

# label dataloader for searching
sampler = SequentialSampler(label_data)
label_padding_func = lambda x: padding_util(x, tokenizer.pad_token_id, 64)
label_dataloader = DataLoader(label_data, sampler=sampler, batch_size=16, collate_fn=label_padding_func)

# test data
data_path = os.path.join(os.path.abspath(os.getcwd()), 'dataset', args.dataset)
try:
	accelerator.print("load cache")
	all_instances = torch.load(os.path.join(data_path, 'all_passages_with_titles.json.cache.pt'))
	test_data = SimpleDataset(all_instances.values())
except:
	if args.mode == 'construct-pseudo':
		test_path = os.path.join(data_path, 'trn.json')
	else:
		test_path = os.path.join(data_path, 'tst.json')
	all_instances = {}
	test_ids = []
	with open(test_path) as fp:
		for line in fp:
			inst = json.loads(line.strip())
			all_instances[inst['uid']] = inst['title'] + '\t' + inst['content']
			test_ids.append(inst['uid'])
	simple_transform = lambda x: tokenizer.encode(x, max_length=288, truncation=True)
	test_data = SimpleDataset(list(all_instances.values()), transform=simple_transform)
	inst_num = len(test_data)

sampler = SequentialSampler(test_data)
sent_padding_func = lambda x: padding_util(x, tokenizer.pad_token_id, 288)
instance_dataloader = DataLoader(test_data, sampler=sampler, batch_size=128, collate_fn=sent_padding_func)

# Prepare everything with our `accelerator`.
model, label_dataloader, instance_dataloader = accelerator.prepare(model, label_dataloader, instance_dataloader)

if args.mode == 'construct-pseudo':
	D, I, _ = perform_eval(accelerator.unwrap_model(model), label_dataloader, label_ids, instance_dataloader, inst_num, test_ids, accelerator)
	pseudo_pair_path = os.path.join(data_path, 'pseudo_pos.json')
	if accelerator.is_local_main_process:
		with open(pseudo_pair_path, 'w') as f:
			for row_id in tqdm(range(inst_num)):
					inst_id = test_ids[row_id]
					item = {'uid': inst_id}
					predict_target = []
					predict_score = []
					for col_id, score in zip(I[row_id][:5], D[row_id][:5]):
						predict_target.append(int(col_id))
						predict_score.append(float(score))
					item['predict_ind'] = predict_target
					item['score'] = predict_score
					f.write(json.dumps(item) + '\n')

else:
	# prepare pairs
	reader = csv.reader(open(os.path.join(data_path, 'all_pairs.txt'), encoding="utf-8"), delimiter=" ")
	qrels = {}
	for id, row in enumerate(reader):
		query_id, corpus_id, score = row[0], row[1], int(row[2])
		if query_id not in qrels:
			qrels[query_id] = {corpus_id: score}
		else:
			qrels[query_id][corpus_id] = score
	eval_and_cluster(args, logger, 0, accelerator.unwrap_model(model), label_dataloader, label_ids, 
					instance_dataloader, inst_num, test_ids, qrels, accelerator)