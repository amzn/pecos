import argparse
import logging
import math
import os, sys
import pickle
import random
import json
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import transformers
import sentence_transformers as sent_trans
import accelerate
from accelerate import Accelerator
from transformers import (
	AdamW,
	SchedulerType,
	get_scheduler,
	set_seed,
)
from utils import eval_and_cluster
from dataset import SimpleDataset, ICTXMCDataset, PosDataset, ICT_batchify, padding_util
from loss import loss_function_reg, loss_function
from model import build_encoder, DualEncoderModel
import csv
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)
retriever_report_topk_accuracies = [1, 5, 10, 20]


def parse_args():
	parser = argparse.ArgumentParser(description="Pretrain two-tower Transformer models with ICT")
	# Data
	parser.add_argument(
		"--corpus-pkl-path",
		type=str,
		help="a processed pickle file that contains title_list and block_list",
	)
	parser.add_argument(
		"--mode",
		type=str, default="ict",
		help="the mode of the training procedure",
	)
	parser.add_argument(
		"--dataset",
		type=str, default="LF-Amazon-131K",
		help="the dataset to run the experiments",
	)
	parser.add_argument(
		"--log",
		type=str, default="test",
		help="log file",
	)
	parser.add_argument(
		"--ratio",
		type=float, default=0.01,
		help="Sampling ratio",
	)
	# Model
	parser.add_argument(
		"--model-name-or-path",
		type=str, default="bert-base-uncased",
		help="Path to pretrained model or model identifier from huggingface.co/models.",
	)
	parser.add_argument(
		"--max-label-length",
		type=int, default=64,
		help="maximum label length for pre-training (default: 64)",
	)
	parser.add_argument(
		"--max-inst-length",
		type=int, default=288,
		help="maximum block (i.e., title + text) length for pre-training (default: 288)",
	)
	parser.add_argument(
		"--pooling-mode",
		type=str, default="cls",
		help="Can be a string: mean/max/cls.",
	)
	parser.add_argument(
		"--proj-emb-dim",
		type=int, default=512,
		help="embedding size of the projection layer in two-tower models",
	)
	# Optimizer
	parser.add_argument(
		"--per-device-train-batch-size",
		type=int, default=16,
		help="training batch size per GPU device (default: 8)",
	)
	parser.add_argument(
		"--learning-rate",
		type=float, default=3e-5,
		help="Initial learning rate (after the potential warmup period) to use.",
	)
	parser.add_argument(
		"--weight-decay",
		type=float, default=0,
		help="Weight decay to use.",
	)
	parser.add_argument(
		"--max-train-steps",
		type=int, default=50000,
		help="Total number of training steps to perform (default: 10,000)",
	)
	parser.add_argument(
		"--gradient-accumulation-steps",
		type=int, default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument(
		"--lr-scheduler-type",
		type=SchedulerType,
		default="linear",
		help="The scheduler type to use.",
		choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
	)
	parser.add_argument(
		"--num-warmup-steps",
		type=int, default=2000,
		help="Number of steps for the warmup in the lr scheduler (default: 1,000)",
	)
	parser.add_argument(
		"--logging-steps",
		type=int, default=100,
		help="Number of steps for the logging information (default 100)",
	)
	parser.add_argument(
		"--eval-steps",
		type=int, default=1000,
		help="Number of steps for evaluation (default 100)",
	)
	parser.add_argument(
		"--saving-steps",
		type=int, default=2000,
		help="Number of steps for the saving checkpoint (default 1000)",
	)
	# Output
	parser.add_argument(
		"--output-dir",
		type=str, default='ckpt',
		help="Where to store the final model.",
	)
	parser.add_argument(
		"--seed",
		type=int, default=None,
		help="A seed for reproducible training.",
	)
	args = parser.parse_args()
	# sanity check
	args.output_dir = os.path.join(os.path.abspath(os.getcwd()), args.output_dir)
	if args.output_dir is not None:
		os.makedirs(args.output_dir, exist_ok=True)
	return args

def main():
	# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
	args = parse_args()
	distributed_args = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
	accelerator = Accelerator(kwargs_handlers=[distributed_args])
	device = accelerator.device
	# Make one log on every process with the configuration for debugging.
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		filename=f'xmc_{args.dataset}_{args.mode}_{args.log}.log',
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
		query_encoder = build_encoder(
			args.model_name_or_path,
			args.max_label_length,
			args.pooling_mode,
			args.proj_emb_dim,
		)
	else:
		query_encoder = sent_trans.SentenceTransformer(args.model_name_or_path)

	tokenizer = query_encoder._first_module().tokenizer

	block_encoder = query_encoder

	model = DualEncoderModel(
		query_encoder,
		block_encoder,
		args.mode
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

	# label dataloader for regularization
	reg_sampler = RandomSampler(label_data)
	reg_dataloader = DataLoader(label_data, sampler=reg_sampler, batch_size=4, collate_fn=label_padding_func)

	if args.mode == 'ict':
		train_data = ICTXMCDataset(tokenizer=tokenizer, dataset=args.dataset)
	elif args.mode == 'self-train':
		train_data = PosDataset(tokenizer=tokenizer, dataset=args.dataset, labels=label_list, mode=args.mode)
	elif args.mode == 'finetune-pair':
		train_path = os.path.join(data_path, 'trn.json')
		pos_pair = []
		with open(train_path) as fp:
			for i, line in enumerate(fp):
				inst = json.loads(line.strip())
				inst_id = inst['uid']
				for ind in inst['target_ind']:
					pos_pair.append((inst_id, ind, i))
		dataset_size = len(pos_pair)
		indices = list(range(dataset_size))
		split = int(np.floor(args.ratio * dataset_size))
		np.random.shuffle(indices)
		train_indices = indices[:split]
		torch.distributed.broadcast_object_list(train_indices, src=0, group=None)
		sample_pairs = [pos_pair[i] for i in train_indices]
		train_data = PosDataset(tokenizer=tokenizer, dataset=args.dataset, labels=label_list, mode=args.mode, sample_pairs=sample_pairs)
	elif args.mode == 'finetune-label':
		label_index = []
		label_path = os.path.join(data_path, 'label_index.json')
		with open(label_path) as fp:
			for line in fp:
				label_index.append(json.loads(line.strip()))
		np.random.shuffle(label_index)
		sample_size = int(np.floor(args.ratio*len(label_index)))
		sample_label = label_index[:sample_size]
		torch.distributed.broadcast_object_list(sample_label, src=0, group=None)
		sample_pairs = []
		for i, label in enumerate(sample_label):
			ind = label['ind']
			for inst_id in label['instance']:
				sample_pairs.append((inst_id, ind, i))
		train_data = PosDataset(tokenizer=tokenizer, dataset=args.dataset, labels=label_list, mode=args.mode, sample_pairs=sample_pairs)

	train_sampler = RandomSampler(train_data)
	padding_func = lambda x: ICT_batchify(x, tokenizer.pad_token_id, 64, 288)
	train_dataloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=args.per_device_train_batch_size,
									  num_workers=4, pin_memory=False,
									  collate_fn=padding_func)


	try:
		accelerator.print("load cache")
		all_instances = torch.load(os.path.join(data_path, 'all_passages_with_titles.json.cache.pt'))
		test_data = SimpleDataset(all_instances.values())
	except:
		all_instances = {}
		test_path = os.path.join(data_path, 'tst.json')
		if args.mode == 'ict':
			train_path = os.path.join(data_path, 'trn.json')
			train_instances = {}
			valid_passage_ids = train_data.valid_passage_ids
			with open(train_path) as fp:
				for line in fp:
					inst = json.loads(line.strip())
					train_instances[inst['uid']] = inst['title'] + '\t' + inst['content']
			for inst_id in valid_passage_ids:
				all_instances[inst_id] = train_instances[inst_id]
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


	# prepare pairs
	reader = csv.reader(open(os.path.join(data_path, 'all_pairs.txt'), encoding="utf-8"), delimiter=" ")
	qrels = {}
	for id, row in enumerate(reader):
		query_id, corpus_id, score = row[0], row[1], int(row[2])
		if query_id not in qrels:
			qrels[query_id] = {corpus_id: score}
		else:
			qrels[query_id][corpus_id] = score

	logging.info("| |ICT_dataset|={} pairs.".format(len(train_data)))


	# Optimizer
	# Split weights in two groups, one with weight decay and the other not.
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{
			"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
			"weight_decay": 0.0,
		},
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

	# Prepare everything with our `accelerator`.
	model, optimizer, train_dataloader, label_dataloader, reg_dataloader, instance_dataloader = accelerator.prepare(
		model, optimizer, train_dataloader, label_dataloader, reg_dataloader, instance_dataloader)

	# Scheduler and math around the number of training steps.
	num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
	# args.max_train_steps = 100000
	args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
	args.num_warmup_steps = int(0.1*args.max_train_steps)
	lr_scheduler = get_scheduler(
		name=args.lr_scheduler_type,
		optimizer=optimizer,
		num_warmup_steps=args.num_warmup_steps,
		num_training_steps=args.max_train_steps,
	)

	# Train!
	total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
	logger.info("***** Running training *****")
	logger.info(f"  Num examples = {len(train_data)}")
	logger.info(f"  Num Epochs = {args.num_train_epochs}")
	logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
	logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
	logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
	logger.info(f"  Learning Rate = {args.learning_rate}")
	logger.info(f"  Total optimization steps = {args.max_train_steps}")
	# Only show the progress bar once on each machine.
	progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
	completed_steps = 0
	from torch.cuda.amp import autocast
	scaler = torch.cuda.amp.GradScaler()
	cluster_result = eval_and_cluster(args, logger, completed_steps, accelerator.unwrap_model(model), 
				label_dataloader, label_ids, instance_dataloader, inst_num, test_ids, qrels, accelerator)
	reg_iter = iter(reg_dataloader)	
	trial_name = f"dim-{args.proj_emb_dim}-bs-{args.per_device_train_batch_size}-{args.dataset}-{args.log}-{args.mode}"		
	for epoch in range(args.num_train_epochs):
		model.train()
		for step, batch in enumerate(train_dataloader):
			batch = tuple(t for t in batch)
			label_tokens, inst_tokens, indices = batch
			if args.mode == 'ict':
				try:
					reg_data = next(reg_iter)
				except StopIteration:
					reg_iter = iter(reg_dataloader)	
					reg_data = next(reg_iter)

			if cluster_result is not None:
				pseudo_labels = cluster_result[indices]
			else:
				pseudo_labels = indices
			with autocast():
				if args.mode == 'ict':
					label_emb, inst_emb, inst_emb_aug, reg_emb = model(label_tokens, inst_tokens, reg_data)
					loss, stats_dict = loss_function_reg(label_emb, inst_emb, inst_emb_aug, reg_emb, pseudo_labels, accelerator)
				else:
					label_emb, inst_emb = model(label_tokens, inst_tokens, reg_data=None)
					loss, stats_dict = loss_function(label_emb, inst_emb, pseudo_labels, accelerator)
				loss = loss / args.gradient_accumulation_steps

			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
			if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
				scaler.step(optimizer)
				scaler.update()
				lr_scheduler.step()
				optimizer.zero_grad()
				progress_bar.update(1)
				completed_steps += 1

			if completed_steps % args.logging_steps == 0:
				if args.mode == 'ict':
					logger.info("| Epoch [{:4d}/{:4d}] Step [{:8d}/{:8d}] Total Loss {:.6e}  Contrast Loss {:.6e}  Reg Loss {:.6e}".format(
						epoch, args.num_train_epochs,
						completed_steps, args.max_train_steps,
						stats_dict["loss"].item(),
						stats_dict["contrast_loss"].item(),
						stats_dict["reg_loss"].item(),
						)
					)
				else:
					logger.info("| Epoch [{:4d}/{:4d}] Step [{:8d}/{:8d}] Total Loss {:.6e}".format(
						epoch, args.num_train_epochs,
						completed_steps, args.max_train_steps,
						stats_dict["loss"].item(),
						)
					)
			if completed_steps % args.eval_steps == 0:
				cluster_result = eval_and_cluster(args, logger, completed_steps, accelerator.unwrap_model(model), 
				label_dataloader, label_ids, instance_dataloader, inst_num, test_ids, qrels, accelerator)
				unwrapped_model = accelerator.unwrap_model(model)
				
				unwrapped_model.label_encoder.save(f"{args.output_dir}/{trial_name}/label_encoder")
				unwrapped_model.instance_encoder.save(f"{args.output_dir}/{trial_name}/instance_encoder")
			
			if completed_steps >= args.max_train_steps:
				break

if __name__ == "__main__":
	main()

