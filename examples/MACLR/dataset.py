import os
import pickle
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize as sent_tok
from torch.utils.data import Dataset
import logging, sys

loggers = {}
def get_logger(name='default'):
	try:
		import loguru
		return loguru.logger
	except ImportError:
		pass

	global loggers
	if loggers.get(name):
		return loggers.get(name)
	else:
		logger = logging.getLogger(__name__)
		logger.setLevel(logging.DEBUG)

		handler = logging.StreamHandler(sys.stdout)
		handler.setLevel(logging.DEBUG)

		# formatter = logging.Formatter(
		#     fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')
		# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		loggers[name] = logger
		return logger

logger = get_logger()
class SimpleDataset(Dataset):
	def __init__(self, data, transform=None):
		'''Simple dataset
		'''
		self.instances = data
		self.transform = transform

	def __len__(self):
		return len(self.instances)

	def __getitem__(self, index):
		if self.transform:
			return self.transform(self.instances[index])
		else:
			return self.instances[index]


class ICTXMCDataset(Dataset):
	"""
	ICT data generator
	"""
	def __init__(self, tokenizer, dataset):
		self.path = os.path.abspath(os.getcwd())
		self.data_path = os.path.join(self.path, 'dataset', dataset)
		self.tokenizer = tokenizer
		self.passage_sent_dict, self.titles = self._sent_tokenize_passages(dataset)
		self.valid_passage_ids = [pid for pid in self.passage_sent_dict if len(self.passage_sent_dict[pid]) > 1]
		self.passageids_set = set(self.passage_sent_dict.keys())

		# This could be done on the fly to save memory
		# self.passage_dict = self._get_pid_to_tok(passage_path)

	def __len__(self):
		return len(self.valid_passage_ids)

	def __getitem__(self, idx):
		inst_id = self.valid_passage_ids[idx]
		inst = self.passage_sent_dict[inst_id]
		assert(len(inst) > 1)
		label = self.titles[inst_id]
		
		label_tokens = self.tokenizer.encode(label)
		full_inst = " ".join(sent.replace(label, "") if random.random() > 0.1 else sent for sent in inst)
		inst_tokens = self.tokenizer.encode(full_inst, max_length=288, truncation=True)

		return label_tokens, inst_tokens, idx


	def _sent_tokenize_passages(self, dataset):
		"""Tokenize the passage text into a list of sentences"""
		try:
			data_path = os.path.join(self.path, 'dataset', dataset)
			passages_path = os.path.join(data_path, 'passages.pkl')
			titles_path = os.path.join(data_path, 'titles.pkl')
			passages = pickle.load(open(passages_path, 'rb'))
			titles = pickle.load(open(titles_path, 'rb'))
		except:
			data_path = os.path.join(self.path, 'dataset', dataset)
			json_path = os.path.join(data_path, 'trn.json')
			passage_lines = open(json_path, 'r').readlines()
			passages = {}
			titles = {}
			for line in tqdm(passage_lines):
				json_line = json.loads(line)
				passages[json_line['uid']] = sent_tok(json_line['content'])
				titles[json_line['uid']] = json_line['title']
			# save as pickle files
			passages_path = os.path.join(data_path, 'passages.pkl')
			titles_path = os.path.join(data_path, 'titles.pkl')
			pickle.dump(passages, open(passages_path, 'wb'))
			pickle.dump(titles, open(titles_path, 'wb'))
		return passages, titles

	def _get_pid_to_tok(self, passage_path):
		passage_path_cache = passage_path + '.cache.pt'
		try:
			pid_tok_dict = torch.load(passage_path_cache)
			logger.info(f'Loading from {passage_path_cache}!')
		except:
			passage_dict = self._load_passages(passage_path)
			pid_tok_dict = {}
			for k, v in passage_dict.items():
				pid_tok_dict[k] = self.tokenizer.encode(v)  # roberta tokenizer API might be different
			torch.save(pid_tok_dict, passage_path_cache)
			logger.info(f'{passage_path_cache} saved!')
		return pid_tok_dict

	def _load_passages(self, passage_path):
		passage_lines = open(passage_path, 'r').readlines()
		passages = {}
		for line in tqdm(passage_lines):
			psgs = json.loads(line)
			if isinstance(psgs, dict): psgs = [psgs]
			for psg in psgs:
				passages[psg['uid']] = psg['title'] + '\t' + psg['content']
		return passages

class PosDataset(ICTXMCDataset):
	def __init__(self, tokenizer, dataset, labels, mode, sample_pairs=None):
		super(PosDataset, self).__init__(tokenizer, dataset)
		
		self.pos_pair = []
		self.labels = labels
		self.mode = mode
		if self.mode == 'self-train':
			with open(os.path.join(self.data_path, 'pseudo_pos.json')) as fp:
				with open(os.path.join(self.data_path, 'pseudo_pos_tfidf.json')) as fp2:
					i = 0
					for line, line2 in zip(fp, fp2):
						pseudo_pair = json.loads(line.strip())
						pseudo_pair2 = json.loads(line2.strip())
						pid = pseudo_pair['uid']
						if len(self.passage_sent_dict[pid]) < 1:
							continue
						tfidf_pred = []
						if len(pseudo_pair2['predict_ind'])==0:
							tfidf_pred = []
						else:
							for tfidf_ind in pseudo_pair2['predict_ind'][:1]:
								self.pos_pair.append((pid, tfidf_ind, i))
								tfidf_pred.append(tfidf_ind)
						for ind, score in zip(pseudo_pair['predict_ind'][:1], pseudo_pair['score'][:1]):
							if score > 36 and ind != tfidf_pred:
							# if score > 30 and ind not in tfidf_pred:
								self.pos_pair.append((pid, ind, i))
						i = i + 1
		else:
			self.pos_pair = sample_pairs


	def __len__(self):
		return len(self.pos_pair)

	def __getitem__(self, idx):

		inst_id = self.pos_pair[idx][0]
		label_id = self.pos_pair[idx][1]
		idx = self.pos_pair[idx][2]
		inst = self.passage_sent_dict[inst_id]
		full_inst = self.titles[inst_id] + '\t' + " ".join(sent for sent in inst)

		label = self.labels[label_id]
		label_tokens = self.tokenizer.encode(label)
		inst_tokens = self.tokenizer.encode(full_inst, max_length=288, truncation=True)
		return label_tokens, inst_tokens, idx

def padding_util(examples, padding_id, seq_len):
	length = max([len(example) for example in examples])
	length = min(length, seq_len)
	batch = np.ones((len(examples), length)) * padding_id
	for i, example in enumerate(examples):
		idx = min(len(example), length)
		batch[i, :idx] = example[:idx - 1] + [example[-1]]
	return torch.tensor(batch, dtype=torch.long)


def ICT_batchify(examples, padding_id=0, max_label_len=64, max_instance_len=288):
	"""
	batch_size x query_length, num_passages x passage_length, batch_size x1 (labels)
	"""
	batch_size = len(examples)
	label_len = max([len(example[0]) for example in examples])
	label_len = min(label_len, max_label_len)
	label_tokens = np.ones((batch_size, label_len)) * padding_id
	indices = np.zeros(batch_size)
	instances = []
	count = 0
	for i, example in enumerate(examples):
		idx = min(len(example[0]), label_len)
		label_tokens[i, :idx] = example[0][:idx - 1] + [example[0][-1]]
		inst = example[1]
		instances.append(inst)
		count += 1
		indices[i] = example[2]

	instance_len = max([len(inst) for inst in instances])
	instance_len = min(instance_len, max_instance_len)
	inst_tokens = np.ones((count, instance_len)) * padding_id
	for i, inst in enumerate(instances):
		idx = min(len(inst), instance_len)
		inst_tokens[i, :idx] = inst[: idx - 1] + [inst[-1]]
	return torch.tensor(label_tokens, dtype=torch.long), torch.tensor(inst_tokens, dtype=torch.long), torch.tensor(indices, dtype=torch.long)