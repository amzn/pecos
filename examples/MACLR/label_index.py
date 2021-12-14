import pandas as pd
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
	"--dataset",
	type=str, default="LF-Amazon-131K",
	help="the dataset to run the experiments",
)
args = parser.parse_args()

data_path = os.path.join(os.path.abspath(os.getcwd()), 'dataset', args.dataset)
label_index = {}
all_labels = pd.read_json(os.path.join(data_path, 'lbl.json'),lines=True)
label_ids = list(all_labels.uid)
with open(os.path.join(data_path, 'trn.json')) as fp:
	for line in fp:
		item = json.loads(line.strip())
		pid = item['uid'].strip()
		for ind in item['target_ind']:
			label_id = label_ids[ind].strip()
			if label_id in label_index:
				label_index[label_id].append(pid)
			else:
				label_index[label_id] = [ind, pid]

with open(os.path.join(data_path, 'label_index.json'), 'w') as f:
	for k, v in label_index.items():
		item = {'uid': k, 'ind': v[0], 'instance': v[1:]}
		f.write(json.dumps(item) + '\n')