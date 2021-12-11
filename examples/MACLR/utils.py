import numpy as np
import math
import faiss
import torch
from tqdm.auto import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval


def perform_eval(model, label_dataloader, label_ids, instance_dataloader, inst_num, test_ids, accelerator):
	label_bert = model.label_encoder
	inst_bert = model.instance_encoder
	torch.cuda.empty_cache()
	label_bert, inst_bert = accelerator.prepare(label_bert, inst_bert)
	# label embeddings
	label_embeds = np.zeros((len(label_dataloader)*16*8, 512)).astype('float32')
	count = 0
	label_bert.eval()
	with torch.no_grad():
		for i, batch in enumerate(tqdm(label_dataloader, desc='Embedding Labels', disable=not accelerator.is_local_main_process)):
			batch_att_mask = ~(batch.eq(0))
			feature = {'input_ids': batch, 'attention_mask': batch_att_mask}
			embed = label_bert(feature)["sentence_embedding"]
			output = accelerator.gather(embed)
			output = output.data.cpu().numpy().astype('float32')
			num_label = output.shape[0]
			label_embeds[count:count + num_label, :] = output
			count += num_label
	label_num = len(label_ids)
	label_embeds = label_embeds[:label_num]
	label_embeds = label_embeds.astype('float32')
	

	# instance embeddings
	inst_embeds = np.zeros((len(instance_dataloader)*128*8, 512)).astype('float32')
	count = 0
	inst_bert.eval()
	with torch.no_grad():
		count = 0
		for i, batch in enumerate(tqdm(instance_dataloader, desc='Embedding Instances', disable=not accelerator.is_local_main_process)):
			batch_att_mask = ~(batch.eq(0))
			feature = {'input_ids': batch, 'attention_mask': batch_att_mask}
			embed = inst_bert(feature)["sentence_embedding"]
			output = accelerator.gather(embed)
			output = output.data.cpu().numpy().astype('float32')
			num_inst = output.shape[0]
			inst_embeds[count:count+num_inst, :] = output
			count += num_inst
	accelerator.print("embedding")
	inst_embeds = inst_embeds.astype('float32')
	inst_embeds = inst_embeds[:inst_num]
	test_inst_embeds = inst_embeds[-len(test_ids):]

	
	accelerator.print("Finish embedding")
	D, I = get_knn(test_inst_embeds, label_embeds, accelerator, bsz=64)
	label_bert.train()
	inst_bert.train()
	del label_embeds

	return D, I, inst_embeds


def perform_clustering(step, features, accelerator):
	# num_cluster = [10000, 20000, 40000, 80000, 100000, 100000]
	num_cluster = [2500, 5000, 10000, 20000, 40000, 80000]
	if step>=0 and step<50000:
		i = step // 10000
		d = features.shape[1]
		k = num_cluster[i]
		clus = faiss.Clustering(d, k)
		clus.verbose = False
		clus.niter = 20
		clus.nredo = 5
		clus.seed = 0
		clus.max_points_per_centroid = 1000
		clus.min_points_per_centroid = 1
		if accelerator.is_local_main_process:	
			clus.verbose = True
			res = faiss.StandardGpuResources()
			flat_config = faiss.GpuIndexFlatConfig()
			flat_config.useFloat16 = False
			flat_config.device = accelerator.local_process_index
			index = faiss.GpuIndexFlatL2(res, d, flat_config)
			features = features.astype('float32')
			clus.train(features, index)
			num_inst = features.shape[0]
			bsz = 16
			nr_batch = int(math.ceil(num_inst / bsz))
			D_list, I_list = [], []
			for bidx in range(nr_batch):
				sidx = bidx * bsz
				eidx = min((bidx + 1) * bsz, num_inst)
				D, I = index.search(features[sidx:eidx], 1)
				D_list.append(D)
				I_list.append(I)
			idxs = np.concatenate(I_list)
			cluster_result = [int(n[0]) for n in idxs]
		else:
			cluster_result = [None for _ in range(features.shape[0])]
		torch.distributed.broadcast_object_list(cluster_result, src=0, group=None)
		cluster_result = torch.LongTensor(cluster_result).to(accelerator.device)
		return cluster_result
	else:
		return None

def get_knn(inst_embeddings, label_embeddings, accelerator, top_k=100, bsz=65536):
	accelerator.print("FAISS")
	# logging.info("FAISS indexer building")
	res = faiss.StandardGpuResources()
	flat_config = faiss.GpuIndexFlatConfig()
	flat_config.useFloat16 = False
	flat_config.device = accelerator.local_process_index
	indexer = faiss.GpuIndexFlatIP(res, inst_embeddings.shape[1], flat_config)
	indexer.add(label_embeddings)
	# logging.info("FAISS indexer searching")
	num_inst = inst_embeddings.shape[0]
	nr_batch = int(math.ceil(num_inst / bsz))
	D_list, I_list = [], []
	accelerator.print("index")
	for bidx in tqdm(range(nr_batch)):
		sidx = bidx * bsz
		eidx = min((bidx + 1) * bsz, num_inst)
		D, I = indexer.search(inst_embeddings[sidx:eidx], top_k)
		D_list.append(D)
		I_list.append(I)
	D = np.concatenate(D_list)
	I = np.concatenate(I_list)
	return D, I


def eval_and_cluster(args, logger, step, model, label_dataloader, label_ids, 
					instance_dataloader, inst_num, test_ids, qrels, accelerator):
	D, I, inst_embeds = perform_eval(model, label_dataloader, label_ids, instance_dataloader, inst_num, test_ids, accelerator)
	num_tst = len(test_ids)
	results = {pid: {} for pid in test_ids}
	accelerator.print("Results")
	for row_id in range(num_tst):
		inst_id = test_ids[row_id]
		for col_id, score in zip(I[row_id], D[row_id]):
			lid = label_ids[col_id]
			results[inst_id][lid] = float(score)

	#### evaluate
	k_values = [1,3,5,10,20,100]
	if accelerator.local_process_index == 0:
		for row_id in range(num_tst):
			pid = test_ids[row_id]
			for col_id, score in zip(I[row_id], D[row_id]):
				tid = label_ids[col_id]
				results[pid][tid] = float(score)
		accelerator.print("end")
		ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)
		logger.info(ndcg)
		logger.info(_map)
		logger.info(precision)
		logger.info(recall)
		del results

	del D, I

	# clustering
	if args.mode != 'ict':
		return None
	else:
		cluster_num = inst_num - len(test_ids)
		cluster_features = inst_embeds[:cluster_num]
		cluster_result = perform_clustering(step, cluster_features, accelerator)
	return cluster_result