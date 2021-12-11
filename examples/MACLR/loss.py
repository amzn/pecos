import torch
import torch.distributed as dist
import torch.nn.functional as F
import mpu_utils

def compute_loss(mask, logits_mask, logits):
	exp_logits = torch.exp(logits) * logits_mask
	softmax_scores = logits - torch.log(exp_logits.sum(1, keepdim=True))
	mean_log_prob_pos = (mask * softmax_scores).sum(1) / mask.sum(1)
	return -mean_log_prob_pos.mean()

def loss_function(label_emb, inst_emb, labels, accelerator):
	assert label_emb.shape[0] == inst_emb.shape[0], "{} is not equal to {}".format(label_emb.shape[0], inst_emb.shape[0])
	assert label_emb.shape[1] == inst_emb.shape[1]
	local_batch_size = label_emb.shape[0]

	# [global_batch_size, hidden_dim]
	global_batch_size = dist.get_world_size() * local_batch_size
	all_label_emb = mpu_utils.AllgatherFromDataParallelRegion.apply(label_emb)
	all_inst_emb = mpu_utils.AllgatherFromDataParallelRegion.apply(inst_emb)
	a_norm = all_label_emb
	b_norm = all_inst_emb
	retrieval_scores = torch.mm(b_norm, a_norm.transpose(0, 1))
	qd_max, _ = torch.max(retrieval_scores, dim=1, keepdim=True)
	qd_stable_scores = retrieval_scores - qd_max.detach()
	softmax_scores = F.log_softmax(qd_stable_scores, dim=1)

	labels = torch.arange(global_batch_size).long().to(accelerator.device)
	loss = F.nll_loss(softmax_scores, labels, reduction='mean')
	reduced_losses = mpu_utils.average_losses_across_data_parallel_group([loss])
	stats_dict = dict(loss=reduced_losses[0])
	return loss, stats_dict

def loss_function_reg(label_emb, inst_emb, inst_emb_aug, reg_emb, labels, accelerator):
	assert label_emb.shape[0] == inst_emb.shape[0], "{} is not equal to {}".format(label_emb.shape[0], inst_emb.shape[0])
	assert label_emb.shape[1] == inst_emb.shape[1]

	all_label_emb = mpu_utils.AllgatherFromDataParallelRegion.apply(label_emb)
	all_inst_emb = mpu_utils.AllgatherFromDataParallelRegion.apply(inst_emb)
	all_inst_emb_aug = mpu_utils.AllgatherFromDataParallelRegion.apply(inst_emb_aug)
	all_reg_emb = mpu_utils.AllgatherFromDataParallelRegion.apply(reg_emb)

	labels = labels.contiguous().view(-1, 1)
	all_labels = accelerator.gather(labels)
	num_inst = all_label_emb.shape[0]
	num_reg = all_reg_emb.shape[0]

	mask = torch.eq(all_labels, all_labels.transpose(0, 1)).float()
	zero_mask = torch.zeros(num_inst, num_reg).to(accelerator.device)
	
	a_norm = all_label_emb
	b_norm = all_inst_emb
	inst_lbl_scores = torch.mm(b_norm, a_norm.transpose(0, 1))
	inst_lbl_max, _ = torch.max(inst_lbl_scores, dim=1, keepdim=True)
	inst_lbl_stable_scores = inst_lbl_scores - inst_lbl_max.detach()

	c_norm = torch.cat([all_inst_emb_aug, all_reg_emb])
	real_scores = torch.mm(b_norm, c_norm.transpose(0, 1))
	real_max, _ = torch.max(real_scores, dim=1, keepdim=True)
	real_stable_scores = real_scores - real_max.detach()
	real_mask = torch.cat([mask, zero_mask], dim=1)

	contrast_loss = compute_loss(mask, torch.ones_like(mask), inst_lbl_stable_scores)
	reg_loss = compute_loss(real_mask, torch.ones_like(real_mask), real_stable_scores)
	loss = contrast_loss + 1*reg_loss
	reduced_losses = mpu_utils.average_losses_across_data_parallel_group([loss, contrast_loss, reg_loss])

	stats_dict = dict(loss=reduced_losses[0], contrast_loss=reduced_losses[1], reg_loss=reduced_losses[2])
	return loss, stats_dict