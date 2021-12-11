import torch
import torch.nn as nn
import sentence_transformers as sent_trans
class DualEncoderModel(torch.nn.Module):
	def __init__(self, label_encoder, instance_encoder, mode='ict'):
		super(DualEncoderModel, self).__init__()
		self.label_encoder = label_encoder
		self.instance_encoder = instance_encoder
		self.mode = mode

	def forward(self, label_tokens, inst_tokens, reg_data):
		# [local_batch_size, query_seq_len]
		label_att_mask = ~(label_tokens.eq(0))
		inst_att_mask = ~(inst_tokens.eq(0))


		label_f = {'input_ids': label_tokens, 'attention_mask': label_att_mask}
		label_emb = self.label_encoder(label_f)
		label_emb = label_emb["sentence_embedding"]

		inst_f = {'input_ids': inst_tokens, 'attention_mask': inst_att_mask}
		inst_emb = self.instance_encoder(inst_f)
		inst_emb = inst_emb["sentence_embedding"]

		if self.mode == 'ict':
			reg_att_mask = ~(reg_data.eq(0))
			reg_f = {'input_ids': reg_data, 'attention_mask': reg_att_mask}
			reg_emb = self.label_encoder(reg_f)
			reg_emb = reg_emb["sentence_embedding"]

			inst_f_aug = {'input_ids': inst_tokens.detach().clone(), 'attention_mask': inst_att_mask.detach().clone()}
			inst_emb_aug = self.instance_encoder(inst_f_aug)
			inst_emb_aug = inst_emb_aug["sentence_embedding"]
			return label_emb, inst_emb, inst_emb_aug, reg_emb
		else:
			return label_emb, inst_emb

def build_encoder(
	model_name_or_path,
	max_seq_length,
	pooling_mode,
	proj_emb_dim,
	drop_prob=0.1,
):
	base_layer = sent_trans.models.Transformer(model_name_or_path, max_seq_length=None)
	pooling_layer = sent_trans.models.Pooling(
		base_layer.get_word_embedding_dimension(),
		pooling_mode=pooling_mode,
	)
	dense_layer = sent_trans.models.Dense(
		in_features=pooling_layer.get_sentence_embedding_dimension(),
		out_features=proj_emb_dim,
		activation_function=nn.Tanh(),
	)
	# normalize_layer = sent_trans.models.LayerNorm(proj_emb_dim)
	normalize_layer = sent_trans.models.Normalize()
	dropout_layer = sent_trans.models.Dropout(dropout=drop_prob)
	proj_layer = sent_trans.models.Dense(
		in_features=512,
		out_features=128,
		activation_function=nn.Tanh(),
	)
	# encoder = sent_trans.SentenceTransformer(
	#     modules=[base_layer, pooling_layer, dense_layer, normalize_layer, dropout_layer],
	# )
	encoder = sent_trans.SentenceTransformer(
	modules=[base_layer, pooling_layer, dense_layer],
)
	return encoder