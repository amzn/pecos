#================= inputs =====================
data_name=$1
model_name=$2
data_dir=$3

X_trn=${data_dir}/X.trn.txt # training text
X_tst=${data_dir}/X.tst.txt # test text

Y_trn=${data_dir}/Y.trn.npz # training label matrix
Y_tst=${data_dir}/Y.tst.npz # test label matrix
X_feat_trn=${data_dir}/tfidf-attnxml/X.trn.npz # training tfidf feature
X_feat_tst=${data_dir}/tfidf-attnxml/X.tst.npz # test tfidf feature

model_dir=models/${data_name}/${model_name}
mkdir -p ${model_dir}

params_dir=params/${data_name}/${model_name}

python3 -m pecos.xmc.xtransformer.train \
                                --trn-text-path ${X_trn} \
                                --trn-feat-path ${X_feat_trn} \
                                --trn-label-path ${Y_trn} \
                                --model-dir ${model_dir} \
                                --params-path ${params_dir}/params.json \
                                |& tee ${model_dir}/train.log

python3 -m pecos.xmc.xtransformer.predict \
				--feat-path ${X_feat_tst} \
				--text-path ${X_tst} \
				--model-folder ${model_dir} \
				--batch-gen-workers 16 \
				--save-pred-path ${model_dir}/Pt.npz \
				--batch-size 128 \
				|& tee ${model_dir}/predict.log

python3 -m pecos.xmc.xlinear.evaluate \
				-y ${Y_tst} \
				-p ${model_dir}/Pt.npz \
				--topk 10 \
				|& tee ${model_dir}/result.log
