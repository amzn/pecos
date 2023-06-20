#================= inputs =====================
model_name=$1
feature_name=$2
work_dir="."
dname=$3
topk=20


params_path=${work_dir}/scripts/params/xtransformer/${dname}/${model_name}.json

model_dir="${work_dir}/models_LF/xtransformer/${dname}/${model_name}/${feature_name}/XYstack"
mkdir -m777 -p ${model_dir}

X_all=${work_dir}/dataset/${dname}/raw/X_all.txt
X_trn=${work_dir}/dataset/${dname}/raw/X.trn.txt
X_tst=${work_dir}/dataset/${dname}/raw/X.tst.txt

if [[ $feature_name == "BoW" ]]
then
    echo $feature_name
    feature_dir="${work_dir}/dataset/${dname}"
    Xf_all=${feature_dir}/X_bow.all.npz
    Xf_trn=${feature_dir}/X_bow.trn.npz
    Xf_tst=${feature_dir}/X_bow.tst.npz
else
    echo $feature_name
    feature_dir="${work_dir}/dataset/${dname}/tfidf/${feature_name}/XYstack/normalized"
    Xf_all=${feature_dir}/X.tfidf.all.npz
    Xf_trn=${feature_dir}/X.tfidf.trn.npz
    Xf_tst=${feature_dir}/X.tfidf.tst.npz
fi
Y_all=${work_dir}/dataset/${dname}/raw/Y_all.npz

# ================ training ====================
if [ -f "${model_dir}/train.log" ]; then
    echo ${model_dir}/train.log exists, skip...
else
python3 -m pecos.xmc.xtransformer.train -t ${X_all} -x ${Xf_all} -y ${Y_all} -m ${model_dir} \
	--params-path ${params_path} \
	|& tee ${model_dir}/train.log
fi
# ================ eval ========================
if [ -f "${model_dir}/eval_tst.log" ]; then
    echo ${model_dir}/eval_tst.log exists, skip...
else
python3 -m pecos.xmc.xtransformer.predict -t ${X_all} -x ${Xf_all} -m ${model_dir} --only-topk $topk\
    -o ${model_dir}/P.${topk}.npz \
    |& tee ${model_dir}/eval_tst.log

python3 -m pecos.xmc.xlinear.evaluate -y ${Y_all} -p ${model_dir}/P.${topk}.npz -k 10 \
    |& tee ${model_dir}/eval_tst.log
fi
# =============== get prediction ==============
if [ -f "${model_dir}/P.${topk}.trn.npz" ]; then
    echo ${model_dir}/P.${topk}.trn.npz exists, skip...
else
python3 -m pecos.xmc.xtransformer.predict \
	-t ${X_trn} -x ${Xf_trn} -m ${model_dir} --only-topk $topk \
    -o ${model_dir}/P.${topk}.trn.npz 
fi
    
if [ -f "${model_dir}/P.${topk}.tst.npz" ]; then
    echo ${model_dir}/P.${topk}.tst.npz exists, skip...
else
python3 -m pecos.xmc.xtransformer.predict \
	-t ${X_tst} -x ${Xf_tst} -m ${model_dir} --only-topk $topk \
    -o ${model_dir}/P.${topk}.tst.npz 
fi    
