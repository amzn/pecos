#================= inputs =====================
data_dir=$1                             # e.g., ./proc_data_xrt/ogbn-arxiv
if [ -z ${data_dir} ] || [ ! -d ${data_dir} ]; then
    echo "DATA_DIR does not exist: ${data_dir}"
    exit
fi
Y_npz_path=${data_dir}/Y.trn.npz        # training label matrix
X_txt_path=${data_dir}/X.trn.txt        # training text
X_npz_path=${data_dir}/X.trn.tfidf.npz  # training tfidf feature
X_pt_path=${data_dir}/X.trn.pt          # save trn tensors here
params_path=${data_dir}/params.json     # train/predict hyper-parameters in json file

#================== outputs ===================
model_dir=${data_dir}/xrt_models
mkdir -p ${model_dir}
TMPDIR=${model_dir}/tmp
mkdir -p ${TMPDIR}
export TMPDIR=${model_dir}/tmp

#==================== train ===================
python -m pecos.xmc.xtransformer.train \
    -t ${X_txt_path} \
    -x ${X_npz_path} \
    -y ${Y_npz_path} \
    -m ${model_dir} \
    --params-path ${params_path} \
    --verbose-level 3 \
    |& tee ${model_dir}/train.log

