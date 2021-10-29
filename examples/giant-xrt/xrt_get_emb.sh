#================= inputs =====================
data_dir=$1                                 # e.g., ./proc_data_xrt/ogbn-arxiv
if [ -z ${data_dir} ] || [ ! -d ${data_dir} ]; then
    echo "DATA_DIR does not exist: ${data_dir}"
    exit
fi
X_txt_path=${data_dir}/X.all.txt            # input node raw text for the entire graph
model_dir=${data_dir}/xrt_models            # input pre-trained Giant-XRT models

#================= output =====================
X_emb_path=${data_dir}/X.all.xrt-emb.npy    # output fine-tuned node embeddings from Giant-XRT

#==================== train ===================
python -m pecos.xmc.xtransformer.encode \
    -t ${X_txt_path} \
    -m ${model_dir} \
    -o ${X_emb_path} \
    --batch-size 64 \
    --verbose-level 3 \
    |& tee ${model_dir}/predict.log

