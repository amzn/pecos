
data_set=$1
model_name=$2
if [ -z ${data_set} ] || [ -z ${model_name} ]; then
    echo "run_encoder.sh [data_set] [model_name]"
    exit
fi

if [ ${data_set} != "nq320k" ] && [ ${data_set} != "trivia" ]; then
    echo "only support data_set={ nq320k | trivia }!"
    exit
fi

model_name_or_path="sentence-transformers/${model_name}"
input_data_dir="./data/xmc/${data_set}"
output_emb_dir="./data/embeddings/${data_set}/${model_name}"
mkdir -p ${output_emb_dir}

txt_name_arr=( "tst" "trn" "trn.abs" "trn.doc" "trn.d2q" )
for txt_name in "${txt_name_arr[@]}"; do
    input_data_path="${input_data_dir}/X.${txt_name}.txt"
    output_emb_path="${output_emb_dir}/X.${txt_name}.npy"
    output_log_path="${output_emb_dir}/X.${txt_name}.log"
    python -u encoder.py \
        ${model_name_or_path} \
        ${input_data_path} \
        ${output_emb_path} \
        |& tee ${output_log_path}
done
