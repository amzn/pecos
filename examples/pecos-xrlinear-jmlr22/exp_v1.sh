
if [ -z ${1} ]; then
    echo "bash exp_v1.sh [dataset]"
    exit
fi
if [ ${1} != "eurlex-4k" ] && [ ${1} != "wiki10-31k" ] && [ ${1} != "amazoncat-13k" ] \
    [ ${1} != "amazon-670k" ] && [ ${1} != "wiki-500k" ] && [ ${1} != "amazon-3m" ]; then
    echo "dataset=${1} is not support!"
    exit
fi

dataset=$1
data_dir=./datasets/xmc-base/${dataset}
output_dir=./exp_v1
beam_size=10

# EXP-1
nrs_arr=( 2 8 32 )
seed_arr=( 0 1 2 )
for nr_splits in "${nrs_arr[@]}"; do
    pred_npz_string=""
    pred_tag_string=""
    
    for seed in "${seed_arr[@]}"; do
        saved_model_dir=${output_dir}/saved_models/${dataset}/nrs-${nr_splits}_seed-${seed}
        mkdir -p ${saved_model_dir}
        pred_npz=${saved_model_dir}/Yp.tst.b-${beam_size}.npz
        pred_tag=${nr_splits}_seed-${seed}
        python -u xrl_train.py \
            -x ${data_dir}/tfidf-attnxml/X.trn.npz \
            -y ${data_dir}/Y.trn.npz \
            -m ${saved_model_dir} \
            --nr-splits ${nr_splits} \
            --seed ${seed} \
            --beam-size ${beam_size} \
            |& tee ${saved_model_dir}/train.log
        python -u xrl_predict.py \
            -m ${saved_model_dir} \
            -x ${data_dir}/tfidf-attnxml/X.tst.npz \
            -y ${data_dir}/Y.tst.npz \
            -o ${pred_npz} \
            |& tee ${saved_model_dir}/eval.log
        pred_npz_string="${pred_npz_string} ${pred_npz}"
        pred_tag_string="${pred_tag_string} ${pred_tag}"
    done
    ##
    ens_method_arr=( average rank_average softmax_average sigmoid_average )
    for ens_method in "${ens_method_arr[@]}"; do
        python -u ensemble_evaluate.py \
            -y ${data_dir}/Y.tst.npz \
            -p ${pred_npz_string} \
            --tags ${pred_tag_string} \
            --ens-method ${ens_method} \
            |& tee ${output_dir}/saved_models/${dataset}/nrs-${nr_splits}_ensemble-${ens_method}.log
    done
    #exit
done
