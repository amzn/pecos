
dataset=$1
if [ ${dataset} != "ogbn-arxiv" ] && [ ${dataset} != "ogbn-products" ]; then
    echo "dataset=${dataset} is not yet supported!"
    exit
fi

data_root_dir=./dataset
xrt_data_dir=./proc_data_xrt
max_degree=1000

python -u proc_data_xrt.py \
    --raw-text-path ${xrt_data_dir}/${dataset}/X.all.txt \
    --vectorizer-config-path ${xrt_data_dir}/vect_config.json \
    --data-root-dir ${data_root_dir} \
    --xrt-data-dir ${xrt_data_dir} \
    --dataset ${dataset} \
    --max-deg ${max_degree}

