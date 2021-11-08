
dataset=$1
gnn_algo=$2
RUNS=10
if [ ${dataset} != "ogbn-arxiv" ] && [ ${dataset} != "ogbn-products" ]; then
    echo "dataset=${dataset} is not yet supported!"
    exit
fi

if [ ${gnn_algo} == "mlp" ]; then
    python -u OGB_baselines/${dataset}/mlp.py \
        --runs ${RUNS} \
        --log_steps 10 \
        --data_root_dir ./dataset \
        --node_emb_path ./proc_data_xrt/${dataset}/X.all.xrt-emb.npy \
        |& tee OGB_baselines/${dataset}/mlp.giant-xrt.log
elif [ ${gnn_algo} == "graph-sage" ]; then
    python -u OGB_baselines/${dataset}/gnn.py \
        --runs ${RUNS} \
        --data_root_dir ./dataset \
        --use_sage \
        --lr 8e-4 \
        --node_emb_path ./proc_data_xrt/${dataset}/X.all.xrt-emb.npy \
        |& tee OGB_baselines/${dataset}/graph-sage.giant-xrt.log
elif [ ${gnn_algo} == "graph-saint" ]; then
    python -u OGB_baselines/${dataset}/graph_saint.py \
        --runs ${RUNS} \
        --data_root_dir ./dataset \
        --eval_steps 5 \
        --epochs 50 \
        --num_layers 1 \
        --walk_length 1 \
        --node_emb_path ./proc_data_xrt/${dataset}/X.all.xrt-emb.npy \
        |& tee OGB_baselines/${dataset}/graph-saint.giant-xrt.log
else 
    echo "gnn_algo=${gnn_algo} is not yet supported!"
fi
