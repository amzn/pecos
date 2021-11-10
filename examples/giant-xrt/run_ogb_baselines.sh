
dataset=$1
gnn_algo=$2

if [ ${dataset} == "ogbn-arxiv" ]; then
    RUNS=10
    if [ ${gnn_algo} == "mlp" ]; then
        python -u OGB_baselines/${dataset}/mlp.py \
            --runs ${RUNS} \
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
    else
        echo "gnn_algo=${gnn_algo} is not yet supported for ogbn-arxiv!"
    fi
elif [ ${dataset} == "ogbn-products" ]; then
    RUNS=10
    if [ ${gnn_algo} == "mlp" ]; then
        python -u OGB_baselines/${dataset}/mlp.py \
            --runs ${RUNS} \
            --data_root_dir ./dataset \
            --node_emb_path ./proc_data_xrt/${dataset}/X.all.xrt-emb.npy \
            |& tee OGB_baselines/${dataset}/mlp.giant-xrt.log
    elif [ ${gnn_algo} == "graph-saint" ]; then
        CUDA_VISIBLE_DEVICES=1 python -u OGB_baselines/${dataset}/graph_saint.py \
            --runs ${RUNS} \
            --data_root_dir ./dataset \
            --eval_steps 10 \
            --epochs 50 \
            --num_layers 1 \
            --walk_length 1 \
            --hidden_channels 192 \
            --lr 1e-3 \
            --node_emb_path ./proc_data_xrt/${dataset}/X.all.xrt-emb.v2.npy \
            |& tee OGB_baselines/${dataset}/graph-saint.giant-xrt.v2.h-192_lr-1e-3..log
    else
        echo "gnn_algo=${gnn_algo} is not supported for ogbn-arxiv!"
    fi
elif [ ${dataset} == "ogbn-papers100M" ]; then
    RUNS=5
    if [ ${gnn_algo} == "mlp" ]; then
        python -u OGB_baselines/${dataset}/mlp_xrt.py \
            --runs ${RUNS} \
            --data_root_dir ./dataset \
            --epochs 50 \
            --lr 1e-3 \
            --node_emb_path ./proc_data_xrt/${dataset}/X.all.xrt-emb.npy \
            |& tee OGB_baselines/${dataset}/mlp.giant-xrt.log
    elif [ ${gnn_algo} == "sgc" ]; then
        #python -u OGB_baselines/${dataset}/sgc.py \
        #    --data_root_dir ./dataset \
        #    --node_emb_path ./proc_data_xrt/${dataset}/X.all.xrt-emb.npy \
        #    --output_path ./proc_data_xrt/${dataset}/X.all.xrt-emb.sgc.pt
        python -u OGB_baselines/${dataset}/mlp_sgc.py \
            --runs ${RUNS} \
            --epochs 50 \
            --lr 1e-3 \
            --sgc_dict_pt ./proc_data_xrt/${dataset}/X.all.xrt-emb.sgc.pt \
            |& tee OGB_baselines/${dataset}/sgc.giant-xrt.log
    else
        echo "gnn_algo=${gnn_algo} is not supported for ogbn-papers100M"
    fi
else
    echo "dataset=${dataset} is not yet supported!"
fi
