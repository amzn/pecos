
dataset=$1
exp_mode=$2
if [ -z "${dataset}" ] || [ -z "${exp_mode}" ]; then
    echo "bash run_exp.sh [dataset] [exp_mode]"
    exit
fi

x_npz_path=./data/${dataset}/tfidf-attnxml/X.trn.npz
y_npz_path=./data/${dataset}/Y.trn.npz
seed_arr=( 1 2 3 4 5 )

if [[ "${exp_mode}" == "single-thread" ]]; then
    n_thread=1
    algo_arr=( scipy pecos intel-mkl pytorch tensorflow )
    for algo in "${algo_arr[@]}"; do
        log_dir=./results/${dataset}/${algo}
        mkdir -p ${log_dir}
        for seed in "${seed_arr[@]}"; do
            export MKL_INTERFACE_LAYER=ILP64
            export OMP_NUM_THREADS=${n_thread}
            export MKL_NUM_THREADS=${n_thread}
            python3 -u run_exp.py \
                --x-npz-path ${x_npz_path} \
                --y-npz-path ${y_npz_path} \
                --spmm-algo ${algo} \
                --threads ${n_thread} \
                |& tee ${log_dir}/t-${n_thread}.s-${seed}.log
        done
    done
elif [[ "${exp_mode}" == "multi-thread" ]]; then
    thread_arr=( 2 4 8 16 32 )
    algo_arr=( pecos intel-mkl )
    for algo in "${algo_arr[@]}"; do
        log_dir=./results/${dataset}/${algo}
        mkdir -p ${log_dir}
        for n_thread in "${thread_arr[@]}"; do
            for seed in "${seed_arr[@]}"; do
                export MKL_INTERFACE_LAYER=ILP64
                export OMP_NUM_THREADS=${n_thread}
                export MKL_NUM_THREADS=${n_thread}
                python3 -u run_exp.py \
                    --x-npz-path ${x_npz_path} \
                    --y-npz-path ${y_npz_path} \
                    --spmm-algo ${algo} \
                    --threads ${n_thread} \
                    |& tee ${log_dir}/t-${n_thread}.s-${seed}.log
            done
        done
    done
else
    echo "exp_mode=${exp_mode} is not support! Consider {single-thread, multi-thread}"
fi
