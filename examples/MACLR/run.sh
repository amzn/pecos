dataset=$1
mode=$2
log=$3
model=$4
accelerate launch --config_file accelerate_config.yaml main.py \
	--dataset $dataset \
	--mode $mode \
	--log $log \
	--model-name-or-path $model \
	--pooling-mode cls \
	--proj-emb-dim 512 \
	--per-device-train-batch-size 16 \
	--learning-rate 1e-5 \
	--max-train-steps 100000 \
	--eval-steps 5000 \