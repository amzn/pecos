dataset=$1
log=$2
mode=$3
model=$4
accelerate launch --config_file accelerate_config.yaml evaluate.py \
	--dataset $dataset \
	--log $log \
	--mode $mode \
	--model-name-or-path $model \
