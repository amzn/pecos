dataset=$1

echo *** data downloading and preprocessing ***
./scripts/download_data.sh ${dataset}

python3 DataPrep_forXCrepo.py --dataset ${dataset}

python3 pecos_dataform_full.py ${dataset}

perl convert_format.pl ./dataset/${dataset}/train.txt ./dataset/${dataset}/trn_X_Xf.txt ./dataset/${dataset}/trn_X_Y.txt
perl convert_format.pl ./dataset/${dataset}/test.txt ./dataset/${dataset}/tst_X_Xf.txt ./dataset/${dataset}/tst_X_Y.txt
python3 PrepareXYstack-raw.py --dataset ${dataset}
echo "Data Preparation for ${dataset} is done!"

echo *** PINA pre-training ***

./scripts/xtransformer-XYstack-raw.sh v0-raw-pre BoW ${dataset}

python3 PINA_augmentation.py --model_name v0-raw-pre --dataset ${dataset}

echo *** Downstream Model Training ***

./scripts/xtransformer-XYstack-DS-raw.sh v0-raw-pre v0-raw-s0 BoW ${dataset} 5 Lft_xrt
./scripts/xtransformer-XYstack-DS-raw.sh v0-raw-pre v0-raw-s1 BoW ${dataset} 5 Lft_xrt
./scripts/xtransformer-XYstack-DS-raw.sh v0-raw-pre v0-raw-s2 BoW ${dataset} 5 Lft_xrt

echo *** Evaluation ***
./scripts/ensemble_evaluation.sh ${dataset} "v0-raw-pre" "v0-raw-s0,v0-raw-s1,v0-raw-s2"
