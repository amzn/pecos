#================= inputs =====================
ens_name="softmax"
dname=$1
model_name=$2
DS_model_names=$3
topk=20
model_dir="./models_LF/xtransformer/${dname}/${model_name}/BoW/XYstack/downstream/${DS_model_names}/5/Lft_xrt"

python3 Ensemble-PINA.py --dataset ${dname} \
                    --model_name ${model_name} \
                    --DS_model_names ${DS_model_names}

echo "===After reciprocal pair removal==="
python3 -u ./evaluate.py \
            "./dataset/${dname}/trn_X_Y.txt" \
            "./dataset/${dname}/tst_X_Y.txt" \
            "${model_dir}/P.${topk}.${ens_name}" ./dataset/${dname} \
            |& tee ${model_dir}/Reciprocal_Removed_eval.log 
