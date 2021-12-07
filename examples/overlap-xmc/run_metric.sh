#!/bin/bash


data=amazon-670k
split=trn
python ./disentangle_metric.py \
  -x ./dataset/xmc-base/${data}/tfidf-attnxml/X.${split}.npz \
  --y-binned ./dataset-binned/${data}/Y.${split}.npz \
  --y-origin ./dataset/xmc-base/${data}/Y.${split}.npz \
  -m ./model/${data}-overlap \
  --binned-mapper ./dataset-binned/${data}/mapper.pkl \
  --pseudo-label-mapper ./model/${data}-overlap/pseudo_label_mapping.pkl \
  --unused-labels ./model/${data}-overlap/unused_labels.pkl \
  -b 10
