#!/bin/bash

# Complete the training of our method
# By default, amazon-670k will be used
data=${1-amazon-670k}

# Step 1. train a XR-Linear model
echo "Training XR-Linear model"
python -m pecos.xmc.xlinear.train \
  -x ./dataset/xmc-base/${data}/tfidf-attnxml/X.trn.npz \
  -y ./dataset/xmc-base/${data}/Y.trn.npz \
  -m ./model/${data} \
  --nr-splits 32 \
  -b 10


# Step 4. create new dataset with overlapping label space
echo "Creating new dataset"
python reorganize_clusters.py \
  -x ./dataset/xmc-base/${data}/tfidf-attnxml/X.trn.npz \
  -y ./dataset/xmc-base/${data}/Y.trn.npz \
  -m model/${data} \
  -b 10 \
  --n_copies 2 \
  -o model/${data}-overlap

# Step 5. train the model again on the new dataset
echo "Train our model"
python -m pecos.xmc.xlinear.train \
  -x ./dataset/xmc-base/${data}/tfidf-attnxml/X.trn.npz \
  -y ./model/${data}-overlap/Y.trn.npz \
  -c ./model/${data}-overlap \
  -m ./model/${data}-overlap \
  -b 10
 

# Step 6. perform error analysis on our model
echo "Computing Prec@k and Recall@k"
python error_analyze.py \
  -x ./dataset/xmc-base/${data}/tfidf-attnxml/X.tst.npz \
  -y ./dataset/xmc-base/${data}/Y.tst.npz \
  -m ./model/${data}-overlap \
  -b 10 \
  --mapper ./model/${data}-overlap/pseudo_label_mapping.pkl \
  --unused-labels ./model/${data}-overlap/unused_labels.pkl
