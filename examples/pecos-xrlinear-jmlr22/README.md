# Experiment Code for PECOS Technical Report, JMLR 2022

This folder contains code to train XR-Linear models and reproduce experiments 
in ["PECOS: Prediction for Enormous and Correlated Output Spaces"](https://arxiv.org/abs/2010.05878).


## Getting Started
* Clone the repository and enter `examples/pecos-xrlinear-jmlr22` directory. 
* First create a [virtual environment](https://docs.python.org/3/library/venv.html) and then install dependencies 
by running the following command:
```bash 
pip install -r requirements.txt
``` 
If you're unfamiliar with Python virtual environments, check out the 
[user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).


## Downloading Data
The XMC datasets can be download at
``` bash
cd ./datasets
# eurlex-4k, wiki10-31k, amazoncat-13k, amazon-670k, wiki-500k, amazon-3m
DATASET="eurlex-4k"
wget https://archive.org/download/pecos-dataset/xmc-base/${DATASET}.tar.gz
tar -zxvf ./${DATASET}.tar.gz
``` 

## XR-Linear Models with Various Hierarchical Label Trees
For the results in Table 1 and Table 3,
we train and evaluate XR-Linear models with different branching factor of hierarchical label trees (HLTs),
which implicitly controls the tree depth of HLTs.

For each braching factor `B={2, 8, 32}`, we first learn three HLTs under three different random seeds.
We then create an ensemble model by aggregating predictions from three XR-Linear models. 
``` bash
bash exp_v1.sh ${DATASET}
```

The experiment results of Table 1 are available at
``` bash
tail ./exp_v1/saved_models/${DATASET}/nrs-32_*.log -n 3
```
Similarly, the experiment results of Table 3 are available at
``` bash
tail ./exp_v1/saved_models/${DATASET}/nrs-*_ensemble-average.log -n 3
```


## XR-Linear Models with Various Negative Sampling Scheme
For the results in Table 2,
we train and evaluate XR-Linear models with different negative sampling scheme.

``` bash
NS_SCHEME="tfn+man"
bash exp_v2.sh ${DATASET} ${NS_SCHEME}
```
The experiment results of Table 3 are available at
``` bash
tail ./exp_v2/saved_models/${DATASET}k/ns-${NS_SCHEME}/beam-50_ensemble-average.log -n 3
```


## XR-Transformer Models
To reproduce experiment results of XR-Transformer models, see
[link](https://github.com/amzn/pecos/tree/mainline/examples/xr-transformer-neurips21)


## Citation

If you find this useful, please consider citing our paper.

* ["PECOS: Prediction for Enormous and Correlated Output Spaces"](https://arxiv.org/abs/2010.05878) [[bib]](./bibtex/yu2020pecos.bib)
