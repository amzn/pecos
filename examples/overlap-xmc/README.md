# Label Disentanglement in Partition-based Extreme Multilabel Classification, NeurIPS 2021

This folder contains code to reproduce the key experiments in "[Label Disentanglement in Partition-based Extreme Multilabel Classification](https://arxiv.org/pdf/2106.12751.pdf)"

## Get Started

+ Clone the repository and enter `examples/overlap-xmc` directory.
+ First create a [virtual environment](https://docs.python.org/3/library/venv.html) and then install dependencies by running the following command:

```
pip install numba==0.52.0
pip install scipy==1.4.1
```

If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

After installing, create following folders:

```
mkdir dataset/ dataset-binned/ model/
```

## Downloading Data

The XMC datasets can be download at

```
# eurlex-4k, wiki10-31k, amazoncat-13k, amazon-670k, wiki-500k, amazon-3m
DATASET="wiki10-31k"
wget https://archive.org/download/pecos-dataset/xmc-base/${DATASET}.tar.gz
tar -zxvf ./${DATASET}.tar.gz
```

Then, move the data folders to `./dataset` folder:

```
cp -rf ./xmc-base ./dataset && rm -rf ./xmc-base
```

## Training and Evaluation

The training and evaluation of our label disentablement model, run

```
# reproduce ours + XR-Linear:
bash run_base.sh ${DATASET}
# reproduce Figure2:
base run_binned.sh ${DATASET}
# reproduce VI (should be launched after completion of `run_binned.sh`):
base run_metric.sh
```

Recommended platform for training: [AWS p3.16xlarge](https://aws.amazon.com/cn/ec2/instance-types/p3/) instance or equivalent.


## Known Issues

+ Be careful about the scipy and numba versions (needs to be 1.4.1 and 0.52.0, respectively).
+ For reproducing `Our method` + `X-Transformer` results, we need to upload the extracted features first. Currently they are missing.

## Citation

Please consider to cite this paper if you find our work helpful:
```
@inproceedings{liu2021label,
  title={Label disentanglement in partition-based extreme multilabel classification},
  author={Liu, Xuanqing and Chang, Wei-Cheng and Yu, Hsiang-Fu and Hsieh, Cho-Jui and Dhillon, Inderjit S},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```