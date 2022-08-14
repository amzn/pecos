# PECOS - Predictions for Enormous and Correlated Output Spaces

[![PyPi Latest Release](https://img.shields.io/pypi/v/libpecos)](https://img.shields.io/pypi/v/libpecos)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

PECOS is a versatile and modular machine learning (ML) framework for fast learning and inference on problems with large output spaces, such as extreme multi-label ranking (XMR) and large-scale retrieval.
PECOS' design is intentionally agnostic to the specific nature of the inputs and outputs as it is envisioned to be a general-purpose framework for multiple distinct applications.

Given an input, PECOS identifies a small set (10-100) of relevant outputs from amongst an extremely large (~100MM) candidate set and ranks these outputs in terms of relevance. 


### Features

#### Extreme Multi-label Ranking and Classification
* X-Linear ([`pecos.xmc.xlinear`](pecos/xmc/xlinear/README.md)): recursive linear models learning to traverse an input from the root of a hierarchical label tree to a few leaf node clusters, and return top-k relevant labels within the clusters as predictions. See more details in the [PECOS paper (Yu et al., 2020)](https://arxiv.org/pdf/2010.05878.pdf).
  + fast real-time inference in C++
  + can handle 100MM output space

* XR-Transformer ([`pecos.xmc.xtransformer`](pecos/xmc/xtransformer/README.md)): Transformer based XMC framework that fine-tunes pre-trained transformers recursively on multi-resolution objectives. It can be used to generate top-k relevant labels for a given instance or simply as a fine-tuning engine for task aware embeddings. See technical details in [XR-Transformer paper (Zhang et al., 2021)](https://arxiv.org/pdf/2110.00685.pdf).
  + easy to extend with many pre-trained Transformer models from [huggingface transformers](https://github.com/huggingface/transformers).
  + establishes the State-of-the-art on public XMC benchmarks.

* ANN Search with HNSW ([`pecos.ann.hnsw`](pecos/ann/hnsw/README.md)): a PECOS Approximated Nearest Neighbor (ANN) search module that implements the Hierarchical Navigable Small World Graphs (HNSW) algorithm ([`Malkov et al., TPAMI 2018`](https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf)).
  + Supports both sparse and dense input features
  +  SIMD optimization for both dense/sparse distance computation
  +  Supports thread-safe graph construction in parallel on multi-core shared memory machines
  +  Supports thread-safe Searchers to do inference in parallel, which reduces inference overhead


## Requirements and Installation

* Python (>=3.6, <=3.9)
* Pip (>=19.3)

See other dependencies in [`setup.py`](https://github.com/amzn/pecos/blob/mainline/setup.py#L135)
You should install PECOS in a [virtual environment](https://docs.python.org/3/library/venv.html).
If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

### Supporting Platforms
* Ubuntu 18.04 and 20.04
* Amazon Linux 2

### Installation from Wheel


PECOS can be installed using pip as follows:
```bash
python3 -m pip install libpecos
```

### Installation from Source

#### Prerequisite builder tools
* For Ubuntu (18.04, 20.04):
``` bash
sudo apt-get update && sudo apt-get install -y build-essential git python3 python3-distutils python3-venv
```
* For Amazon Linux 2 Image:
``` bash
sudo yum -y install python3 python3-devel python3-distutils python3-venv && sudo yum -y groupinstall 'Development Tools' 
```
One needs to install at least one BLAS library to compile PECOS, e.g. `OpenBLAS`:
* For Ubuntu (18.04, 20.04):
``` bash
sudo apt-get install -y libopenblas-dev
```
* For Amazon Linux 2 Image and AMI:
``` bash
sudo amazon-linux-extras install epel -y
sudo yum install openblas-devel -y
```

#### Install and develop locally
```bash
git clone https://github.com/amzn/pecos
cd pecos
python3 -m pip install --editable ./
```


## Quick Tour
To have a glimpse of how PECOS works, here is a quick tour of using PECOS API for the XMR problem.

### Toy Example
The eXtreme Multi-label Ranking (XMR) problem is defined by two matrices
* instance-to-feature matrix `X`, of shape `N by D` in [`SciPy CSR format`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)
* instance-to-label matrix `Y`, of shape `N by L` in [`SciPy CSR format`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)

Some toy data matrices are available in the [`tst-data`](https://github.com/amzn/pecos/tree/mainline/test/tst-data/xmc/xlinear) folder. 

PECOS constructs a hierarchical label tree and learns linear models recursively (e.g., XR-Linear):
```python
>>> from pecos.xmc.xlinear.model import XLinearModel
>>> from pecos.xmc import Indexer, LabelEmbeddingFactory

# Build hierarchical label tree and train a XR-Linear model
>>> label_feat = LabelEmbeddingFactory.create(Y, X)
>>> cluster_chain = Indexer.gen(label_feat)
>>> model = XLinearModel.train(X, Y, C=cluster_chain)
>>> model.save("./save-models")
```

After learning the model, we do prediction and evaluation 
```python
>>> from pecos.utils import smat_util
>>> Yt_pred = model.predict(Xt)
# print precision and recall at k=10
>>> print(smat_util.Metrics.generate(Yt, Yt_pred))
```

PECOS also offers optimized C++ implementation for fast real-time inference
```python
>>> model = XLinearModel.load("./save-models", is_predict_only=True)
>>> for i in range(X_tst.shape[0]):
>>>   y_tst_pred = model.predict(X_tst[i], threads=1)
```


## Citation

If you find PECOS useful, please consider citing the following paper:

* [PECOS: Prediction for Enormous and Correlated Output Spaces (Yu et al., JMLR 2022)](https://arxiv.org/pdf/2010.05878.pdf) [[bib]](./bibtex/yu2020pecos.bib)

Some papers from our group using PECOS:

* [FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search (Chen et al., ArXiv 2022)](https://arxiv.org/pdf/2206.11408.pdf) [[bib]](./bibtex/)

* [Relevance under the Iceberg: Reasonable Prediction for Extreme Multi-label Classification (Jiang et al., SIGIR 2022)](https://dl.acm.org/doi/abs/10.1145/3477495.3531767) [[bib]](./bibtex/)

* [Extreme Zero-Shot Learning for Extreme Text Classification (Xiong et al., NAACL 2022)](https://aclanthology.org/2022.naacl-main.399.pdf) [[bib]](./bibtex/)

* [Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction (Chien et al., ICLR 2022)](https://arxiv.org/pdf/2111.00064.pdf) [[bib]](./bibtex/chien2021node.bib)

* [Accelerating Inference for Sparse Extreme Multi-Label Ranking Trees (Etter et al., WWW 2022)](https://arxiv.org/pdf/2106.02697.pdf) [[bib]](./bibtex/etter2021accelerating.bib)

* [Fast Multi-Resolution Transformer Fine-tuning for Extreme Multi-label Text Classification (Zhang et al., NeurIPS 2021)](https://arxiv.org/pdf/2110.00685.pdf) [[bib]](./bibtex/zhang2021fast.bib)

* [Label Disentanglement in Partition-based Extreme Multilabel Classification (Liu et al., NeurIPS 2021)](https://arxiv.org/pdf/2106.12751.pdf) [[bib]](./bibtex/liu2021label.bib)

* [Enabling Efficiency-Precision Trade-offs for Label Trees in Extreme Classification (Baharav et al., CIKM 2021)](https://arxiv.org/pdf/2106.00730.pdf) [[bib]](./bibtex/baharav2021enabling.bib)

* [Extreme Multi-label Learning for Semantic Matching in Product Search (Chang et al., KDD 2021)](https://arxiv.org/pdf/2106.12657.pdf) [[bib]](./bibtex/chang2021extreme.bib)

* [Session-Aware Query Auto-completion using Extreme Multi-label Ranking (Yadav et al., KDD 2021)](https://arxiv.org/pdf/2012.07654.pdf)  [[bib]](./bibtex/yadav2021session.bib)

* [Top-k eXtreme Contextual Bandits with Arm Hierarchy (Sen et al., ICML 2021)](https://arxiv.org/pdf/2102.07800.pdf) [[bib]](./bibtex/sen2021top.bib)

* [Taming pretrained transformers for extreme multi-label text classification (Chang et al., KDD 2020)](https://arxiv.org/pdf/1905.02331.pdf) [[bib]](./bibtex/chang2020taming.bib)


## License

Copyright (2021) Amazon.com, Inc.
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


