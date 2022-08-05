# Distributed PECOS eXtreme Multi-label Classification: XR-Transformer

`pecos.distributed.xmc.xtransformer` enables distributed fine-tuning for PECOS XR-Transformer model ([`pecos.xmc.xtransformer`](../../../xmc/xtransformer/README.md)).

Note that this module only supports fine-tuning of XR-Transformer encoders, not hierarchical labal tree building or linear ranker training.

## Prerequisites

### Hardware

You need the following hardware to train distributed PECOS:

* Cluster of machines connected by network which can password-less SSH to each other.
  * IP address of every machine in the cluster is known.
* Shared network disk mounted on all machines.
  * For accessing data and saving trained models.

Currently we do not provide guides to setup a cluster but we might provide in the future. For now, please refer to your organization's hardware management for help.


### Software

Install the following software on **every** machine of your cluster:

#### Install PECOS
Please follow the [main guide for PECOS installation](https://github.com/amzn/pecos#requirements-and-installation).

#### Install DeepSpeed

```bash
DS_BUILD_OPS=1 DS_BUILD_SPARSE_ATTN=0 pip3 install deepspeed
```

### Workspace Setup
Make a workspace directory on your shared network disk:

```bash
cd <SHARED-NETWORK-DISK-PATH>
mkdir pecos-workspace && cd pecos-workspace
```
Create a `hostfile` consisting of all cluster machines' IP addresses and number of GPUs on each line:
```bash
cat << EOF > hostfile
<IP-machine-1> slots=<n_1>
<IP-machine-2> slots=<n_2>
...
<IP-machine-N> slots=<n_N>
EOF
```
Test cluster connectivity:
```bash
 deepspeed --hostfile machinefile --module pecos.distributed.diagnostic_tools.deepspeed_comm --timeout 60 --shared-workdir .
```

## Getting started

### Basic Command-line Usage

The distributed training CLI `pecos.distributed.xmc.xtransformer.train` is similar to that of `pecos.xmc.xtransformer`.

There are several additional things to note:

* **Have the Hierarchical Label Tree (HLT) ready**: The distributed training module will not automatically construct HLT for you. So you need to supply existing label clustering with `--code-path`, otherwise the module will fall back to One-Versus-All fine-tuning (not applicable for large label space).
* **Do not accept instance numerical feature:** The training of sparse+dense concat linear models are disabled.

You can generate a `.json` file with all of the parameters that you can edit and fill in.
```bash
  > python3 -m pecos.distributed.xmc.xtransformer.train --generate-params-skeleton &> params.json
```

After editing the `params.json` file, you can do training via:

```bash
python3 -m pecos.distributed.xmc.xtransformer.train \
                                --trn-text-path ${T_path} \
                                --trn-label-path ${Y_path} \
                                --code-path ${C_path} \
                                --model-dir ${model_dir} \
                                --params-path params.json
```
where
* `T_path` is the path to the input text file of the training instances. Text file with `N` lines where each line is the text feature of the corresponding training instance.
* `Y_path` is the path to the CSR npz file of the training label matrices with shape `(N, L)`.
* `C_path` is the path to the CSC npz file of the clustering matrix with shape `(N, K)`, where `K` is the number of leaf clusters.
* `model_dir` is the path to the model folder where the trained model will be saved to, will be created if not exist.

After fine-tuning, you can generate the instance embeddings via:
```bash
deepspeed --hostfile machinefile --module pecos.distributed.xmc.xtransformer.encode -t ${T_path} -m ${model_dir} -o ${result_dir}
```
where `result_dir` is the folder (under your shared network disk) in which the embeddings will be written.
To handle large data, the embeddings will be written in shards to `${result_dir}/X.emb.0.npy`, `${result_dir}/X.emb.1.npy` ... `${result_dir}/X.emb.[WORLD_SIZE].npy`.

For small data, you can also use the single node XR-Transformer module
```
python3 -m pecos.xmc.xtransformer.encode -t ${T_path} -m ${model_dir} -o ${result_path}
```
***

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
