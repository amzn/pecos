# Distributed PECOS eXtreme Multi-label Classification: XLinear

`pecos.distributed.xmc.xlinear` enables distributed training for PECOS XLinear model ([`pecos.xmc.xlinear`](../../../xmc/xlinear/README.md)).


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

#### Install MPI and mpi4py

For Ubuntu:
```bash
sudo apt-get update && sudo apt-get install python3 python3-dev mpich -y
```

For Amazon Linux 2:
```bash
sudo /usr/bin/yum install -y http://mirror.centos.org/centos/7/os/x86_64/Packages/mpich-3.2-3.2-2.el7.x86_64.rpm
sudo /usr/bin/yum install -y http://mirror.centos.org/centos/7/os/x86_64/Packages/mpich-3.2-devel-3.2-2.el7.x86_64.rpm
sudo touch /etc/profile.d/mpich.sh
echo 'export PATH=/usr/lib64/mpich-3.2/bin/:$PATH' | sudo tee /etc/profile.d/mpich.sh
sudo yum -y install python3 python3-devel
```
Logout and re-login to load the new `PATH`.

For all OS:
```bash
python3 -m pip install mpi4py
```

Check installation success:
```bash
mpicc -v
which mpiexec
mpiexec -n 5 python3 -m mpi4py.bench helloworld
```

#### Install PECOS
Please follow the [main guide for PECOS installation](https://github.com/amzn/pecos#requirements-and-installation).


### Workspace Setup
Make a workspace directory on your shared network disk:

```bash
cd <SHARED-NETWORK-DISK-PATH>
mkdir pecos-workspace && cd pecos-workspace
```
Create a `hostfile` consisting of all cluster machines' IP addresses on each line. **Replace all `<IP-machine-X>` by the real IP addresses**:
```bash
cat << EOF > hostfile
<IP-machine-1>
<IP-machine-2>
...
<IP-machine-N>
EOF
```
Test cluster connectivity:
```bash
mpiexec -f hostfile python3 -m mpi4py.bench helloworld
```
You should receive N "Hello, World!" messages from each machine of the cluster.

## Getting started

### Basic Command-line Usage
On the main node:
```bash
mpiexec -f hostfile -n ${NUM_MACHINE} \
python3 -m pecos.distributed.xmc.xlinear.train \
-x ${X_PATH} -y ${Y_PATH} -m ${MODEL_DIR}
```
where:
* `NUM_MACHINE` is the number of machines used for distributed training, should <= number of machines in the cluster.
* `X_PATH` is the paths to the CSR npz or Row-majored npy files of the training feature matrices with shape `(N, d)`.
* `Y_PATH` is the paths to the CSR npz files of the training label matrices with shape `(N, L)`.
* `MODEL_DIR` is the path to the model folder where the trained model will be saved to, will be created if not exist.

All the paths: `X_PATH`, `Y_PATH` and `MODEL_DIR` should be accessible by all machines on a shared network disk.

For detailed parameters list, please refer to:
```bash
python3 -m pecos.distributed.xmc.xlinear.train --help
```

After training the model, you can use [`pecos.xmc.xlinear.predict`](../../../xmc/xlinear/README.md#basic-command-line-usage) to do inferences.

The model trained by distributed `pecos.distributed.xmc.xlinear.train` is almost identical to the model trained by single-box `pecos.xmc.xlinear.train`, if not considering the randomness introduced by sub-models. Accordingly, the inference results from distributed and single-box models should also be very similar.

### Example: Distributed XLinear Model Training on eurlex-4k Data
Make sure you have setup all above hardware, software and workspace prerequisites.

Prepare `eurlex-4k` data:
```bash
cd <SHARED-NETWORK-DISK-PATH>/pecos-workspace

wget https://archive.org/download/pecos-dataset/xmc-base/eurlex-4k.tar.gz
tar -zxvf eurlex-4k.tar.gz
```

Train:
```bash
mpiexec -f hostfile \
python3 -m pecos.distributed.xmc.xlinear.train \
-x ./xmc-base/eurlex-4k/tfidf-attnxml/X.trn.npz \
-y ./xmc-base/eurlex-4k/Y.trn.npz \
--nr-splits 2 -b 50 -k 100 -m eurlex_4k_model \
--min-n-sub-tree 16 -t 0.1 \
--meta-label-embedding-method pii \
--sub-label-embedding-method pifa \
--verbose-level 3
```

Predict:
```bash
python3 -m pecos.xmc.xlinear.predict \
-x ./xmc-base/eurlex-4k/tfidf-attnxml/X.tst.npz \
-y ./xmc-base/eurlex-4k/Y.tst.npz \
-m ./eurlex_4k_model
```

## Appendix: Distributed XLinear Training Q&A
Extra distributed training parameters are available for better time and memory efficiency.

**Q: What should I do if meta-clustering is too slow, or consume too much memory on main node?**

A: Use `PII` embedding for meta-tree label embedding to accelerate the clustering:
```bash
mpiexec -f hostfile -n ${NUM_MACHINE} \
python3 -m pecos.distributed.xmc.xlinear.train \
-x ${X_PATH} -y ${Y_PATH} -m ${MODEL_DIR} \
--meta-label-embedding-method pii
```
This would produce slightly different models.

**Q: What should I do if training throws `MPIBufferSizeExceedError` exceptions?**

A: Use larger `min-n-sub-tree`. It will not only reduces the size of sub-models to pass back to main node but also gain better runtime workload balance on worker nodes:
```bash
# Set MIN_N_SUB_TREE to a larger number
mpiexec -f hostfile -n ${NUM_MACHINE} \
python3 -m pecos.distributed.xmc.xlinear.train \
-x ${X_PATH} -y ${Y_PATH} -m ${MODEL_DIR} \
--min-n-sub-tree ${MIN_N_SUB_TREE}
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
