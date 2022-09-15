

## Notice
- Currently we only support L2 distance with 4 Bits Product Quantization. 
- We are working on extending to angular and ip distance measures.

## Install Prerequisite

To run this project, prerequisite is the same as building PECOS.

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

## Prepare Data

Get the exemplar sift-128-eucldiean dataset

```bash 
wget https://archive.org/download/pecos-dataset/ann-benchmarks/sift-euclidean-128.tar.gz
``` 

Extract the dataset

```bash 
tar -xf sift-euclidean-128.tar.gz
``` 

The prepared dataset consists of 3 .npy files : X.trn.npy (training data),  X.tst.npy (testing data) and Y.tst.npy (10 Nearest neighbors in training data of test data).

## Compile the source code

```bash 
Make clean go
``` 

a runnable named "go" will be generated.

## Running the compiled runnable

the runnable take arguments in the following form :
```bash 
./go data_folder model_folder space M efC #threads efs num_rerank sub_dimension
``` 

data_folder is the place where 3 npy files stored. model_folder is the place to store the trained model. If a saved model is found, we will load the model instead of training a new one. space denotes the distance measure to use. Currently, we only support L2. M is the maximal edge connection used in HNSW. efC is the Maximal connecting edges during construction used in HNSW. #threads is the number of threads to build the graph. Up to now, these hypaer-parameters relate to the construction, and they will be used to name the trained model directory. efs is the search queue size in the inference step. num_rerank is the number of points in the queue that we will further rerank again using original features instead of quantized distance. sub_dimension is the dimension of each subspace in Product Quantization. If sub_dimension is set to 0, it will use default scheme. That is, if original data dimension <= 400, we use sub_dimension == 1, otherwise we use sub_dimension == 2.
 
Here, we provide an example of command executing the runnable : 

```bash 
./go sift-euclidean-128 sift-euclidean-128 l2 8 500 24 10 10 0
``` 

## Experiment

The compiled source code in example.cpp already repeats the inference 10 times. So to evaluate under ann-benchmark protocol, we could simply use python to iterate hyper-parameters and record done results. 

```bash 
python run.py
``` 
