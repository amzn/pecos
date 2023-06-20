# Factorization Machine XMC
Factorization machines (FM) are models that factorized input features into smaller dimensions for retrieval systems.
The factorization enables modeling of cross-terms between input features, which isn't possible with purely inner-product based models.
FM are particularly useful when the input feature dimension is sparse (e.g. TD-IDF embedding) and cross-terms between feature dimensions contain relevant information to perform prediction.
This library implements efficient training of factorization machines for extreme multilabel classification.
The optmization algorithm is AdaGrad with L2 regularization, with the option of parallelization with HogWild!.

## Install
```
git clone -b v0.4.0 https://github.com/amzn/pecos.git
cd pecos/example/fm
make all
```
You might need to specify path to OpenBlas with `-L[/openblas_dir/lib]`

## Training
```
./fm_train $PARAMS $Q_TRN_PATH $QP_PAIR_TRN_PATH $Q_TST_PATH $QP_PAIR_TST_PATH $P_PATH $MODEL_PATH
```
Trained model will be saved in `MODEL_PATH`.

### Parameters
* `-t`: Number of training epochs (int, default 10)
* `-k`: Number of factorized dimensions (int, default 4)
* `-l`: L2 regularization factor (float, default 2e-5)
* `-r`: Adagrad learning rate (float, defalt 2e-2)
* `--n_threads`: Number of threads for parallel training when OpenMP is enabled. Current implementation adopts [HogWild!](https://arxiv.org/abs/1106.5730) lock-free, parallelized gradient descent. (int, default 1)
* `--auto-stop`: Flag to early-stop training when testing loss starts to increase. Factorization machines are know to overfit and is not fixable by increasing L2 regularization.
* `--factorized`: Flag to perform gradient update per training sample, as opposed to per factor. Improves the training speed from $O(d^2k)$ to $O(dk)$ where $d$ is the number of input feature dimensions and $k$ is the number of factors (see more details [here](./FM_O_dk.pdf)). The improvement in training speed comes as no cost of trained model performance. *Highly recommend turning on.*
* `--dense`: Flag to specify whether input features in `X_TRN_PATH`, `X_TST_PATH`, and `Z_PATH` are dense (expect Numpy [`.npy`](https://numpy.org/doc/stable/reference/routines.io.html)) or sparse (expect Scipy spare [`.npz`](https://docs.scipy.org/doc/scipy/reference/sparse.html)).

### Training and testing data paths
* `Q_TRN_PATH`: Path to training query features file ( `npy` or `npz`) storing a $n_q \times d_q$ matrix, where $n_q$ is the number of unique training queries and $d_q$ is the dimension of query features.
* `QP_PAIR_TRN_PATH`: Path to training (query, product) pairs (`npz`) storing a $n_q \times n_p$ sparse binary matrix $M$, where $n_q$ is the number of unique training queries and $n_p$ is the number of unique products. $M[i, j] == 1$ indicates that the $i$-th query and $j$-th product is a positive pair.
* `Q_TST_PATH`: Path to testing query features file ( `npy` or `npz`) storing a $n_q \times d_q$ matrix, where $n_q$ is the number of unique testing queries and $d_q$ is the dimension of query features.
* `QP_PAIR_TST_PATH`: Path to testing (query, product) pairs (`npz`) storing a $n_q \times n_p$ sparse binary matrix $M$, where $n_q$ is the number of unique testing queries and $n_p$ is the number of unique products. $M[i, j] == 1$ indicates that the $i$-th query and $j$-th product is a positive pair.
* `P_PATH`: Path to product features file ( `npy` or `npz`) storing a $n_p \times d_p$ matrix, where $n_p$ is the number of unique products and $d_p$ is the dimension of product features.

## FM to shifted inner product (SIP)
For better integration with existing inner-product-search based retrieval systems, factorization machines can embed queries and products into vectors such that inner product with the vectors recovers the factorization machine prediction.
To do so, simply run
```
./fm_embgen $MODEL_PATH $Q_PATH $P_PATH $SAVE_DIR
```
### Arguments
* `MODEL_PATH`: Path to saved model.
* `Q_PATH`: Path to query features file ( `npy` or `npz`) storing a $n_q \times d_q$ matrix, where $n_q$ is the number of unique queries and $d_q$ is the dimension of query features.
* `P_PATH`: Path to product features file ( `npy` or `npz`) storing a $n_p \times d_p$ matrix, where $n_p$ is the number of unique products and $d_p$ is the dimension of product features.
* `SAVE_DIR`: Directory to save the embeddings.

### Importing embeddings into Python
Users may want to consume the binary embeddings in Python. To convert the embeddings to Numpy arrays, run
```
python binary_emb_to_npy.py [EMB_DIR]
```

