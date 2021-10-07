# PECOS ANN Search with HNSW Algorithm

`pecos.ann.hnsw` is a PECOS Approximated Nearest Neighbor (ANN) search module that implements the Hierarchical Navigable Small World Graphs (HNSW) algorithm (Malkov et al., TPAMI 2018).

## featured Designs
* Supports both **sparse** and **dense** input features
* **SIMD optimization** for both dense/sparse distance computation
* Supports **thread-safe** graph construction in parallel on multi-core shared memory machines
* Supports **thread-safe** Searchers to do inference in parallel, which reduces inference overhead

## Python API examples

#### Prepare data
First, let's create the database matrix `X_trn` and query matrix `X_tst`. We will use dense numpy matrices in this illustration. But keep in mind that we also support sparse input features!
```python
import numpy as np
X_trn = np.random.randn(10000, 100).astype(np.float32)
X_tst = np.random.randn(1000, 100).astype(np.float32)
```
Note that the data type needed to be `np.float32`.   

#### HNSW Training
Train the HNSW model (i.e., building the graph-based indexing data structure) with maximum number of threads available on your machine (`threads=0`):
```python
from pecos.ann.hnsw import HNSW
train_params = HNSW.TrainParams(M=32, efC=300, metric_type="ip", threads=-1)
model = HNSW.train(X_trn, train_params=train_params)
```

#### HNSW Save and Load
After training, we can save the model to file and re-load
```python
model_folder = "./tmp-hsnw-model"
model.save(model_folder)
del model
model = HNSW.load(model_folder)
```

#### HNSW Searcher
Next, we initialize multiple searchers for the inference stage. The searchers will pre-allocate some intermediate variables later to be used by HNSW graph search (e.g., which nodes being visited, priority queues storing the candidates, etc).
``` python
# here we would like to FOUR threads to do parallel inference
searchers = model.searchers_create(num_searcher=4)
```

#### HNSW Inference
Finally, we conduct ANN inference by inputing searchers to the HNSW model.
```python
pred_params = HNSW.PredParams(efS=100, topk=10)
indices, distances = model.predict(X_tst, pred_params=pred_params, searchers=searchers, ret_csr=False)
```
Alternatively, it is also feasible to do inference without pre-allocating searchers, which may have larger overhead since it will **re-allocate** intermediate graph-searhing variables for each query matrix `X_tst`.
```python
pred_params.threads = 2
indices, distances = model.predict(X_tst, pred_params=pred_params, ret_csr=False)
```
When `ret_csr=True`, the prediction function will return a single csr matrix that combines the indices and distances numpy array.
