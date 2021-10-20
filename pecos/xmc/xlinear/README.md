# PECOS eXtreme Multi-label Classification: XLinear

`pecos.xmc.xlinear` is a PECOS module for extreme multi-label classification and ranking.
It takes sparse or dense numerical vectors as the input and outputs relevant labels for the input vectors.

## Getting started
### Basic Command line usage

Basic Training and predicting:
```bash
  > python3 -m pecos.xmc.xlinear.train -x ${X_path} -y ${Y_path} -m ${model_dir}
  > python3 -m pecos.xmc.xlinear.predict -x ${Xt_path} -m ${model_dir} -o ${Yp_path}
```

To get the evaluation metrics for top-10 predictions
```bash
  > python3 -m pecos.xmc.xlinear.evaluate -y ${Yt_path} -p ${Yp_path} -k 10
```
where
* `X_path` and `Xt_path` are the paths to the CSR npz or Row-majored npy files of the training/test feature matrices with shape `(N, d)` and `(Nt, d)`.
* `Y_path` and `Yt_path` are the paths to the CSR npz files of the training/test label matrices with shape `(N, L)` and `(Nt, L)`.
* `model_dir` is the path to the model folder where the trained model will be saved to, will be created if not exist.
* `Yp_path` is the path to save the prediction label matrix with shape `(Nt, L)`

For detailed usage, please refer to
```bash
  > python3 -m pecos.xmc.xlinear.train --help
  > python3 -m pecos.xmc.xlinear.predict --help
  > python3 -m pecos.xmc.xlinear.evaluate --help
```

### Advanced Usage: Give parameters via a JSON file
`pecos.xmc.xlinear.train` supports accepting training and predicting parameters from an input JSON file.
Moreover, `python3 -m pecos.xmc.xlinear.train` helpfully provide the option to generate all parameters in JSON format to stdout.

You can generate a `.json` file with all of the parameters that you can edit and fill in.
```bash
  > python3 -m pecos.xmc.xlinear.train --generate-params-skeleton &> params.json
```
After editing the `params.json` file, you can do training via:
```bash
  > python3 -m pecos.xmc.xlinear.train \
	-x ${X_path} -y ${Y_path} -m ${model_dir} \
	--params-path params.json
```

### Python Example
This toy example demonstrates how to train and predict with PECOS XLinear module using Python API.

Loading the training data
```python
from pecos.xmc.xlinear.model import XLinearModel
# load training feature with shape=(N, d) and label with shape=(N, L)
X = XLinearModel.load_feature_matrix("test/tst-data/xmc/xlinear/X.npz")
Y = XLinearModel.load_label_matrix("test/tst-data/xmc/xlinear/Y.npz", for_training=True)
```
Train the one-versus-all model
```python
xlm = XLinearModel.train(X, Y)
```
(optional) You can construct label indexing before training the model by using
`pecos.xmc.Indexer` and `pecos.xmc.LabelEmbeddingFactory` modules.
This creates label-hierarchical tree and allows O(log(L)) time
complexity in prediction.
For more information, please refer to our [tech doc](https://arxiv.org/abs/2010.05878).
```python
from pecos.xmc import Indexer, LabelEmbeddingFactory
# construct label feature for clustering
label_feat = LabelEmbeddingFactory.create(Y, X, method="pifa")
# generate label indexing
cluster_chain = Indexer.gen(label_feat, indexer_type="hierarchicalkmeans")

xlm = XLinearModel.train(X, Y, C=cluster_chain)
```

Save and load model to/from the disk.
Note that loading with `is_predict_only=True` will result in faster prediction speed but will disable you from further modify the model such as pruning.
See `XLinearModel.load` for details.
```python
xlm.save("model")
xlm = XLinearModel.load("model", is_predict_only=False)
```

Predict on test data
```python
# load test data with shape=(Nt, d)
Xt = XLinearModel.load_feature_matrix("test/tst-data/xmc/xlinear/Xt.npz")
# Batch prediction, Y_pred is a csr_matrix with shape=(Nt, L)
Y_pred = xlm.predict(Xt)
```
Evaluate prediction result
```python
from pecos.utils import smat_util
# load test labels with shape=(Nt, L)
Yt = XLinearModel.load_label_matrix("test/tst-data/xmc/xlinear/Yt.npz")
metric = smat_util.Metrics.generate(Yt, Yt_pred, topk=10)
print(metric)
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

