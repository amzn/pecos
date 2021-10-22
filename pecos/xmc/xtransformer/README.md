# PECOS eXtreme Multi-label Classification: XR-Transformer

`pecos.xmc.xtransformer` is a PECOS module for extreme multi-label classification and ranking using transformer models.
It takes both numerical vectors and instance text as the input and outputs relevant labels for the inputs.
GPUs with CUDA support is recommended to achieve the best performance of the module.


## Getting started

### Basic Command line usage

Basic Training and predicting:
```bash
  > python3 -m pecos.xmc.xtransformer.train -t ${T_path} -x ${X_path} -y ${Y_path} -m ${model_dir}
  > python3 -m pecos.xmc.xtransformer.predict -t ${Tt_path} -x ${Xt_path} -m ${model_dir} -o ${Pt_path}
```

To get the evaluation metrics for top-10 predictions:
```bash
  > python3 -m pecos.xmc.xlinear.evaluate -y ${Yt_path} -p ${Pt_path} -k 10
```

You can also get the fine-tuned text embeddings via:
```bash
  > python3 -m pecos.xmc.xtransformer.encode -t ${Tt_path} -m ${model_dir} -o ${Emb_path}
```

where
* `T_path` and `Tt_path` are the paths to the input text file of the training/test instances. Text files with `N`/`Nt` lines where each line is the text feature of the corresponding training/test instance.
* `X_path` and `Xt_path` are the paths to the CSR npz or Row-majored npy files of the training/test feature matrices with shape `(N, d)` and `(Nt, d)`.
  * Note that you can use the PECOS built in text preprocessing/vectorizing module [pecos.utils.featurization.text.preprocess](https://github.com/amzn/pecos/tree/mainline/pecos/utils/featurization/text) to generate numerical features if you do not already have them.
  * Usually providing instance numerical features is recommended. However, if you choose not to provide numerical features, `code-path` or `label-feat-path` is required to generate the hierarchical label trees.
* `Y_path` and `Yt_path` are the paths to the CSR npz files of the training/test label matrices with shape `(N, L)` and `(Nt, L)`.
* `model_dir` is the path to the model folder where the trained model will be saved to, will be created if not exist.
* `Pt_path` is the path to save the prediction label matrix with shape `(Nt, L)`
* `Emb_path` is the path to save the prediction label matrix with shape `(Nt, hidden_dim)`

For detailed usage, please refer to
```bash
  > python3 -m pecos.xmc.xtransformer.train --help
  > python3 -m pecos.xmc.xtransformer.predict --help
  > python3 -m pecos.xmc.xtransformer.encode --help
```

### Advanced Usage: Give parameters via a JSON file
`pecos.xmc.xtransformer` supports accepting training and predicting parameters from an input JSON file.
Moreover, `python3 -m pecos.xmc.xtransformer.train` helpfully provide the option to generate all parameters in JSON format to stdout.

You can generate a `.json` file with all of the parameters that you can edit and fill in.
```bash
  > python3 -m pecos.xmc.xtransformer.train --generate-params-skeleton &> params.json
```
After editing the `params.json` file, you can do training via:
```bash
  > python3 -m pecos.xmc.xtransformer.train -t ${T_path} -x ${X_path} -y ${Y_path} -m ${model_dir} \
	--params-path params.json
```

### Python Example
This toy example demonstrates how to train and predict with PECOS XR-Transformer module using Python API.

Loading the training data
```python
# load training numerical feature with shape=(N, d) and label with shape=(N, L)
from pecos.utils import smat_util
X = smat_util.load_matrix("test/tst-data/xmc/xtransformer/train_feat.npz")
Y = smat_util.load_matrix("test/tst-data/xmc/xtransformer/train_label.npz")
# load training text features
from pecos.utils.featurization.text.preprocess import Preprocessor
text = Preprocessor.load_data_from_file("test/tst-data/xmc/xtransformer/train.txt", text_pos=0)["corpus"]
```
Train the XR-Transformer model
```python
from pecos.xmc.xtransformer.model import XTransformer
from pecos.xmc.xtransformer.module import MLProblemWithText
prob = MLProblemWithText(text, Y, X_feat=X)
xtf = XTransformer.train(prob)
```

Save and load model to/from the disk.
```python
xtf.save("model")
xtf = XTransformer.load("model")
```

Predict
```python
# P is a csr_matrix with shape=(N, L)
P = xtf.predict(text, X)
```
Evaluate prediction result
```python
metric = smat_util.Metrics.generate(Y, P, topk=10)
print(metric)
```

You can also get the text embeddings using the fine-tuned transformer model via:
```python
# X_emb is a ndarray with shape=(N, hidden_dim)
X_emb = xtf.encode(text)
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

