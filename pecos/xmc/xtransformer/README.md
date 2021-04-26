# PECOS eXtreme Multi-label Classification: XTransformer

`pecos.xmc.xtransformer` is a PECOS module for extreme multi-label classification and ranking using transformer models.
It takes both numerical vectors and instance text as the input and outputs relevant labels for the input vectors.
GPUs with CUDA support is recommended to achieve the best performance of the module.

## Getting started

### Command line usage
Basic Training command:
```bash
  > python3 -m pecos.xmc.xtransformer.train --trn-text-path ${X_txt_path} \
                                            --trn-feat-path ${X_path}  \
                                            --trn-label-path ${Y_path} \
                                            --model-dir ${model_dir}
```
Predict on training dataset
```bash
  > python3 -m pecos.xmc.xtransformer.predict --feat-path ${X_path} \
                                              --text-path ${X_txt_path} \
                                              --model-folder ${model_dir} \
                                              --output-dir ${Yp_path}
```
where
* `X_txt_path` is the path to the input text file of the training instances. Should be a text file with `N` lines where each line is the text feature of the corresponding training instance.
* `X_path` is the path to the CSR npz or Row-majored npy file of the training feature matrices with shape `(N, d)`.
* `Y_path` is the path to the CSR npz file of the training label matrices with shape `(N, L)`.
* `model_dir` is the path to the model folder where the trained model will be saved to, will be created if not exist.
* `Yp_path` is the path to save the prediction label matrix with shape `(N, L)`



To get the evaluation metrics for top-10 predictions
```bash
  > python3 -m pecos.xmc.xlinear.evaluate -y ${Y_path} -p ${Yp_path} -k 10
```

For detailed usage, please refer to
```bash
  > python3 -m pecos.xmc.xtransformer.train --help
  > python3 -m pecos.xmc.xtransformer.predict --help
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

