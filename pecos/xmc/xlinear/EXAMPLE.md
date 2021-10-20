# PECOS XR-Linear Tutorials on Text Applications

In this example, we introduce a typical pipeline of using XR-Linear model on raw text input/output data.
We will create input/label embeddings from the TF-IDF vectorizer.
Note that XR-Linear can work with any vectorized input matrices, where TF-IDF is just an example here for the ease of demonstration.


## 1. Input data format
`pecos.xmc.xlinear` modules for text applications in general requires the folowing input data files
* `X.trn.txt`: each line is an instance's raw text from the training set, with `n_trn` number of lines
* `X.tst.txt`: each line is an instance's raw text from the test set, with `n_tst` number of lines
* `Y.trn.npz`: a `scipy.sparse.csr_matrix` of shape `[n_trn, L]`, storing the positive instance-label pairs from the training set
* `Y.tst.npz`: a `scipy.sparse.csr_matrix` of shape `[n_tst, L]`, storing the positive instance-label pairs from the test set
* `Z.all.txt` (optional): each line is a label's raw text from the label space with `L` number of line. The line number is the `label_id` that corresponds to columns of `Y.trn.npz` and `Y.tst.npz`

An example XMC dataset can be downloaded at [link](https://archive.org/download/pecos-dataset/xmc-base/eurlex-4k/),
where you can rename the `output-items.txt` to be `Z.all.txt` 
```bash
mkdir ./toy-example
cd ./toy-example
wget https://archive.org/download/pecos-dataset/xmc-base/eurlex-4k.tar.gz
tar -zxvf eurlex-4k.tar.gz
ln -S ./xmc-base/eurlex-4k/X.trn.txt ./X.trn.txt
ln -S ./xmc-base/eurlex-4k/X.tst.txt ./X.tst.txt
ln -S ./xmc-base/eurlex-4k/Y.trn.npz ./Y.trn.npz
ln -S ./xmc-base/eurlex-4k/Y.tst.npz ./Y.tst.npz
ln -S ./xmc-base/eurlex-4k/output-items.txt ./Z.all.txt
```
From now on, we assume the working directory is under `./toy-example/` 


## 2. TF-IDF featurization
(1) Let's first concatenate `X.trn.txt` and `Z.all.txt` into a joint corpus for fitting TF-IDF model:
```bash
cat ./X.trn.txt Z.all.txt > corpus.txt
```

(2) We will use this `config.json` as our TF-IDF model hyper-parameters:
```json
{
  "type": "tfidf",
  "kwargs": {
    "base_vect_configs": [
      {
        "ngram_range": [1, 1],
        "min_df_cnt": 5,
        "max_df_ratio": 0.98,
        "truncate_length": 256,
        "analyzer": "word" 
      }
    ]
  }
}
```
See more details in [`pecos.utils.featurization.text`](https://github.com/amzn/pecos/tree/mainline/pecos/utils/featurization/text)

(3) Next we fit the TF-IDF models
```python
python3 -m pecos.utils.featurization.text.preprocess build \
    --input-text-path ./corpus.txt \
    --vectorizer-config-path ./config.json \
    --output-model-folder ./tfidf-model \
    --text-pos 0
```

(4) We then predict the TF-IDF matrix for all text files
```bash
# For X.trn.txt
python3 -m pecos.utils.featurization.text.preprocess run \
    --input-preprocessor-folder ./tfidf-model \
    --input-text-path ./X.trn.txt \
    --output-inst-path ./X.trn.npz \
    --text-pos 0
# For X.tst.txt
python3 -m pecos.utils.featurization.text.preprocess run \
    --input-preprocessor-folder ./tfidf-model \
    --input-text-path ./X.tst.txt \
    --output-inst-path ./X.tst.npz \
    --text-pos 0
# For Z.all.txt
python3 -m pecos.utils.featurization.text.preprocess run \
    --input-preprocessor-folder ./tfidf-model \
    --input-text-path ./Z.all.txt \
    --output-inst-path ./Z.all.npz \
    --text-pos 0
```
We should now see there are `X.trn.npz`, `X.tst.npz`, `Z.all.npz` files under your current folders.


## 3. PIFA label embedding
The below two sections (Sec 3 and Sec 4) are done in Python Interactive enviroments (e.g., ipython, notebook) for demonstration purpose.

(1) If we just want the default PIFA label embedding,
```python
from pecos.xmc import Indexer, LabelEmbeddingFactory
from pecos.utils import smat_util
X_trn = smat_util.load_matrix("./X.trn.npz")
Y_trn = smat_util.load_matrix("./Y.trn.npz")
Z_pifa = LabelEmbeddingFactory.create(Y_trn, X_trn, method="pifa")
```

(2) If we want to have PIFA concatenating with additional label embeddings (e.g., `Z.all.npz`),
```python
label_feat_tfidf = smat_util.load_matrix("./Z.all.npz")
Z_pifa_concat = LabelEmbeddingFactory.create(Y_trn, X_trn, Z=label_feat_tfidf, method="pifa_lf_concat")
```

(3) Finally, we obtain the `cluster_chain` of hierarchical label tree (HLT) using `nr_splits=32`
```python
cluster_chain = Indexer.gen(Z_pifa_concat, indexer_type="hierarchicalkmeans", nr_splits=32)
```

## 4. XR-Linear Model
(1) train XR-Linear Models
```python
from pecos.xmc.xlinear.model import XLinearModel
xlm = XLinearModel.train(X_trn, Y_trn, C=cluster_chain)
```
Note that if `cluster_chain=None`, then `XLinearModel` will train a flat one-versus-all model, which may take very long time!

(2) Save and load model to/from the disk. Note that loading with `is_predict_only=True` will result in faster prediction speed but will disable you from further modify the model such as pruning. See `XLinearModel.load` for details.
```python
xlm.save("model")
xlm = XLinearModel.load("model", is_predict_only=False)
```

(3) predict on the test set
```python
X_tst = smat_util.load_matrix("./X.tst.npz")
Y_tst = smat_util.load_matrix("./Y.tst.npz")
Y_pred = xlm.predict(X_tst)
```

(4) evaluate precision/recall
```python
metric = smat_util.Metrics.generate(Y_tst, Y_pred, topk=10)
print(metric)
```
