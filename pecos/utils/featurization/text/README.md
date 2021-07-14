# PECOS for Text Preprocessing/Vectorizing

Given an input text, pecos.utils.featurization.text.preprocess generate numerical vectors from text.
The input file should be a text sequence for each line.

## Getting started
### Usage
Build a preprocessor
```
  > python3 -m pecos.utils.featurization.text.preprocess build --help
```
Generate numerical vectors from text via a preprocessor
```
  > python3 -m pecos.utils.featurization.text.preprocess run --help
```

### Usage example: TFIDF featurization
This toy example demonstrates how to construct n-gram TFIDF features containing word unigrams, word bigrams, and character trigrams.
Note that each of the n-gram feature can have different hyper-parameters such as `max_feature`, `max_df`, and more.
For an complete list of hyper-parameters to build n-gram TFIDF, see [link](https://github.com/amzn/pecos/blob/mainline/pecos/core/base.py#L1174).

Consider the following toy input file `input.txt` (e.g., each line is an input instance of raw text):
```
Alan Turing is widely considered to be the father of theoretical computer science and artificial intelligence.
Hinton was co-author of a highly cited paper published in 1986 that popularized the backpropagation algorithm for training multi-layer neural networks.
Hinton received the 2018 Turing Award, together with Yoshua Bengio and Yann LeCun, for their work on artificial intelligence and deep learning.
In 1989, Yann LeCun et al. applied the standard backpropagation algorithm on neural networks for hand digit recognition.
```

Here is an toy json file `config.json` that defines the n-gram TFIDF hyper-parameters.
```
{
  "type": "tfidf",
  "kwargs": {
    "base_vect_configs": [
      {
        "ngram_range": [1, 1],
        "max_df_ratio": 0.98,
        "analyzer": "word"
      },
      {
        "ngram_range": [2, 2],
        "max_df_ratio": 0.98,
        "analyzer": "word"
      },
      {
        "ngram_range": [3, 3],
        "max_df_ratio": 0.98,
        "analyzer": "char_wb"
      }
    ]
  }
}
```

***WARNING***: Users need to properly set `max_feature` (e.g., hundred of thousands or millions) based on the corpus size and downstream tasks!


We first build the TF-IDF vectorizer model via this command line
```
python3 -m pecos.utils.featurization.text.preprocess build \
  --text-pos 0 \
  --input-text-path ./input.txt \
  --vectorizer-config-path ./config.json \
  --output-model-folder ./tfidf-model
```
The TF-IDF model is built and saved in the `./tfidf-model` folder.
Now we are ready to create TF-IDF features via this command line:
```
python3 -m pecos.utils.featurization.text.preprocess run \
  --text-pos 0 \
  --input-preprocessor-folder ./tfidf-model \
  --input-text-path ./input.txt \
  --output-inst-path ./input.tfidf.npz
```
Finally, the sparse TF-IDF feature matrix is saved in the `./input.tfidf.npz` file.
The matrix is stored in the Sparse CSR format, see more details at [link](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).

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

