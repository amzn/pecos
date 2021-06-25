# PECOS for text2text Applications

Given an input text, pecos.apps.text2text can generate a subset of labels relevant to this input from a fixed set of output labels.
The input should be a text sequence, while the output labels can be either text-based or symbol-based (although the symbols are usually represented in text format).
In text classification, for example, the input text can be a paragraph and the relevant labels can be categories that are tagged to this paragraph.
Another example is retrieval, where the input text can be natural question and the relevant labels can be paragraphs that contain answer span to that natural question.

## Getting started
### Usage
```bash
  > python3 -m pecos.apps.text2text.train --help
  > python3 -m pecos.apps.text2text.predict --help
  > python3 -m pecos.apps.text2text.evaluate --help
```


### Usage example: Multi-label Category Tagging for Web Documents
This toy example demonstrates how to run the training and prediction of PECOS text2text applications.
Note that we use utf-8 encoding for all text files.

First, consider the following input text file `training-data.txt`:
```
0,1,2<TAB>Alan Turing is widely considered to be the father of theoretical computer science and artificial intelligence.
0,2,3<TAB>Hinton was co-author of a highly cited paper published in 1986 that popularized the backpropagation algorithm for training multi-layer neural networks.
3,4,5<TAB>Hinton received the 2018 Turing Award, together with Yoshua Bengio and Yann LeCun, for their work on artificial intelligence and deep learning.
3,4,5<TAB>In 1989, Yann LeCun et al. applied the standard backpropagation algorithm on neural networks for hand digit recognition.
```
Each line contain two fields, separated by `<TAB>`, the former is relevant output label ids and the latter is the input text:
```
OUTPUT_ID1,OUTPUT_ID2,OUTPUT_ID3,...<TAB>INPUT_TEXT
```

The output ids are zero-based and correspond to the line numbers in the output label file.
In particular, the corresponding output label file `output-labels.txt` takes the format of:
```
Artificial intelligence researchers
Computability theorists
British computer scientists
Machine learning researchers
Turing Award laureates
Deep Learning
```
Each line in the file is a representation of an output label, categories in Wikipedia.


Now, we training the text2text model, which include feature vectorization and learning PECOS model.
We use toy TFIDF hyper-parameters in `config.json`, which can be found at [link](https://github.com/amzn/pecos/tree/mainline/pecos/utils/featurization/text).
```
python3 -m pecos.apps.text2text.train \
  --input-text-path ./training-data.txt \
  --vectorizer-config-path ./config.josn \
  --output-item-path ./output-labels.txt \
  --model-folder ./pecos-text2text-model
```
The models are saved into the `./pecos-text2text-model`.

For batch Predicting, user should give the input text file `test-data.txt`, which has the same format as `training-data.txt`:
```
python3 -m pecos.apps.text2text.predict \
  --input-text-path ./test-data.txt \
  --model-folder ./pecos-text2text-model \
  --predicted-output-item-path ./test-prediction.txt
```
The predictions are saved in the `./test-prediction.txt`.
Each line contains the generated output labels and score as a json-format dictionary for the corresponding input from the input file.

For Online Predicting (interactive mode)
```
python3 -m pecos.apps.text2text.predict --model-folder ./pecos-text2text-model
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

