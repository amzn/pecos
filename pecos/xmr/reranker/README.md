# PECOS XMR Reranker

This is a reranker for the PECOS XMR model. It is based on huggingface's transformers library. The reranker can be run in both
single process and distributed mode. It is based on the paper [Fine-Tuning LLaMA for Multi-Stage Text Retrieval](https://arxiv.org/abs/2310.08319).

## How to run

### Training
To train the reranker, we suggest to use the `torchrun` command:

```bash
torchrun --nnodes 1 --nproc-per-node 8 \
    -m pecos.xmr.reranker.train \
    --config_json_path <path_to_config_file>
```

### Predictions
To run the reranker in prediction mode, you can use the following command:
```bash
python -m pecos.xmr.reranker.predict --config_json_path <path_to_config_file>
```

## Config JSON Files
See example training/predict JSON files in `pecos/examples/msmarco-rankllama` folders.

## Data Schema
The column names for the data schema are configurable through the json configuration file.
Following are the various schemas that are supported by the reranker:

(1) Learning Target Schema
```
# +-----------------+---------------+-----------------------+
# | Column Name     | Data Type     | Description           |
# +-----------------+---------------+-----------------------+
# | inp_id          | int32         | input id              |
# | lbl_id          | array<int32>  | an array of label_id  |
# | score           | array<float>  | an array of rel_score |
# +-----------------+---------------+-----------------------+
```

(2) Input Feature Store Schema
```
# +-----------------+---------------+-----------------------+
# | Column Name     | Data Type     | Description           |
# +-----------------+---------------+-----------------------+
# | inp_id          | int32         | input id              |
# | keywords        | string        | keyword string        |
# +-----------------+---------------+-----------------------+
```

(3) Label Feature Store Schema

The label feature store supports variable number of columns. The column names 
can be provided in the configuration file.
```
# +-----------------+---------------+-----------------------+
# | Column Name     | Data Type     | Description           |
# +-----------------+---------------+-----------------------+
# | lbl_id          | int32         | input id              |
# | title           | string        | title text            |
# | content         | string        | content string        |
# | ...             | string        | content string        |
# +-----------------+---------------+-----------------------+
```

(4) Evaluation Schema
```
# +-----------------+---------------+-----------------------+
# | Column Name     | Data Type     | Description           |
# +-----------------+---------------+-----------------------+
# | inp_id          | int32         | input id              |
# | lbl_id          | int32         | label_id              |
# +-----------------+---------------+-----------------------+
```

(5) Evaluation Input Feature Store Schema
```
# +-----------------+---------------+-----------------------+
# | Column Name     | Data Type     | Description           |
# +-----------------+---------------+-----------------------+
# | inp_id          | int32         | input id              |
# | keywords        | string        | keyword string        |
# +-----------------+---------------+-----------------------+
```

(6) Evaluation Label Feature Store Schema
```
# +-----------------+---------------+-----------------------+
# | Column Name     | Data Type     | Description           |
# +-----------------+---------------+-----------------------+
# | lbl_id          | int32         | input id              |
# | title           | string        | title text            |
# | content         | string        | content string        |
# | ...             | string        | content string        |
# +-----------------+---------------+-----------------------+
```
