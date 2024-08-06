# PECOS XMR Reranker

This is a reranker for the PECOS XMR model. It is based on huggingface's transformers library. The reranker can be run in both
single process and distributed mode. It is based on the paper [Fine-Tuning LLaMA for Multi-Stage Text Retrieval](https://arxiv.org/abs/2310.08319).

## How to run
### Single process
To run the reranker in single process mode, you can use the following command:

```bash
python -m pecos.xmr.reranker.train --config_json_path <path_to_config_file>
```

### Distributed mode
To run the reranker in distributed mode, you can use the following command to initialize the distributed configuration:
```bash
accelerate config
```

Then you can run the reranker using the following command:
```bash
accelerate launch -m pecos.xmr.reranker.train --config_json_path <path_to_config_file>
```

### Predictions
To run the reranker in prediction mode, you can use the following command:
```bash
python -m pecos.xmr.reranker.predict --config_json_path <path_to_config_file>
```

## Configuration file

### Training
Here is an example of the configuration file for training:
```json
{
  "train_params": {
    "__meta__": {
      "class_fullname": "pecos.xmr.reranker.model###RankingModel.TrainParams"
    },
    "target_data_folder": "/home/ec2-user/docker_disk/datasets/ms_marco_partitioned/target",
    "input_data_folder":  "/home/ec2-user/docker_disk/datasets/ms_marco_partitioned/input",
    "label_data_folder":  "/home/ec2-user/docker_disk/datasets/ms_marco_partitioned/label",
    "training_args": {
      "__meta__": {
        "class_fullname": "pecos.xmr.reranker.trainer###RankLlamaTrainer.TrainingArgs"
      },
      "learning_rate": 1e-4,
      "output_dir": "./ds_model",
      "per_device_train_batch_size": 8,
      "gradient_accumulation_steps": 8,
      "max_steps": -1,
      "logging_strategy": "steps",
      "logging_first_step": false,
      "logging_steps": 10,
      "save_strategy": "steps",
      "save_steps": 50,
      "save_total_limit": 5,
      "seed": 42,
      "data_seed": 42,
      "bf16": true,
      "dataloader_num_workers": 2,
      "dataloader_prefetch_factor": 10,
      "gradient_checkpointing": true,
      "train_group_size": 16
    }
  },
  "model_params": {
    "__meta__": {
      "class_fullname": "pecos.xmr.reranker.model###RankingModel.ModelParams"
    },
    "encoder_args": {
      "__meta__": {
        "class_fullname": "pecos.xmr.reranker.model###CrossEncoder.Config"
      },
      "model_shortcut": "meta-llama/Llama-2-7b-hf",
      "model_init_kwargs": {},
      "model_modifier": {
                    "modifier_type": "peft",
                    "config_type": "LoraConfig" ,
                    "config": {
                        "r": 8,
                        "lora_alpha": 64,
                        "target_modules": ["q_proj", "v_proj"],
                        "modules_to_save": ["score", "classifier"],
                        "lora_dropout": 0.1
                    }
      }
    },
    "positive_passage_no_shuffle": false,
    "negative_passage_no_shuffle": false,
    "rerank_max_len": 196,
    "query_prefix": "query: ",
    "passage_prefix": "document: ",
    "inp_id_col": "inp_id",
    "lbl_idxs_col": "ret_idxs",
    "score_col": "rel",
    "keyword_col_name": "keywords",
    "content_col_names": ["title", "contents"],
    "append_eos_token": false,
    "pad_to_multiple_of": 16
  }
}
```

### Prediction
Following is the example of the configuration file for prediction:
```json
{
    "model_name_or_path": "/tmp/pecosdev/ds_model",
    "target_data_folder": "/home/ec2-user/docker_disk/datasets/msmarcoeval/target",
    "input_data_folder":  "/home/ec2-user/docker_disk/datasets/msmarcoeval/input",
    "label_data_folder":  "/home/ec2-user/docker_disk/datasets/msmarcoeval/label",
    "output_dir": "/tmp/xmrout",
    "per_device_eval_batch_size": 512,
    "dataloader_num_workers": 1,
    "dataloader_prefetch_factor": 10,
    "rerank_max_len": 196,
    "query_prefix": "query: ",
    "passage_prefix": "document: ",
    "inp_id_col": "inp_id",
    "lbl_id_col": "lbl_id",
    "keyword_col_name": "keywords",
    "content_col_names": ["title", "contents"],
    "append_eos_token": false,
    "pad_to_multiple_of": 8,
    "device": "cuda",
    "model_init_kwargs": {
        "device_map": "auto"
    }
}
```

## Data Schema
The column names for the data schema are configurable through the json configuration file. Following 
are the various schemas that are supported by the reranker:

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
