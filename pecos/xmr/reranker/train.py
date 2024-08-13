import argparse
import json
import logging
import os

import datasets.distributed
import torch
from datasets import load_dataset
from transformers import set_seed

from .data_utils import RankingDataUtils
from .model import RankingModel

logger = logging.getLogger(__name__)

"""
Usage:
```bash
python -m pecos.xmr.reranker.train --config_json_path config.json
```
"""


def main(config_json_path: str):
    """
    Args:
        config_json_path: JSON configuration for running the training
    """
    # parse train_params and model_params from json
    with open(config_json_path, "r") as fin:
        param = json.load(fin)
    model_params: RankingModel.ModelParams = RankingModel.ModelParams.from_dict(
        param.get("model_params", None),
        recursive=True,
    )

    train_params: RankingModel.TrainParams = RankingModel.TrainParams.from_dict(
        param.get("train_params", None),
        recursive=True,
    )

    set_seed(train_params.training_args.seed)

    # helper function for getting the list of filepaths in a folder
    def construct_file_list(folder):
        return [os.path.join(folder, x) for x in os.listdir(folder)]

    input_files = construct_file_list(train_params.input_data_folder)
    label_files = construct_file_list(train_params.label_data_folder)
    input_files, label_files = RankingDataUtils.get_sorted_data_files(
        input_files, "inp_id"
    ), RankingDataUtils.get_sorted_data_files(label_files, "lbl_id")

    train_dataset = load_dataset(
        "parquet", data_dir=train_params.target_data_folder, streaming=True, split="train"
    )
    train_dataset_rows = RankingDataUtils.get_parquet_rows(train_params.target_data_folder)
    logger.info(f"total target inputs: {train_dataset_rows}")

    training_args = train_params.training_args
    # set the max_steps in accordance with the number of num_rows
    if training_args.max_steps <= 0:
        ws = training_args.world_size
        bs = training_args.per_device_train_batch_size
        gas = training_args.gradient_accumulation_steps
        batch_size = ws * bs * gas
        max_steps = train_dataset_rows // batch_size
        training_args.max_steps = max_steps
        logger.info(f"total batch size: {batch_size}, train steps: {max_steps}")
    else:
        logger.info(f"max steps: {training_args.max_steps}")

    table_stores = {
        "input": load_dataset("parquet", data_files=input_files, split="train"),
        "label": load_dataset("parquet", data_files=label_files, split="train"),
    }

    train_dataset = train_dataset.shuffle(buffer_size=5000, seed=training_args.data_seed)
    train_dataset = datasets.distributed.split_dataset_by_node(
        train_dataset, training_args.local_rank, training_args.world_size
    )

    logger.info("Waiting for main process to perform the mapping")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    RankingModel.train(train_dataset, table_stores, model_params, train_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json_path", type=str, required=True)
    args = parser.parse_args()
    main(args.config_json_path)
