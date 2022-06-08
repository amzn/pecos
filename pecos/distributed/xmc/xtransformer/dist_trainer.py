#!/usr/bin/env python3 -u
#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.

import argparse
import deepspeed
import os
import json
import time
import torch
import logging
import random

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from pecos.utils import cli, logging_util
from pecos.xmc.xtransformer.matcher import TransformerMatcher
from pecos.xmc.xtransformer.module import XMCTextDataset

from .module import AllInOneForXMCModel, DeepSpeedUtils

LOGGER = logging.getLogger(__name__)


def parse_arguments():
    """Parse arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the training data (XMCTextDataset) to load",
    )

    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the TransformerMatcher model to fine-tune",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to save the output checkpoint",
    )

    parser.add_argument(
        "-p",
        "--params-path",
        type=str,
        default=None,
        metavar="PARAMS_PATH",
        help="Json file for params (default None)",
    )

    parser.add_argument(
        "--verbose-level",
        type=int,
        choices=logging_util.log_levels.keys(),
        default=2,
        metavar="INT",
        help=f"the verbose level, {', '.join([str(k) + ' for ' + logging.getLevelName(v) for k, v in logging_util.log_levels.items()])}. Default 2",
    )

    parser.add_argument(
        "--fp16",
        type=cli.str2bool,
        metavar="[true/false]",
        default=True,
        help="If true, do half-precision training",
    )

    parser.add_argument(
        "--shard-scheme",
        type=str,
        choices=["synchronized", "ordered"],
        metavar="STR",
        default="synchronized",
        help="access scheme for training data shards",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        metavar="RANK",
        help="local rank passed from torch distributed launcher",
    )
    return parser


def dist_fine_tune(args):
    """Fine tune on a single XMC task

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """
    # env set by deepspeed launcher
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    # === get fine-tuning params ===
    if args.params_path:
        with open(args.params_path, "r") as fin:
            params = json.load(fin)

    train_params = params.get("train_params", None)
    pred_params = params.get("pred_params", None)

    train_params = TransformerMatcher.TrainParams.from_dict(train_params)
    train_params.fp16 = args.fp16

    pred_params = TransformerMatcher.PredParams.from_dict(pred_params)

    # === load training data and model ===
    data_stats = XMCTextDataset.get_data_stats(args.data_path)
    num_instances = data_stats["num_instances"]
    num_shards = data_stats["num_shards"]
    LOGGER.info(f"Train data info loaded from {args.data_path}")

    model = AllInOneForXMCModel.load(args.model_path)
    loss_function = TransformerMatcher.get_loss_function(train_params.loss_function)
    LOGGER.info(f"Model loaded from {args.model_path}")

    # === compute stopping criteria ===
    total_batch_size = world_size * train_params.batch_size
    batches_per_epoch = (num_instances + total_batch_size - 1) // total_batch_size
    steps_per_epoch = batches_per_epoch // train_params.gradient_accumulation_steps

    if train_params.max_steps > 0:
        t_total = train_params.max_steps
        train_params.num_train_epochs = (
            train_params.max_steps + steps_per_epoch - 1
        ) // steps_per_epoch
    else:
        t_total = steps_per_epoch * train_params.num_train_epochs
        train_params.max_steps = t_total

    train_params.save_steps = min(train_params.save_steps, t_total)
    train_params.logging_steps = min(train_params.logging_steps, t_total)

    # === setup deepspeed config and engin ===
    ds_config = DeepSpeedUtils.get_config_from_params(train_params)
    params_to_optmize = model.prepare_params(train_params.weight_decay)
    ds_engine, _, _, scheduler = deepspeed.initialize(
        model=model,
        config_params=ds_config,
        model_parameters=params_to_optmize,
        training_data=TensorDataset(torch.zeros((num_instances,))),
    )

    # === start fine-tuning ===
    global_rank = torch.distributed.get_rank()
    if world_size > num_shards and args.shard_scheme != "synchronized":
        LOGGER.warning(f"more workers than shards, fall back to synchronized shard access")
        args.shard_scheme = "synchronized"

    if global_rank == 0:
        LOGGER.info("***** Running training *****")
        LOGGER.info("  Num examples = %d", num_instances)
        LOGGER.info("  Num labels = %d", model.nr_labels)
        LOGGER.info("  Num Epochs = %d", train_params.num_train_epochs)
        LOGGER.info("  Learning Rate Schedule = %s", train_params.lr_schedule)
        LOGGER.info("  Batch size = %d", total_batch_size)
        LOGGER.info("  Gradient Accumulation steps = %d", train_params.gradient_accumulation_steps)
        LOGGER.info("  Num batches per epoch = %d", batches_per_epoch)
        LOGGER.info("  Total optimization steps = %d", train_params.max_steps)
        LOGGER.info("  Dist World Size = %d", world_size)
        LOGGER.info("  Data shard access scheme = %s", args.shard_scheme)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    total_train_time, logging_elapsed = 0.0, 0.0
    logging_steps = train_params.logging_steps

    for epoch in range(1, int(train_params.num_train_epochs) + 1):
        shard_access_order = list(range(num_shards))
        random.shuffle(shard_access_order)

        if args.shard_scheme == "ordered":
            shard_load_per_epoch = (num_shards + world_size - 1) // world_size
            shard_access_order = [
                (global_rank + kk * world_size) % num_shards for kk in range(shard_load_per_epoch)
            ]

        ds_engine.module.train()

        # load next shard
        for shard_id in shard_access_order:
            train_data = XMCTextDataset.load(args.data_path, shard=shard_id)

            # data loader and actual max_steps
            if args.shard_scheme == "synchronized":
                sampler = DistributedSampler(
                    train_data,
                    num_replicas=world_size,
                    rank=global_rank,
                )
            else:
                sampler = RandomSampler(train_data)

            train_dataloader = DataLoader(
                train_data,
                sampler=sampler,
                pin_memory=False,
                batch_size=train_params.batch_size,
                num_workers=train_params.batch_gen_workers // max(world_size, 1),
            )
            LOGGER.debug(
                f"Rank{global_rank}/{world_size}: Training data({shard_id}/{num_shards}) loaded, num_batches={len(train_dataloader)}"
            )

            for batch_cnt, batch in enumerate(train_dataloader):
                start_time = time.time()
                batch = tuple(t.to(ds_engine.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "instance_number": batch[3],
                    "label_values": batch[4],
                    "label_indices": batch[-1] if train_data.has_ns else None,
                }
                logits = ds_engine(**inputs)
                loss = loss_function(logits, inputs["label_values"]).mean()
                ds_engine.backward(loss)
                ds_engine.step()

                tr_loss += loss.detach().item()
                global_step += 1
                logging_elapsed += time.time() - start_time
                total_train_time += time.time() - start_time

                if global_rank == 0:
                    if logging_steps > 0 and global_step % logging_steps == 0:
                        cur_loss = (tr_loss - logging_loss) / logging_steps

                        # incase .step() hasn't been called (fp16 could skip steps)
                        try:
                            cur_lr = scheduler.get_last_lr()[0]
                        except AssertionError:
                            cur_lr = 0

                        LOGGER.info(
                            "| [{:4d}/{:4d}][{:6d}/{:6d}] | {:4d}/{:4d} batches | ms/batch {:5.4f} | train_loss {:6e} | lr {:.6e}".format(
                                int(epoch),
                                int(train_params.num_train_epochs),
                                int(global_step),
                                int(train_params.max_steps),
                                int(batch_cnt),
                                batches_per_epoch,
                                logging_elapsed * 1000.0 / logging_steps,
                                cur_loss,
                                cur_lr,
                            )
                        )
                        logging_loss = tr_loss
                        logging_elapsed = 0

                    if global_step % train_params.save_steps == 0:
                        ds_engine.module.save(args.output_path)

                if global_step >= train_params.max_steps:
                    # within shard
                    break
            if global_step >= train_params.max_steps:
                # within epoch
                break
        if global_step >= train_params.max_steps:
            # outmost loop
            break
    if global_rank == 0:
        ds_engine.module.save(args.output_path)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    logging_util.setup_logging_config(level=args.verbose_level)
    dist_fine_tune(args)
