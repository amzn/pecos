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
import torch
import logging
import itertools

from pecos.utils import smat_util, logging_util
from pecos.xmc.xtransformer.model import XTransformer

LOGGER = logging.getLogger(__name__)


def parse_arguments():
    """Parse evaluation arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--text-path",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the instance text file.",
    )
    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to load x-transformer model.",
    )
    parser.add_argument(
        "-o",
        "--save-emb-folder",
        type=str,
        required=True,
        metavar="PATH",
        help="The folder in which the embeddings will be written (in WORLD_SIZE shards)",
    )

    # ======= Other parameters ========
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        metavar="INT",
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--batch-gen-workers",
        type=int,
        metavar="INT",
        default=4,
        help="number of CPUs to use for batch generation",
    )
    parser.add_argument(
        "--truncate-length",
        default=None,
        type=int,
        metavar="INT",
        help="max number of tokens to encode",
    )
    parser.add_argument(
        "--max-pred-chunk",
        default=10**7,
        metavar="INT",
        type=int,
        help="Max number of instances to predict on at once, set to avoid OOM. Set to None to predict on all instances at once. Default 10^7",
    )
    parser.add_argument(
        "--verbose-level",
        type=int,
        choices=logging_util.log_levels.keys(),
        default=1,
        metavar="INT",
        help=f"the verbose level, {', '.join([str(k) + ' for ' + logging.getLevelName(v) for k, v in logging_util.log_levels.items()])}, default 1",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        metavar="RANK",
        help="local rank passed from torch distributed launcher",
    )
    return parser


def dist_encode(args):
    """Encode with XTransformer model in distributed fashion.
    Each worker will encode an exclusive chunk and save the result to args.save_emb_folder/X.emb.[WORKER_RANK].npy

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """
    os.makedirs(args.save_emb_folder, exist_ok=True)

    deepspeed.init_distributed(dist_backend="nccl")
    global_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    LOGGER.info(
        f"Initialized device for rank={args.local_rank}, global_rank={global_rank}, world_size={world_size}"
    )
    xtf = XTransformer.load(args.model_folder)

    # get number of lines without reading all in memory
    nr_inst = sum(1 for line in open(args.text_path, "r"))
    chunk_size = (nr_inst + world_size - 1) // world_size
    start = global_rank * chunk_size
    end = min(nr_inst, start + chunk_size)

    with open(args.text_path, "r") as fin:
        X = []
        for line in itertools.islice(fin, start, end):
            X.append(line.strip())
    LOGGER.info(f"Rank{global_rank}/{world_size} will encode {start} to {end}")

    pred_params = xtf.get_pred_params()
    for i in range(len(pred_params.matcher_params_chain)):
        if args.truncate_length:
            pred_params.matcher_params_chain[i].truncate_length = args.truncate_length

    X_emb = xtf.encode(
        X,
        batch_size=args.batch_size,
        batch_gen_workers=args.batch_gen_workers,
        device_id=args.local_rank,
        pred_params=pred_params,
        max_pred_chunk=args.max_pred_chunk,
    )

    local_tgt = os.path.join(args.save_emb_folder, f"X.emb.{global_rank}.npy")
    smat_util.save_matrix(local_tgt, X_emb)
    LOGGER.info(f"Rank{global_rank}/{world_size} saved embedding {X_emb.shape} to {local_tgt}")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    logging_util.setup_logging_config(level=args.verbose_level)
    dist_encode(args)
