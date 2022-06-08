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
import json
import logging
import os

import numpy as np
from pecos.utils import cli, logging_util, smat_util, torch_util
from pecos.utils.cluster_util import ClusterChain

from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.xmc.xtransformer.model import XTransformer
from pecos.xmc.xtransformer.module import MLProblemWithText
from pecos.xmc.xtransformer.train import parse_arguments
from pecos.distributed.diagnostic_tools import deepspeed_comm as ds_diagnose
from .model import XTransformerDistTrainer
from .module import DeepSpeedUtils as ds_utils

LOGGER = logging.getLogger(__name__)


def add_dist_arguments(parser):
    """Add distributed training arguments"""
    # ========= train data paths ============
    parser.add_argument(
        "--hostfile",
        type=str,
        metavar="PATH",
        default="",
        help="path to the hostfile",
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
        "--shared-workdir",
        type=str,
        metavar="PATH",
        default=".",
        help="the shared workdir for distributed training",
    )
    parser.add_argument(
        "--max-shard-size",
        type=int,
        default=10**7,
        metavar="INT",
        help="max number of instances in each data shard",
    )
    return parser


def do_train(args):
    """Train and save XR-Transformer model.

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """
    params = dict()
    if args.generate_params_skeleton:
        params["train_params"] = XTransformer.TrainParams.from_dict({}, recursive=True).to_dict()
        params["pred_params"] = XTransformer.PredParams.from_dict({}, recursive=True).to_dict()
        params["dist_params"] = XTransformerDistTrainer.DistParams.from_dict(
            {}, recursive=True
        ).to_dict()
        print(f"{json.dumps(params, indent=True)}")
        return

    if args.params_path:
        with open(args.params_path, "r") as fin:
            params = json.load(fin)

    train_params = params.get("train_params", None)
    pred_params = params.get("pred_params", None)
    dist_params = params.get("dist_params", None)

    if train_params is not None:
        train_params = XTransformer.TrainParams.from_dict(train_params)
    else:
        train_params = XTransformer.TrainParams.from_dict(
            {k: v for k, v in vars(args).items() if v is not None},
            recursive=True,
        )

    if pred_params is not None:
        pred_params = XTransformer.PredParams.from_dict(pred_params)
    else:
        pred_params = XTransformer.PredParams.from_dict(
            {k: v for k, v in vars(args).items() if v is not None},
            recursive=True,
        )

    if dist_params is not None:
        dist_params = XTransformerDistTrainer.DistParams.from_dict(dist_params)
    else:
        dist_params = XTransformerDistTrainer.DistParams.from_dict(
            {k: v for k, v in vars(args).items() if v is not None},
            recursive=True,
        )

    # health check for distributed cluster and shared dir
    ds_utils.cli_launcher(
        ds_diagnose.__name__,
        hostfile=dist_params.hostfile,
        module_args={
            "timeout": 60,
            "shared_workdir": dist_params.shared_workdir,
            "verbose_level": 2,
        },
    )

    torch_util.set_seed(args.seed)
    LOGGER.info("Setting random seed {}".format(args.seed))

    # Load training feature
    if args.trn_feat_path:
        LOGGER.warning(f"Numerical features are ignored in current distributed implementation!")

    # Load training labels
    Y_trn = smat_util.load_matrix(args.trn_label_path, dtype=np.float32)
    LOGGER.info("Loaded training label matrix with shape={}".format(Y_trn.shape))

    # Load training texts
    trn_corpus = Preprocessor.load_data_from_file(
        args.trn_text_path,
        label_text_path=None,
        text_pos=0,
    )["corpus"]
    LOGGER.info("Loaded {} training sequences".format(len(trn_corpus)))

    # Load test feature if given
    if args.tst_feat_path or args.tst_label_path or args.tst_text_path:
        LOGGER.warning(f"Validation set is ignored in distributed training")

    # load cluster chain
    if os.path.exists(args.code_path):
        cluster_chain = ClusterChain.from_partial_chain(
            smat_util.load_matrix(args.code_path),
            min_codes=train_params.preliminary_indexer_params.min_codes,
            nr_splits=train_params.preliminary_indexer_params.nr_splits,
        )
        LOGGER.info("Loaded from code-path: {}".format(args.code_path))
    else:
        cluster_chain = None
        LOGGER.warning(
            "Label partition not provided, falling back to one-versue-all training. \
            For multi-resolution training, provide label partition with --code-path"
        )

    trn_prob = MLProblemWithText(trn_corpus, Y_trn)

    xtf = XTransformerDistTrainer.train(
        trn_prob,
        clustering=cluster_chain,
        train_params=train_params,
        pred_params=pred_params,
        dist_params=dist_params,
        beam_size=args.beam_size,
    )

    xtf.save(args.model_dir)


if __name__ == "__main__":
    parser = add_dist_arguments(parse_arguments())
    args = parser.parse_args()
    logging_util.setup_logging_config(level=args.verbose_level)
    do_train(args)
