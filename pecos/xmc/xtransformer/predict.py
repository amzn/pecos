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
import logging
import os

from pecos.utils import cli, logging_util, smat_util, torch_util
from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.xmc import PostProcessor

from .model import XTransformer

LOGGER = logging.getLogger(__name__)


def parse_arguments():
    """Parse predicting arguments"""
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "-x",
        "--feat-path",
        type=str,
        metavar="PATH",
        help="Path to the instance feature matrix.",
    )
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
        "--save-pred-path",
        type=str,
        required=True,
        metavar="PATH",
        help="The path where the model predictions will be written.",
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
        "--max-pred-chunk",
        default=10 ** 7,
        metavar="INT",
        type=int,
        help="Max number of instances to predict on at once, set to avoid OOM. Set to None to predict on all instances at once. Default 10^7",
    )
    parser.add_argument(
        "--only-topk",
        default=None,
        type=int,
        metavar="INT",
        help="override the topk specified in the ranker (default None to disable overriding) ",
    )
    parser.add_argument(
        "-b",
        "--beam-size",
        type=int,
        default=None,
        metavar="INT",
        help="override the beam size specified in the ranker (default None to disable overriding)",
    )

    parser.add_argument(
        "-pp",
        "--post-processor",
        type=str,
        choices=PostProcessor.valid_list(),
        default=None,
        metavar="STR",
        help="override the post processor specified in the ranker (default None to disable overriding)",
    )
    parser.add_argument(
        "--use-gpu",
        type=cli.str2bool,
        metavar="[true/false]",
        default=True,
        help="if true, use CUDA if available. Default true",
    )
    parser.add_argument(
        "--batch-gen-workers",
        type=int,
        metavar="INT",
        default=4,
        help="number of CPUs to use for batch generation",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=-1,
        metavar="THREADS",
        help="number of threads to use for linear models(default -1 to denote all the CPUs)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, metavar="INT", help="random seed for initialization"
    )
    parser.add_argument(
        "--verbose-level",
        type=int,
        choices=logging_util.log_levels.keys(),
        default=1,
        metavar="INT",
        help=f"the verbose level, {', '.join([str(k) + ' for ' + logging.getLevelName(v) for k, v in logging_util.log_levels.items()])}, default 1",
    )
    return parser


def do_predict(args):
    """Predict with XTransformer and save the result.

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """
    if os.path.isdir(args.save_pred_path):
        args.save_pred_path = os.path.join(args.save_pred_path, "P.npz")

    torch_util.set_seed(args.seed)

    xtf = XTransformer.load(args.model_folder)

    # load instance feature and text
    if args.feat_path:
        X_feat = smat_util.load_matrix(args.feat_path)
    else:
        X_feat = None
    X_text = Preprocessor.load_data_from_file(args.text_path, label_text_path=None, text_pos=0)[
        "corpus"
    ]

    P_matrix = xtf.predict(
        X_text,
        X_feat=X_feat,
        batch_size=args.batch_size,
        batch_gen_workers=args.batch_gen_workers,
        use_gpu=args.use_gpu,
        beam_size=args.beam_size,
        only_topk=args.only_topk,
        post_processor=args.post_processor,
        max_pred_chunk=args.max_pred_chunk,
        threads=args.threads,
    )

    smat_util.save_matrix(args.save_pred_path, P_matrix)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    logging_util.setup_logging_config(level=args.verbose_level)
    do_predict(args)
