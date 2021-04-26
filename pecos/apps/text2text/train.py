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
import sys

import numpy as np
from pecos.core import XLINEAR_SOLVERS
from pecos.utils import cli, logging_util
from pecos.utils.featurization.text.vectorizers import Vectorizer
from pecos.xmc import Indexer, PostProcessor

from .model import Text2Text


def parse_arguments(args):
    """Parse Text2Text model training arguments"""

    parser = argparse.ArgumentParser(
        description="Text2Text: Read input text training files, output item files and train a model"
    )

    parser.add_argument(
        "-i",
        "--input-text-path",
        type=str,
        required=True,
        metavar="INPUT_TEXT_PATH",
        help="Text input file name. Format: in each line, OUTPUT_ID1,OUTPUT_ID2,OUTPUT_ID3,...\t INPUT_TEXT \
                    where OUTPUT_IDs are the zero-based output item indices corresponding to the line numbers of OUTPUT_ITEM_PATH. We assume utf-8 encoding for text.",
    )

    parser.add_argument(
        "-q",
        "--output-item-path",
        type=str,
        required=True,
        metavar="OUTPUT_ITEM_PATH",
        help="Output item file name. Format: each line corresponds to a representation of the output item. We assume utf-8 encoding for text.",
    )

    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        required=True,
        metavar="MODEL_FOLDER",
        help="Output model folder name",
    )

    parser.add_argument(
        "--workspace-folder",
        type=str,
        default=None,
        metavar="WORKSPACE_FOLDER",
        help="A folder name for storing intermediate variables during training",
    )

    vectorizer_config_group_parser = parser.add_mutually_exclusive_group()
    vectorizer_config_group_parser.add_argument(
        "--vectorizer-config-path",
        type=str,
        default=None,
        metavar="VECTORIZER_CONFIG_PATH",
        help="Json file for vectorizer config (default tfidf vectorizer)",
    )

    vectorizer_config_group_parser.add_argument(
        "--vectorizer-config-json",
        type=str,
        default='{"type":"tfidf", "kwargs":{}}',
        metavar="VECTORIZER_CONFIG_JSON",
        help='Json-format string for vectorizer config (default {"type":"tfidf", "kwargs":{}})',
    )

    parser.add_argument(
        "--dtype",
        type=lambda x: np.float32 if "32" in x else np.float64,
        default=np.float32,
        help="data type for the csr matrix. float32 | float64. (default float32)",
    )

    parser.add_argument(
        "--max-leaf-size",
        type=cli.comma_separated_type(int),
        default=[100],
        metavar="INT-LIST",
        help="The max size of the leaf nodes of hierarchical 2-means clustering. Multiple values (separated by comma) are supported and will lead to different individual models for ensembling. (default [100])",
    )

    parser.add_argument(
        "--nr-splits",
        type=int,
        default=2,
        metavar="INT",
        help="number of splits used to construct hierarchy (a power of 2 is recommended, default 2)",
    )

    parser.add_argument(
        "--imbalanced-ratio",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Value between 0.0 and 0.5 (inclusive). Indicates how relaxed the balancedness constraint of 2-means can be. Specifically, if an iteration of 2-means is clustering L labels, the size of the output 2 clusters will be within approx imbalanced_ratio * 2 * L of each other. (default 0.0)",
    )

    parser.add_argument(
        "--imbalanced-depth",
        type=int,
        default=100,
        metavar="INT",
        help="After hierarchical 2-means clustering has reached this depth, it will continue clustering as if --imbalanced-ratio is set to 0.0. (default 100)",
    )

    parser.add_argument(
        "--label-embed-type",
        type=cli.comma_separated_type(str),
        default="pifa",
        metavar="STR-LIST",
        help="Label embedding types. (default pifa). Multiple values (separated by comma) are supported and will lead to different individual models for ensembling.",
    )

    parser.add_argument(
        "--indexer",
        choices=Indexer.indexer_dict.keys(),
        default="hierarchicalkmeans",
        metavar="STR",
        help=f"Indexer algorithm (default hierarchicalkmeans). Available choices are {', '.join(Indexer.indexer_dict.keys())}",
    )

    parser.add_argument(
        "--no-spherical",
        action="store_true",
        default=False,
        help="Do not l2-normalize cluster centers while clustering",
    )

    parser.add_argument(
        "--seed",
        type=cli.comma_separated_type(int),
        default=[0],
        metavar="INT-LIST",
        help="Random seeds (default 0). Multiple values (separated by comma) are supported and will lead to different individual models for ensembling.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        metavar="INT",
        help="The max iteration for indexing (default 20)",
    )

    parser.add_argument(
        "-n",
        "--threads",
        type=int,
        default=-1,
        metavar="INT",
        help="Number of threads to use (default -1 to denote all the CPUs)",
    )

    # Linear matching/ranking parameters
    parser.add_argument(
        "-s",
        "--solver-type",
        type=str,
        default="L2R_L2LOSS_SVC_DUAL",
        metavar="STR",
        help="{} (default L2R_L2LOSS_SVC_DUAL)".format(" | ".join(XLINEAR_SOLVERS.keys())),
    )

    parser.add_argument(
        "--Cp",
        type=float,
        default=1.0,
        metavar="VAL",
        help="Coefficient for positive class in the loss function (default 1.0)",
    )

    parser.add_argument(
        "--Cn",
        type=float,
        default=1.0,
        metavar="VAL",
        help="Coefficient for negative class in the loss function (default 1.0)",
    )

    parser.add_argument(
        "--bias", type=float, default=1.0, metavar="VAL", help="bias term (default 1.0)"
    )

    parser.add_argument(
        "-ns",
        "--negative-sampling",
        type=str,
        choices=["tfn", "man", "tfn+man"],
        default="tfn",
        metavar="STR",
        help="Negative Sampling Schemes",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.1,
        metavar="VAL",
        help="Threshold to sparsify the model weights (default 0.1)",
    )

    # Prediction kwargs
    parser.add_argument(
        "-k",
        "--only-topk",
        type=int,
        default=20,
        metavar="INT",
        help="the default number of top labels used in the prediction",
    )

    parser.add_argument(
        "-b",
        "--beam-size",
        type=int,
        default=10,
        metavar="INT",
        help="the default size of beam search used in the prediction",
    )

    parser.add_argument(
        "-pp",
        "--post-processor",
        type=str,
        choices=PostProcessor.valid_list(),
        default="l3-hinge",
        metavar="STR",
        help="the default post processor used in the prediction",
    )

    parser.add_argument(
        "--verbose-level",
        type=int,
        choices=logging_util.log_levels.keys(),
        default=1,
        metavar="INT",
        help=f"the verbose level, {', '.join([str(k) + ' for ' + logging.getLevelName(v) for k, v in logging_util.log_levels.items()])}, default 1",
    )

    parsed_args = parser.parse_args(args)
    return parsed_args


def train(args):
    """Train Text2Text model

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """

    pred_kwargs = {
        "beam_size": args.beam_size,
        "only_topk": args.only_topk,
        "post_processor": args.post_processor,
    }

    vectorizer_config = Vectorizer.load_config_from_args(args)

    t2t_model = Text2Text.train(
        args.input_text_path,
        args.output_item_path,
        label_embed_type=args.label_embed_type,
        max_leaf_size=args.max_leaf_size,
        nr_splits=args.nr_splits,
        vectorizer_config=vectorizer_config,
        dtype=args.dtype,
        indexer_algo=[args.indexer],
        imbalanced_ratio=args.imbalanced_ratio,
        imbalanced_depth=args.imbalanced_depth,
        spherical=not args.no_spherical,
        seed=args.seed,
        max_iter=args.max_iter,
        threads=args.threads,
        solver_type=args.solver_type,
        Cp=args.Cp,
        Cn=args.Cn,
        bias=args.bias,
        threshold=args.threshold,
        negative_sampling_scheme=args.negative_sampling,
        pred_kwargs=pred_kwargs,
        workspace_folder=args.workspace_folder,
    )

    t2t_model.save(args.model_folder)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    logging_util.setup_logging_config(level=args.verbose_level)
    train(args)
