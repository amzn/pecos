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
import json

from pecos.core import XLINEAR_SOLVERS
from pecos.utils import cli, logging_util
from pecos.utils.featurization.text.vectorizers import Vectorizer
from pecos.xmc import PostProcessor

from .model import Text2Text


def parse_arguments(args):
    """Parse Text2Text model training arguments"""

    parser = argparse.ArgumentParser(
        description="Text2Text: Read input text training files, output item files and train a model"
    )
    parser.add_argument(
        "--generate-params-skeleton",
        action="store_true",
        help="generate template params-json to stdout",
    )

    skip_training = "--generate-params-skeleton" in sys.argv
    # ========= parameter jsons ============
    parser.add_argument(
        "--params-path",
        type=str,
        default=None,
        metavar="PARAMS_PATH",
        help="Json file for params (default None)",
    )
    # ======= actual arguments ========

    parser.add_argument(
        "-i",
        "--input-text-path",
        type=str,
        required=not skip_training,
        metavar="INPUT_TEXT_PATH",
        help="Text input file name. Format: in each line, OUTPUT_ID1,OUTPUT_ID2,OUTPUT_ID3,...\t INPUT_TEXT \
                    where OUTPUT_IDs are the zero-based output item indices corresponding to the line numbers of OUTPUT_ITEM_PATH. We assume utf-8 encoding for text.",
    )

    parser.add_argument(
        "-q",
        "--output-item-path",
        type=str,
        required=not skip_training,
        metavar="OUTPUT_ITEM_PATH",
        help="Output item file name. Format: each line corresponds to a representation of the output item. We assume utf-8 encoding for text.",
    )

    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        required=not skip_training,
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
        "--max-leaf-size",
        type=int,
        default=100,
        metavar="INT",
        help="The max size of the leaf nodes of hierarchical 2-means clustering. (default 100)",
    )

    parser.add_argument(
        "--nr-splits",
        type=int,
        default=16,
        metavar="INT",
        help="number of splits used to construct hierarchy (a power of 2 is recommended, default 16)",
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
        type=str,
        default="pifa",
        metavar="STR",
        help="Label embedding types. (default pifa).\
            We support pifa, pifa_lf_concat::Z=path, and pifa_lf_convex_combine::Z=path::alpha=scalar_value,\
            where path is the additional user-porivded label embedding path and alpha is the scalar value for convex combination.",
    )

    parser.add_argument(
        "--spherical",
        type=cli.str2bool,
        metavar="[true/false]",
        default=True,
        help="If true, do l2-normalize cluster centers while clustering. Default true.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="INT",
        help="Random seeds (default 0).",
    )

    parser.add_argument(
        "--kmeans-max-iter",
        type=int,
        default=20,
        metavar="INT",
        help="The max number of k-means iteration for indexing (default 20)",
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
        dest="neg_mining_chain",
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
        default=None,
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
        default=None,
        metavar="STR",
        help="the default post processor used in the prediction",
    )

    parser.add_argument(
        "--rel-norm",
        type=str,
        choices=["l1", "l2", "max", "no-norm"],
        default="no-norm",
        metavar="STR",
        help="norm type to row-wise normalzie relevance matrix for cost-sensitive learning",
    )

    parser.add_argument(
        "--rel-mode",
        type=str,
        metavar="STR",
        default="disable",
        help="mode to use relevance score for cost sensitive learning ['disable'(default), 'induce', 'ranker-only']",
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
    params = dict()
    if args.generate_params_skeleton:
        params["train_params"] = Text2Text.TrainParams.from_dict({}, recursive=True).to_dict()
        params["pred_params"] = Text2Text.PredParams.from_dict({}, recursive=True).to_dict()
        print(f"{json.dumps(params, indent=True)}")
        return

    if args.params_path:
        with open(args.params_path, "r") as fin:
            params = json.load(fin)

    train_params = params.get("train_params", None)
    pred_params = params.get("pred_params", None)

    if train_params is not None:
        train_params = Text2Text.TrainParams.from_dict(train_params)
    else:
        train_params = Text2Text.TrainParams.from_dict(
            {k: v for k, v in vars(args).items() if v is not None},
            recursive=True,
        )

    if pred_params is not None:
        pred_params = Text2Text.PredParams.from_dict(pred_params)
    else:
        pred_params = Text2Text.PredParams.from_dict(
            {k: v for k, v in vars(args).items() if v is not None},
            recursive=True,
        )

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
        vectorizer_config=vectorizer_config,
        train_params=train_params,
        pred_params=pred_params,
        workspace_folder=args.workspace_folder,
        **pred_kwargs,
    )

    t2t_model.save(args.model_folder)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    logging_util.setup_logging_config(level=args.verbose_level)
    train(args)
