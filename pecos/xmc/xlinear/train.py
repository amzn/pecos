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
import os

from pecos.core import XLINEAR_SOLVERS
from pecos.utils import smat_util
from pecos.utils.cluster_util import ClusterChain
from pecos.xmc import Indexer, LabelEmbeddingFactory, PostProcessor

from .model import XLinearModel


def parse_arguments():
    """Parse training arguments"""

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "-x",
        "--inst-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the CSR npz or Row-majored npy file of the feature matrix (nr_insts * nr_feats)",
    )

    parser.add_argument(
        "-y",
        "--label-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the CSR npz file of the label matrix (nr_insts * nr_labels)",
    )

    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        required=True,
        metavar="DIR",
        help="path to the model folder.",
    )

    # Optional

    # Indexing parameters
    parser.add_argument(
        "-f",
        "--label-feat-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the CSR npz or Row-majored npy file of the label feature matrix (nr_labels * nr_label_feats)",
    )

    parser.add_argument(
        "--nr-splits",
        type=int,
        default=2,
        metavar="INT",
        help="number of splits used to construct hierarchy (a power of 2 is recommended)",
    )

    parser.add_argument(
        "--indexer",
        choices=Indexer.indexer_dict.keys(),
        default="hierarchicalkmeans",
        metavar="STR",
        help=f"Indexer algorithm (default hierarchicalkmeans). Available choices are {', '.join(Indexer.indexer_dict.keys())}",
    )

    parser.add_argument(
        "--max-leaf-size",
        type=int,
        default=100,
        metavar="INT",
        help="The max size of the leaf nodes of hierarchical 2-means clustering. Multiple values (separated by comma) are supported and will lead to different individual models for ensembling. (default [100])",
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
        "--no-spherical",
        action="store_true",
        default=False,
        help="Do not l2-normalize cluster centers while clustering",
    )

    parser.add_argument(
        "--seed", type=int, default=0, metavar="INT", help="random seed (default 0)"
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        metavar="INT",
        help="max iterations for indexer (default 20)",
    )

    parser.add_argument(
        "-n",
        "--threads",
        type=int,
        default=-1,
        metavar="INT",
        help="number of threads to use (default -1 to denote all the CPUs)",
    )

    parser.add_argument(
        "-c",
        "--code-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the npz file of the code matrix (CSC, nr_labels * nr_codes)",
    )

    parser.add_argument(
        "-um",
        "--usn-match-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the npz file of the user supplied matching matrix (CSR, nr_insts * nr_codes), will be add to negative sampling if given",
    )

    parser.add_argument(
        "-uy",
        "--usn-label-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the npz file of the user supplied label importance matrix (CSR, nr_insts * nr_labels), will be add to negative sampling if given",
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
        help="coefficient for positive class in the loss function (default 1.0)",
    )

    parser.add_argument(
        "--Cn",
        type=float,
        default=1.0,
        metavar="VAL",
        help="coefficient for negative class in the loss function (default 1.0)",
    )

    parser.add_argument(
        "--bias", type=float, default=1.0, metavar="VAL", help="bias term (default 1.0)"
    )

    parser.add_argument(
        "-ns",
        "--negative-sampling",
        type=str,
        choices=["tfn", "man", "tfn+man", "usn", "usn+tfn", "usn+man", "usn+tfn+man"],
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
        help="threshold to sparsify the model weights (default 0.1)",
    )

    parser.add_argument(
        "-z",
        "--max-nonzeros-per-label",
        type=int,
        default=0,
        metavar="NONZEROS",
        help="keep at most NONZEROS weight parameters per label in model(default 0 to denote nr_features + 1)",
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

    return parser


def do_train(args):
    """Train and Save xlinear model

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """

    # Create model folder
    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)

    # Load training inputs and labels
    X = XLinearModel.load_feature_matrix(args.inst_path)
    Y = XLinearModel.load_label_matrix(args.label_path, for_training=True)

    if args.code_path:
        cluster_chain = ClusterChain.load(args.code_path)
    else:
        if args.label_feat_path:
            label_feat = XLinearModel.load_feature_matrix(args.label_feat_path)
        else:
            label_feat = LabelEmbeddingFactory.create(Y, X, method="pifa")

        cluster_chain = Indexer.gen(
            label_feat,
            args.indexer,
            nr_splits=args.nr_splits,
            max_leaf_size=args.max_leaf_size,
            imbalanced_depth=args.imbalanced_depth,
            imbalanced_ratio=args.imbalanced_ratio,
            seed=args.seed,
            max_iter=args.max_iter,
            threads=args.threads,
            spherical=not args.no_spherical,
        )

    # load label importance matrix if given
    if args.usn_label_path:
        usn_label_mat = smat_util.load_matrix(args.usn_label_path)
    else:
        usn_label_mat = None
    # load user supplied matching matrix if given
    if args.usn_match_path:
        usn_match_mat = smat_util.load_matrix(args.usn_match_path)
    else:
        usn_match_mat = None
    usn_match_dict = {0: usn_label_mat, 1: usn_match_mat}

    pred_kwargs = {}
    for kw in ["beam_size", "only_topk", "post_processor"]:
        if getattr(args, kw, None) is not None:
            pred_kwargs[kw] = getattr(args, kw)

    xlm = XLinearModel.train(
        X,
        Y,
        cluster_chain,
        user_supplied_negatives=usn_match_dict,
        negative_sampling_scheme=args.negative_sampling,
        pred_kwargs=pred_kwargs,
        nr_splits=args.nr_splits,
        threads=args.threads,
        solver_type=args.solver_type,
        Cp=args.Cp,
        Cn=args.Cn,
        bias=args.bias,
        threshold=args.threshold,
        max_nonzeros_per_label=args.max_nonzeros_per_label,
    )

    xlm.save(args.model_folder)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    do_train(args)
