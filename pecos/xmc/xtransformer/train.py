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
import json
import logging
import os
import sys

import numpy as np
from pecos.utils import cli, logging_util, smat_util, torch_util
from pecos.utils.cluster_util import ClusterChain
from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.xmc import PostProcessor

from .matcher import TransformerMatcher
from .model import XTransformer
from .module import MLProblemWithText

LOGGER = logging.getLogger(__name__)


def parse_arguments():
    """Parse training arguments"""
    parser = argparse.ArgumentParser()
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
    # ========= train data paths ============
    parser.add_argument(
        "-t",
        "--trn-text-path",
        type=str,
        metavar="PATH",
        required=not skip_training,
        help="path to the training text file",
    )
    parser.add_argument(
        "-x",
        "--trn-feat-path",
        type=str,
        default="",
        metavar="PATH",
        help="path to the instance feature matrix (CSR matrix, nr_insts * nr_features)",
    )
    parser.add_argument(
        "-y",
        "--trn-label-path",
        type=str,
        metavar="PATH",
        required=not skip_training,
        help="path to the training label matrix (CSR matrix, nr_insts * nr_labels)",
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        metavar="DIR",
        required=not skip_training,
        help="the output directory where the models will be saved.",
    )
    # ========= test data paths ============
    parser.add_argument(
        "-tt",
        "--tst-text-path",
        type=str,
        metavar="PATH",
        default="",
        help="path to the test text file",
    )
    parser.add_argument(
        "-xt",
        "--tst-feat-path",
        type=str,
        metavar="PATH",
        default="",
        help="path to the test instance feature matrix",
    )
    parser.add_argument(
        "-yt",
        "--tst-label-path",
        type=str,
        metavar="PATH",
        default="",
        help="path to the file of the test label matrix",
    )
    # ========= indexer parameters ============
    parser.add_argument(
        "--fix-clustering",
        type=cli.str2bool,
        metavar="[true/false]",
        default=False,
        help="if true, use the same hierarchial label tree for fine-tuning and final prediction. Default false.",
    )
    parser.add_argument(
        "--code-path",
        type=str,
        default="",
        metavar="PATH",
        help="path to the clustering file (CSR matrix, nr_insts * nr_labels). Will be used for both prelimiary and refined HLT if provided",
    )
    parser.add_argument(
        "--label-feat-path",
        type=str,
        default="",
        metavar="PATH",
        help="path to the CSR npz or Row-majored npy file of the label feature matrix (nr_labels * nr_label_feats). Will be used to generate prelimiary HLT.",
    )
    parser.add_argument(
        "--nr-splits",
        type=int,
        default=16,
        metavar="INT",
        help="number of splits used to construct hierarchy (a power of 2 is recommended)",
    )
    parser.add_argument(
        "--min-codes",
        type=int,
        default=None,
        metavar="INT",
        help="minimal number of codes, default None to use nr-splits",
    )
    parser.add_argument(
        "--max-leaf-size",
        type=int,
        default=100,
        metavar="INT",
        help="The max size of the leaf nodes of hierarchical clustering. If larger than the number of labels, OVA model will be trained. Default 100.",
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
    # ========= matcher parameters ============
    parser.add_argument(
        "--max-match-clusters",
        type=int,
        default=32768,
        metavar="INT",
        help="max number of clusters on which to fine-tune transformers. Default 32768",
    )
    parser.add_argument(
        "--do-fine-tune",
        type=cli.str2bool,
        metavar="[true/false]",
        default=True,
        help="If true, do fine-tune on loaded/downloaded transformers. Default true",
    )
    parser.add_argument(
        "--model-shortcut",
        type=str,
        metavar="STR",
        default="bert-base-cased",
        help="pre-trained transformer model name shortcut for download (default bert-base-cased)",
    )
    parser.add_argument(
        "--init-model-dir",
        type=str,
        metavar="PATH",
        default="",
        help="path to load existing TransformerMatcher model from disk, overrides model-shortcut",
    )
    # ========== ranker parameters =============
    parser.add_argument(
        "--only-encoder",
        type=cli.str2bool,
        metavar="[true/false]",
        default=False,
        help="if true, only train text encoder. Default false.",
    )
    parser.add_argument(
        "-b",
        "--beam-size",
        type=int,
        default=None,
        metavar="INT",
        help="the default size of beam search used in the prediction",
    )
    parser.add_argument(
        "-k",
        "--only-topk",
        default=None,
        metavar="INT",
        type=int,
        help="the default number of top labels used in the prediction",
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
        "--ensemble-method",
        type=str,
        choices=["concat-only", "transformer-only", "average", "rank_average", "round_robin"],
        default="transformer-only",
        metavar="STR",
        help="ensemble method for transformer/concat prediction ensemble",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        metavar="VAL",
        help="threshold to sparsify the model weights (default 0.1)",
    )
    # ========== Other parameters ===========
    parser.add_argument(
        "--loss-function",
        type=str,
        choices=TransformerMatcher.LOSS_FUNCTION_TYPES.keys(),
        default="squared-hinge",
        metavar="STR",
        help="loss function type for transformer training",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        metavar="PATH",
        type=str,
        help="dir to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--saved-trn-pt",
        default="",
        metavar="PATH",
        type=str,
        help="dir to save/load tokenized train tensor",
    )
    parser.add_argument(
        "--saved-val-pt",
        default="",
        metavar="PATH",
        type=str,
        help="dir to save/load tokenized validation tensor",
    )
    parser.add_argument(
        "--truncate-length",
        default=128,
        metavar="INT",
        type=int,
        help="if given, truncate input text to this length, else use longest input length as truncate-length.",
    )
    parser.add_argument(
        "--hidden-dropout-prob",
        default=0.1,
        metavar="VAL",
        type=float,
        help="hidden dropout prob in deep transformer models.",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        metavar="INT",
        type=int,
        help="batch size per GPU.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        metavar="INT",
        default=1,
        help="number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-4,
        metavar="VAL",
        type=float,
        help="maximum learning rate for Adam.",
    )
    parser.add_argument(
        "--weight-decay",
        default=0,
        metavar="VAL",
        type=float,
        help="weight decay rate for regularization",
    )
    parser.add_argument(
        "--adam-epsilon",
        default=1e-8,
        metavar="VAL",
        type=float,
        help="epsilon for Adam optimizer.",
    )
    parser.add_argument(
        "--max-grad-norm", default=1.0, metavar="VAL", type=float, help="max gradient norm."
    )
    parser.add_argument(
        "--num-train-epochs",
        default=5,
        metavar="INT",
        type=int,
        help="total number of training epochs to perform for each sub-task.",
    )
    parser.add_argument(
        "--max-steps",
        default=0,
        metavar="INT",
        type=int,
        help="if > 0: set total number of training steps to perform for each sub-task. Overrides num-train-epochs.",
    )
    parser.add_argument(
        "--steps-scale",
        nargs="+",
        type=float,
        default=None,
        metavar="FLOAT",
        help="scale number of transformer fine-tuning steps for each layer. Default None to ignore",
    )
    parser.add_argument(
        "--max-no-improve-cnt",
        type=int,
        default=-1,
        metavar="INT",
        help="if > 0, training will stop when this number of validation steps result in no improvment. Default -1 to ignore",
    )
    parser.add_argument(
        "--lr-schedule",
        default="linear",
        metavar="STR",
        type=str,
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="learning rate schedule for transformer fine-tuning. See transformers.SchedulerType for details",
    )
    parser.add_argument(
        "--warmup-steps",
        default=0,
        metavar="INT",
        type=int,
        help="Linear warmup over warmup-steps.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        metavar="INT",
        default=50,
        help="log training information every NUM updates steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        metavar="INT",
        default=100,
        help="save checkpoint every NUM updates steps.",
    )
    parser.add_argument(
        "--max-active-matching-labels",
        default=None,
        metavar="INT",
        type=int,
        help="max number of active matching labels, will subsample from existing negative samples if necessary. Default None to ignore.",
    )
    parser.add_argument(
        "--max-num-labels-in-gpu",
        default=65536,
        metavar="INT",
        type=int,
        help="Upper limit on labels to put output layer in GPU. Default 65536",
    )
    parser.add_argument(
        "--save-emb-dir",
        default=None,
        metavar="PATH",
        type=str,
        help="dir to save the final instance embeddings.",
    )
    parser.add_argument(
        "--use-gpu",
        type=cli.str2bool,
        metavar="[true/false]",
        default=True,
        help="if true, use CUDA training if available. Default true",
    )
    parser.add_argument(
        "--bootstrap-method",
        type=str,
        default="linear",
        choices=["linear", "inherit", "no-bootstrap"],
        help="initialization method for the text_model weights. Ignored if None is given. Default linear",
    )
    parser.add_argument(
        "--batch-gen-workers",
        type=int,
        metavar="INT",
        default=4,
        help="number of CPUs to use for batch generation",
    )
    parser.add_argument(
        "--seed", type=int, metavar="INT", default=0, help="random seed for initialization"
    )
    parser.add_argument(
        "--verbose-level",
        type=int,
        choices=logging_util.log_levels.keys(),
        default=2,
        metavar="INT",
        help=f"the verbose level, {', '.join([str(k) + ' for ' + logging.getLevelName(v) for k, v in logging_util.log_levels.items()])}. Default 2",
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
        print(f"{json.dumps(params, indent=True)}")
        return

    if args.params_path:
        with open(args.params_path, "r") as fin:
            params = json.load(fin)

    train_params = params.get("train_params", None)
    pred_params = params.get("pred_params", None)

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

    torch_util.set_seed(args.seed)
    LOGGER.info("Setting random seed {}".format(args.seed))

    # Load training feature
    if args.trn_feat_path:
        X_trn = smat_util.load_matrix(args.trn_feat_path, dtype=np.float32)
        LOGGER.info("Loaded training feature matrix with shape={}".format(X_trn.shape))
    else:
        X_trn = None
        LOGGER.info("Training feature matrix not provided")
        if not args.label_feat_path and not args.code_path:
            raise ValueError("trn-feat is required unless code-path or label-feat is provided.")

    # Load training labels
    Y_trn = smat_util.load_matrix(args.trn_label_path, dtype=np.float32)
    LOGGER.info("Loaded training label matrix with shape={}".format(Y_trn.shape))

    # Load test feature if given
    if args.tst_feat_path:
        X_tst = smat_util.load_matrix(args.tst_feat_path, dtype=np.float32)
        LOGGER.info("Loaded test feature matrix with shape={}".format(X_tst.shape))
    else:
        X_tst = None

    # Load test labels if given
    if args.tst_label_path:
        Y_tst = smat_util.load_matrix(args.tst_label_path, dtype=np.float32)
        LOGGER.info("Loaded test label matrix with shape={}".format(Y_tst.shape))
    else:
        Y_tst = None

    # Load training texts
    trn_corpus = Preprocessor.load_data_from_file(
        args.trn_text_path,
        label_text_path=None,
        text_pos=0,
    )["corpus"]
    LOGGER.info("Loaded {} training sequences".format(len(trn_corpus)))

    # Load test text if given
    if args.tst_text_path:
        tst_corpus = Preprocessor.load_data_from_file(
            args.tst_text_path,
            label_text_path=None,
            text_pos=0,
        )["corpus"]
        LOGGER.info("Loaded {} test sequences".format(len(tst_corpus)))
    else:
        tst_corpus = None

    # load cluster chain or label features
    cluster_chain, label_feat = None, None
    if os.path.exists(args.code_path):
        cluster_chain = ClusterChain.from_partial_chain(
            smat_util.load_matrix(args.code_path),
            min_codes=args.min_codes,
            nr_splits=args.nr_splits,
        )
        LOGGER.info("Loaded from code-path: {}".format(args.code_path))
    else:
        if os.path.isfile(args.label_feat_path):
            label_feat = smat_util.load_matrix(args.label_feat_path, dtype=np.float32)
            LOGGER.info(
                "Loaded label feature matrix shape={}, from {}".format(
                    label_feat.shape, args.label_feat_path
                )
            )

    trn_prob = MLProblemWithText(trn_corpus, Y_trn, X_feat=X_trn)
    if all(v is not None for v in [tst_corpus, Y_tst]):
        val_prob = MLProblemWithText(tst_corpus, Y_tst, X_feat=X_tst)
    else:
        val_prob = None

    xtf = XTransformer.train(
        trn_prob,
        clustering=cluster_chain,
        val_prob=val_prob,
        train_params=train_params,
        pred_params=pred_params,
        beam_size=args.beam_size,
        steps_scale=args.steps_scale,
        label_feat=label_feat,
    )

    xtf.save(args.model_dir)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    logging_util.setup_logging_config(level=args.verbose_level)
    do_train(args)
