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
import gc
import logging
import os

import numpy as np
from pecos.utils import logging_util, smat_util, torch_util
from pecos.utils.cluster_util import ClusterChain
from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.xmc import Indexer, LabelEmbeddingFactory, PostProcessor

from .matcher import TransformerMatcher
from .model import XTransformer
from .module import MLProblemWithText

LOGGER = logging.getLogger(__name__)


def parse_arguments():
    """Parse training arguments"""
    parser = argparse.ArgumentParser()
    # ========= Required parameters =========
    parser.add_argument(
        "--trn-text-path",
        type=str,
        metavar="PATH",
        required=True,
        help="path to the training text file",
    )
    parser.add_argument(
        "--trn-feat-path",
        type=str,
        metavar="PATH",
        required=True,
        help="path to the instance feature matrix (CSR matrix, nr_insts * nr_features)",
    )
    parser.add_argument(
        "--trn-label-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the training label matrix (CSR matrix, nr_insts * nr_labels)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        metavar="PATH",
        help="the output directory where the models will be saved.",
    )
    # ========= test data paths ============
    parser.add_argument(
        "--tst-text-path",
        type=str,
        metavar="PATH",
        default="",
        help="path to the test text file",
    )
    parser.add_argument(
        "--tst-feat-path",
        type=str,
        metavar="PATH",
        default="",
        help="path to the test instance feature matrix",
    )
    parser.add_argument(
        "--tst-label-path",
        type=str,
        metavar="PATH",
        default="",
        help="path to the file of the test label matrix",
    )
    # ========= indexer parameters ============
    parser.add_argument(
        "--code-path",
        type=str,
        default="",
        metavar="PATH",
        help="path to the clustering file (CSR matrix, nr_insts * nr_labels)",
    )
    parser.add_argument(
        "--label-feat-path",
        type=str,
        default="",
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
        "--min-codes",
        type=int,
        default=-1,
        metavar="INT",
        help="minimal number of codes, default -1 to use number of leaf clusters",
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
        "--max-iter",
        type=int,
        default=20,
        metavar="INT",
        help="max iterations for indexer (default 20)",
    )
    parser.add_argument(
        "--max-no-improve-cnt",
        type=int,
        default=-1,
        metavar="INT",
        help="if > 0, training will stop when this number of validation steps result in no improvment.",
    )
    # ========= matcher parameters ============
    parser.add_argument(
        "--max-match-clusters",
        type=int,
        default=-1,
        metavar="INT",
        help="max number of clusters on which to train matcher; if <0, set to number of leaf clusters. Default -1",
    )
    parser.add_argument(
        "--no-fine-tune",
        action="store_true",
        help="whether do fine-tune on loaded/downloaded transformers",
    )
    parser.add_argument(
        "--model-shortcut",
        type=str,
        metavar="STR",
        default="bert-base-uncased",
        help="pre-trained transformer model name shortcut for download (default bert-base-uncased)",
    )
    parser.add_argument(
        "--init-model-dir",
        type=str,
        metavar="PATH",
        default="",
        help="path to load existing TransformerMatcher checkpoint from disk, overrides model-shortcut",
    )
    # ========== ranker parameters =============
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
        default="noop",
        metavar="STR",
        help="the default post processor used in the prediction",
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
        "--ensemble-method",
        type=str,
        choices=["concat-only", "transformer-only", "average", "rank_average", "round_robin"],
        default="transformer-only",
        metavar="STR",
        help="ensemble method for transformer/concat prediction ensemble",
    )
    parser.add_argument(
        "-t",
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
        choices=TransformerMatcher.LOSS_FUNCTION_TYPES,
        default="squared-hinge",
        metavar="STR",
        help="loss function type",
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
        "--save-m-mat",
        default="",
        type=str,
        metavar="PATH",
        help="path to save the matching matrix",
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
        default=8,
        metavar="INT",
        type=int,
        help="batch size per GPU.",
    )
    parser.add_argument(
        "--max-pred-chunk",
        default=None,
        metavar="INT",
        type=int,
        help="Max number of instances to predict on at once, set to avoid OOM. Default None to disable",
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
        default=5e-5,
        metavar="VAL",
        type=float,
        help="maximum learning rate for Adam.",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.0,
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
        default=5.0,
        metavar="INT",
        type=int,
        help="total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max-steps",
        default=-1,
        metavar="INT",
        type=int,
        help="if > 0: set total number of training steps to perform. Override num-train-epochs.",
    )
    parser.add_argument(
        "--warmup-steps",
        default=0,
        metavar="INT",
        type=int,
        help="Linear warmup over warmup-steps.",
    )
    parser.add_argument(
        "--logging-steps", type=int, metavar="INT", default=50, help="log every X updates steps."
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        metavar="INT",
        default=100,
        help="save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--only-topk",
        default=20,
        metavar="INT",
        type=int,
        help="store topk prediction at ranker prediction stage",
    )
    parser.add_argument(
        "--max-active-matching-labels",
        default=None,
        metavar="INT",
        type=int,
        help="max number of active matching labels, will subsample from existing negative samples if necessary",
    )
    parser.add_argument(
        "--save-emb-dir",
        default="",
        metavar="PATH",
        type=str,
        help="dir to save instance embeddings.",
    )
    parser.add_argument(
        "--disable-gpu",
        action="store_true",
        help="disable CUDA training even if it's available",
    )
    parser.add_argument(
        "--force-label-embed-in-gpu",
        action="store_true",
        help="always put label embed in GPU. This will increase GPU memory cost but accelerate training.",
    )
    parser.add_argument(
        "--do-encoder-bootstrap",
        action="store_true",
        help="initialize lower layer model weights from upper layer model",
    )
    parser.add_argument(
        "--do-text-model-bootstrap",
        action="store_true",
        help="initialize the text_model from xlinear training. Ignored if do-encoder-bootstrap is not used.",
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
        default=1,
        metavar="INT",
        help=f"the verbose level, {', '.join([str(k) + ' for ' + logging.getLevelName(v) for k, v in logging_util.log_levels.items()])}, default 1",
    )

    return parser


def do_train(args):
    """Train and save X-Transformer model.

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """
    torch_util.set_seed(args.seed)
    LOGGER.info("Setting random seed {}".format(args.seed))

    # Load training feature
    X_trn = smat_util.load_matrix(args.trn_feat_path, dtype=np.float32)
    LOGGER.info("Loaded training feature matrix with shape={}".format(X_trn.shape))

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
    _, trn_corpus = Preprocessor.load_data_from_file(
        args.trn_text_path,
        label_text_path=None,
        text_pos=0,
    )
    LOGGER.info("Loaded {} training sequences".format(len(trn_corpus)))

    # Load test text if given
    if args.tst_text_path:
        _, tst_corpus = Preprocessor.load_data_from_file(
            args.tst_text_path,
            label_text_path=None,
            text_pos=0,
        )
        LOGGER.info("Loaded {} test sequences".format(len(tst_corpus)))
    else:
        tst_corpus = None

    # construct full cluster chain
    if os.path.exists(args.code_path):
        cluster_chain = ClusterChain.load(args.code_path)
        LOGGER.info("Loaded from code-path: {}".format(args.code_path))
    else:
        if os.path.isfile(args.label_feat_path):
            label_feat = smat_util.load_matrix(args.label_feat_path, dtype=np.float32)
            LOGGER.info(
                "Loaded label feature matrix shape={}, from {}".format(
                    label_feat.shape, args.label_feat_path
                )
            )
        else:
            label_feat = LabelEmbeddingFactory.pifa(Y_trn, X_trn)
            if args.label_feat_path:
                smat_util.save_matrix(args.label_feat_path, label_feat)
                LOGGER.info(
                    "Created label feature matrix with shape={}, saved to {}".format(
                        label_feat.shape, args.label_feat_path
                    )
                )

        cluster_chain = Indexer.gen(
            label_feat,
            args.indexer,
            nr_splits=args.nr_splits,
            min_codes=args.min_codes,
            max_leaf_size=args.max_leaf_size,
            imbalanced_depth=args.imbalanced_depth,
            imbalanced_ratio=args.imbalanced_ratio,
            seed=args.seed,
            max_iter=args.max_iter,
            spherical=not args.no_spherical,
        )
        del label_feat
        gc.collect()
        if args.code_path:
            cluster_chain.save(args.code_path)
            LOGGER.info("Created clustering chain, saved to {}".format(args.code_path))

    LOGGER.info(
        "Constructed clustering chain for ranker: {}".format([cc.shape for cc in cluster_chain])
    )
    # if not given, match number of leaf clusters
    nr_leaf_clusters = cluster_chain[-1].shape[1]
    if args.max_match_clusters < 0:
        args.max_match_clusters = nr_leaf_clusters

    if args.min_codes < 0:
        args.min_codes = nr_leaf_clusters

    # get the matcher-ranker split level
    if args.max_match_clusters < cluster_chain[-1].shape[0]:  # if not matcher for all
        args.ranker_level = len(cluster_chain) - next(
            level
            for level, C in enumerate(cluster_chain[:])
            if C.shape[1] >= args.max_match_clusters
        )
        LOGGER.info(
            "Apply matcher at ranker-level {} with nr_labels={}".format(
                args.ranker_level, cluster_chain[-args.ranker_level].shape[1]
            )
        )
    else:
        args.ranker_level = 0
        LOGGER.info(
            "Apply matcher at ranker-level 0 with nr_labels={}".format(cluster_chain[-1].shape[0])
        )

    trn_prob = MLProblemWithText(trn_corpus, X_trn, Y_trn)
    if all(v is not None for v in [tst_corpus, X_tst, Y_tst]):
        val_prob = MLProblemWithText(tst_corpus, X_tst, Y_tst)
    else:
        val_prob = None

    # for HierarchicalMLModel.TrainParams
    args.neg_mining_chain = args.negative_sampling

    train_params = XTransformer.TrainParams.from_dict(vars(args), recursive=True)
    pred_params = XTransformer.PredParams.from_dict(vars(args), recursive=True)

    xtf = XTransformer.train(
        trn_prob,
        cluster_chain,
        val_prob=val_prob,
        train_params=train_params,
        pred_params=pred_params,
        beam_size=args.beam_size,
    )

    xtf.save(args.model_dir)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    logging_util.setup_logging_config(level=args.verbose_level)
    do_train(args)
