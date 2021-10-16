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
import numpy as np
from pecos.utils import smat_util
from .model import HNSW


def parse_arguments():
    """Parse Inference arguments"""

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "-x",
        "--inst-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the CSR npz or Row-majored npy file of the feature matrix (nr_insts * nr_feats) to be indexed by HNSW",
    )
    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        required=True,
        metavar="DIR",
        help="path to the model folder to load the HNSW index for inference",
    )

    # Optional
    parser.add_argument(
        "-efS",
        "--efSearch",
        type=int,
        default=100,
        metavar="INT",
        help="size of the priority queue when performing best first search during inference. (Default 100)",
    )
    parser.add_argument(
        "-k",
        "--only-topk",
        type=int,
        default=10,
        metavar="INT",
        help="maximum number of candidates (sorted by distances, nearest first) to be returned",
    )
    parser.add_argument(
        "-n",
        "--threads",
        type=int,
        default=-1,
        metavar="int",
        help="number of threads to use for inference of hnsw indexer (default -1 to use all)",
    )
    parser.add_argument(
        "-y",
        "--label-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the npz file of the ground truth label matrix (CSR, nr_tst * nr_items)",
    )
    parser.add_argument(
        "-o",
        "--save-pred-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to save the predictions (CSR sorted by distances, nr_tst * nr_items)",
    )

    return parser


def do_predict(args):
    """Predict and Evaluate for HNSW model

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """

    # Load data
    Xt = smat_util.load_matrix(args.inst_path).astype(np.float32)

    # Load model
    model = HNSW.load(args.model_folder)

    # Setup HNSW Searchers for thread-safe inference
    threads = os.cpu_count() if args.threads <= 0 else args.threads
    searchers = model.searchers_create(num_searcher=threads)

    # Setup prediction params
    # pred_params.threads will be overrided if searchers are provided in model.predict()
    pred_params = HNSW.PredParams(
        efS=args.efSearch,
        topk=args.only_topk,
        threads=threads,
    )

    # Model Predicting
    Yt_pred = model.predict(
        Xt,
        pred_params=pred_params,
        searchers=searchers,
        ret_csr=True,
    )

    # Save prediction
    if args.save_pred_path:
        smat_util.save_matrix(args.save_pred_path, Yt_pred)

    # Evaluate Recallk@k
    if args.label_path:
        Yt = smat_util.load_matrix(args.label_path)
        # assuming ground truth is similarity-based (larger the better)
        Yt_topk = smat_util.sorted_csr(Yt, only_topk=args.only_topk)
        # assuming prediction matrix is distance-based, so need 1-dist=similiarty
        Yt_pred.data = 1.0 - Yt_pred.data
        metric = smat_util.Metrics.generate(Yt_topk, Yt_pred, topk=args.only_topk)
        print(
            "Recall{}@{} {:.6f}%".format(args.only_topk, args.only_topk, 100.0 * metric.recall[-1])
        )


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    do_predict(args)
