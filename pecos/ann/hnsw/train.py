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
    """Parse training arguments"""

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "-x",
        "--inst-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the CSR npz or Row-majored npy file of the item matrix (nr_items * nr_feats) to be indexed by HNSW",
    )
    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        required=True,
        metavar="DIR",
        help="path to the model folder that saved the HNSW index",
    )

    # Optional

    # HNSW Indexing parameters
    parser.add_argument(
        "--metric-type",
        type=str,
        default="ip",
        metavar="STR",
        help="distance metric type, can be ip (inner product) or l2 (Euclidean distance), default is set to ip",
    )
    parser.add_argument(
        "-M",
        "--max-edge-per-node",
        type=int,
        default=32,
        metavar="INT",
        help="maximum number of edges per node for layer l=1,...,L. For l=0, it becomes 2*M (default 32)",
    )
    parser.add_argument(
        "-efC",
        "--efConstruction",
        type=int,
        default=100,
        metavar="INT",
        help="size of the priority queue when performing best first search during construction (default 100)",
    )
    parser.add_argument(
        "-n",
        "--threads",
        type=int,
        default=-1,
        metavar="int",
        help="number of threads to use for training and inference of hnsw indexer (default -1 to use all)",
    )
    parser.add_argument(
        "-L",
        "--max-level-upper-bound",
        type=int,
        default=-1,
        metavar="int",
        help="number of maximum layers in the hierarchical graph (default -1 to ignore)",
    )

    # HNSW Prediction kwargs
    parser.add_argument(
        "-efS",
        "--efSearch",
        type=int,
        default=100,
        metavar="INT",
        help="size of the priority queue when performing best first search during inference (default 100)",
    )
    parser.add_argument(
        "-k",
        "--only-topk",
        type=int,
        default=10,
        metavar="INT",
        help="maximum number of candidates (sorted by distances, nearest first) to be returned (default 10)",
    )

    return parser


def do_train(args):
    """Train and Save HNSW model

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """

    # Create model folder
    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)

    # Load training inputs
    X = smat_util.load_matrix(args.inst_path).astype(np.float32)

    # Setup training and prediction params
    # Note that prediction params can be overrided in inference time
    train_params = HNSW.TrainParams(
        M=args.max_edge_per_node,
        efC=args.efConstruction,
        metric_type=args.metric_type,
        max_level_upper_bound=args.max_level_upper_bound,
        threads=args.threads,
    )
    pred_params = HNSW.PredParams(
        efS=args.efSearch,
        topk=args.only_topk,
        threads=args.threads,
    )

    # train and save HNSW indexer
    model = HNSW.train(
        X,
        train_params=train_params,
        pred_params=pred_params,
    )

    model.save(args.model_folder)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    do_train(args)
