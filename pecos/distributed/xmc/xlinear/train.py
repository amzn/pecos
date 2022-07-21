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
from pecos.distributed.xmc.base import DistClustering
from pecos.distributed.xmc.xlinear.model import DistributedCPUXLinearModel
from pecos.distributed.comm.mpi_comm import MPIComm
from pecos.utils import logging_util
from pecos.utils.profile_util import MemInfo
from pecos.xmc import PostProcessor
from pecos.xmc.xlinear.model import XLinearModel
from pecos.xmc.base import HierarchicalKMeans


LOGGER = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()

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

    parser.add_argument(
        "-nst",
        "--min-n-sub-tree",
        type=int,
        default=16,
        metavar="INT",
        help="the minimum number of sub-trees in training step, should be more than number of distributed machines.",
    )
    parser.add_argument(
        "--nr-splits",
        type=int,
        default=16,
        metavar="INT",
        help="number of splits used to construct hierarchy (a power of 2 is recommended)",
    )
    parser.add_argument(
        "-mle",
        "--meta-label-embedding-method",
        type=str,
        default="pifa",
        help="label embedding method for meta-tree",
    )
    parser.add_argument(
        "-sle",
        "--sub-label-embedding-method",
        type=str,
        default="pifa",
        help="label embedding method for sub-tree",
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
        default=None,
        metavar="INT",
        help="the default size of beam search used in the prediction",
    )
    parser.add_argument(
        "--max-leaf-size",
        type=int,
        default=100,
        metavar="INT",
        help="The max size of the leaf nodes of hierarchical 2-means clustering. Multiple values (separated by comma) are supported and will lead to different individual models for ensembling. (default [100])",
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
        "--seed", type=int, default=0, metavar="INT", help="random seed (default 0)"
    )
    parser.add_argument(
        "--bias", type=float, default=1.0, metavar="VAL", help="bias term (default 1.0)"
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
        "-mwf",
        "--main-workload-factor",
        type=float,
        default=0.3,
        metavar="FLOAT",
        help="main node vs worker node workload factor in distributed model training",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.1,
        metavar="VAL",
        help="threshold to sparsify the model weights.",
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
    """Distributed CPU training and saving XLinear model"""

    # Distributed communicator
    mpi_comm = MPIComm()

    # Parse args
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    train_params = DistributedCPUXLinearModel.TrainParams.from_dict(args_dict, recursive=True)
    cluster_params = DistClustering.ClusterParams(
        indexer_params=HierarchicalKMeans.TrainParams.from_dict(args_dict),
        meta_label_embedding_method=args.meta_label_embedding_method,
        sub_label_embedding_method=args.sub_label_embedding_method,
    )
    pred_params = DistributedCPUXLinearModel.PredParams.from_dict(args_dict, recursive=True)
    dist_params = DistributedCPUXLinearModel.DistParams.from_dict(args_dict)

    # Prepare data
    LOGGER.info(f"Started loading data on Rank {mpi_comm.get_rank()} ... {MemInfo.mem_info()}")
    X = XLinearModel.load_feature_matrix(args.inst_path)
    Y = XLinearModel.load_label_matrix(args.label_path, for_training=True)
    LOGGER.info(f"Done loading data on Rank {mpi_comm.get_rank()}. {MemInfo.mem_info()}")

    # Train Distributed XLinearModel
    xlm = DistributedCPUXLinearModel.train(
        dist_comm=mpi_comm,
        X=X,
        Y=Y,
        cluster_params=cluster_params,
        train_params=train_params,
        pred_params=pred_params,
        dist_params=dist_params,
    )

    # Main node write model
    if mpi_comm.get_rank() == 0:
        LOGGER.info(f"Saving model to {args.model_folder}...")
        xlm.save(args.model_folder)
        LOGGER.info("Done saving model.")


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    logging_util.setup_logging_config(level=args.verbose_level)
    do_train(args)
