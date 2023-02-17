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
import logging
import json
from pecos.distributed.xmc.base import DistClustering
from pecos.distributed.xmc.xlinear.model import DistributedCPUXLinearModel as DistXLM
from pecos.distributed.comm.mpi_comm import MPIComm
from pecos.utils import logging_util
from pecos.utils.profile_util import MemInfo
from pecos.xmc.xlinear.model import XLinearModel
from pecos.xmc.xlinear.train import parse_arguments


LOGGER = logging.getLogger(__name__)


def add_dist_arguments(parser):
    parser.add_argument(
        "-nst",
        "--min-n-sub-tree",
        type=int,
        default=16,
        metavar="INT",
        help="the minimum number of sub-trees in training step, should be more than number of distributed machines.",
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
    parser.add_argument(
        "-mwf",
        "--main-workload-factor",
        type=float,
        default=0.3,
        metavar="FLOAT",
        help="main node vs worker node workload factor in distributed model training",
    )

    return parser


def do_train(args):
    """Distributed CPU training and saving XLinear model"""

    params = dict()
    if args.generate_params_skeleton:
        params["train_params"] = DistXLM.TrainParams.from_dict({}, recursive=True).to_dict()
        params["pred_params"] = DistXLM.PredParams.from_dict({}, recursive=True).to_dict()
        params["dist_params"] = DistXLM.DistParams.from_dict({}, recursive=True).to_dict()
        params["cluster_params"] = DistClustering.TrainParams.from_dict(
            {}, recursive=True
        ).to_dict()
        print(f"{json.dumps(params, indent=True)}")
        return

    if args.params_path:
        with open(args.params_path, "r") as fin:
            params = json.load(fin)

    # Parse args
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    train_params = params.get("train_params", None)
    pred_params = params.get("pred_params", None)
    dist_params = params.get("dist_params", None)
    cluster_params = params.get("cluster_params", None)

    if train_params is not None:
        train_params = DistXLM.TrainParams.from_dict(train_params)
    else:
        train_params = DistXLM.TrainParams.from_dict(args_dict, recursive=True)

    if pred_params is not None:
        pred_params = DistXLM.PredParams.from_dict(pred_params)
    else:
        pred_params = DistXLM.PredParams.from_dict(args_dict, recursive=True)

    if dist_params is not None:
        dist_params = DistXLM.DistParams.from_dict(dist_params)
    else:
        dist_params = DistXLM.DistParams.from_dict(args_dict)

    if cluster_params is not None:
        cluster_params = DistClustering.TrainParams.from_dict(cluster_params)
    else:
        cluster_params = DistClustering.TrainParams.from_dict(args_dict, recursive=True)

    # Distributed communicator
    mpi_comm = MPIComm()

    # Prepare data
    LOGGER.info(f"Started loading data on Rank {mpi_comm.get_rank()} ... {MemInfo.mem_info()}")
    X = XLinearModel.load_feature_matrix(args.inst_path)
    Y = XLinearModel.load_label_matrix(args.label_path, for_training=True)
    LOGGER.info(f"Done loading data on Rank {mpi_comm.get_rank()}. {MemInfo.mem_info()}")

    # Train Distributed XLinearModel
    xlm = DistXLM.train(
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
    parser = add_dist_arguments(parse_arguments())
    args = parser.parse_args()
    logging_util.setup_logging_config(level=args.verbose_level)
    do_train(args)
