#!/usr/bin/env python3 -u
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
import deepspeed
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import socket
import signal
from pecos.utils import logging_util
from pecos.distributed.xmc.xtransformer.module import DeepSpeedUtils

logging.getLogger(torch.__name__).setLevel(logging.WARNING)
logging.getLogger("DeepSpeed").setLevel(logging.WARNING)

LOGGER = logging.getLogger(__name__)


def parse_arguments():
    """Parse evaluation arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shared-workdir",
        type=str,
        metavar="PATH",
        default=None,
        help="the shared workdir which can be accessed by each worker. Default None to disable check",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        metavar="INT",
        help=f"timeout in seconds for the diagnostic check. Default 60",
    )

    parser.add_argument(
        "--verbose-level",
        type=int,
        choices=logging_util.log_levels.keys(),
        default=2,
        metavar="INT",
        help=f"the verbose level, {', '.join([str(k) + ' for ' + logging.getLevelName(v) for k, v in logging_util.log_levels.items()])}. Default 2",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        metavar="RANK",
        help="local rank passed from torch distributed launcher",
    )
    return parser


def distributed_cluster_check(workdir=None, timeout=60):
    """
    Check deepspeed distributed setup using a dummy model

    Args:
        timeout (int): number of seconds to wait before raising exception.
            Default 60.
    """

    class TimeOutException(Exception):
        pass

    def alarm_handler(signum, frame):
        raise TimeOutException()

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)

        def forward(self, x):
            return F.relu(self.conv1(x))

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout)

    try:
        model = DummyModel()
        ds_config = DeepSpeedUtils.get_config_from_params()
        _ = deepspeed.initialize(
            model=model,
            config_params=ds_config,
            model_parameters=model.parameters(),
        )
        torch_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        ip = socket.gethostbyname(socket.gethostname())
        LOGGER.info(f"Report from {ip}: torch_rank={torch_rank}, world_size={world_size}")

        if workdir is not None:
            workdir = os.path.abspath(workdir)

            if torch_rank == 0:
                master_stamp = tempfile.TemporaryDirectory(dir=workdir)
                master_stamp_name = [f"{master_stamp.name}/_MASTER_STAMP"]
                open(master_stamp_name[0], "w").close()
            else:
                master_stamp_name = [None]

            torch.distributed.broadcast_object_list(master_stamp_name, 0)

            if not os.path.isfile(master_stamp_name[0]):
                raise ValueError(f"Rank{torch_rank} not able to access workdir at {workdir}")
            else:
                LOGGER.info(f"Rank{torch_rank} is able to access workdir at {workdir}")
            torch.distributed.barrier()

    except TimeOutException as ex:
        raise ex
    signal.alarm(0)


if __name__ == "__main__":
    """
    Sanity check for deepspeed distributed

    Usage:
        deepspeed --hostfile [PATH_TO_HOSTFILE] --module pecos.distributed.diagnostic_tools.deepspeed_comm
    """
    parser = parse_arguments()
    args = parser.parse_args()
    logging_util.setup_logging_config(level=args.verbose_level)
    distributed_cluster_check(workdir=args.shared_workdir, timeout=args.timeout)
