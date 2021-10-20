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

from pecos.utils import smat_util


def parse_arguments():
    """Parse evaluation arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-y",
        "--truth-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the file of with ground truth output (CSR: nr_insts * nr_items)",
    )

    parser.add_argument(
        "-p",
        "--pred-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the file of predicted output (CSR: nr_insts * nr_items)",
    )

    parser.add_argument(
        "-k",
        "--topk",
        "--only-topk",
        type=int,
        default=10,
        dest="topk",
        metavar="INT",
        help="evaluate @k",
    )

    return parser


def do_evaluation(args):
    """Evaluate xlinear predictions

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """

    Y_true = smat_util.load_matrix(args.truth_path).tocsr()
    Y_pred = smat_util.load_matrix(args.pred_path).tocsr()
    metric = smat_util.Metrics.generate(Y_true, Y_pred, topk=args.topk)
    print("==== evaluation results ====")
    print(metric)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    do_evaluation(args)
