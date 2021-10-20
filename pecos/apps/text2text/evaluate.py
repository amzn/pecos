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
import json

import scipy.sparse as smat
from pecos.utils import smat_util


def parse_arguments():
    """Parse Text2Text model evaluation arguments"""

    parser = argparse.ArgumentParser(
        description="Calculate precision and recall for text item outputs, where both ground truth target items and predicted items are text-based. "
    )

    parser.add_argument(
        "-p",
        "--pred-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the file of predicted output. Format follows the output from pecos.apps.text2text.predict",
    )

    parser.add_argument(
        "-y",
        "--truth-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the file of with ground truth output. \
                Format 1 (Only has ground truth text items): each line is a list of tab-separated sorted text items. \
                Format 2 (Same as training input format): each line is OUTPUT_ID1,OUTPUT_ID2,OUTPUT_ID3,...\t INPUT_TEXT. \
                    where OUTPUT_IDs are the zero-based output text item indices corresponding to the line numbers of TEXT_ITEM_PATH.",
    )

    parser.add_argument(
        "-q",
        "--text-item-path",
        type=str,
        default=None,
        metavar="TEXT_ITEM_PATH",
        help="Text item file name. Format: each line corresponds to a text item. If this path is given, we assume TRUTH_PATH uses Format 2. Otherwise, TRUTH_PATH uses Format 1",
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
    """Evaluate predicted labels for Text2Text model

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """

    # Initialize an item-to-index dictionary, item_dict
    if args.text_item_path is None:
        item_dict = {}
    else:
        item_dict = {
            line.strip("\n"): i
            for i, line in enumerate(open(args.text_item_path, "r", encoding="utf-8"))
        }

    # parse the ground truth items
    col_id_t = []
    row_id_t = []
    val_t = []
    num_samples_t = 0
    with open(args.truth_path, "r", encoding="utf-8") as fg:
        for i, line in enumerate(fg):
            num_samples_t = num_samples_t + 1
            # If text_item_path is not given, use Format 1 for truth_path: each line is a list of tab-separated sorted text items
            if args.text_item_path is None:
                for item in line.strip().split("\t"):
                    if item not in item_dict:
                        item_dict[item] = len(item_dict)
                    row_id_t.append(i)
                    col_id_t.append(item_dict[item])
                    val_t.append(1.0)
            # If text_item_path is given, use Format 2 for truth_path: each line is OUTPUT_ID1,OUTPUT_ID2,OUTPUT_ID3,...\t INPUT_TEXT
            else:
                for idx in line.strip().split("\t")[0].split(","):
                    row_id_t.append(i)
                    col_id_t.append(int(idx))
                    val_t.append(1.0)

    # parse the predicted items
    col_id_p = []
    row_id_p = []
    val_p = []
    num_samples_p = 0
    with open(args.pred_path, "r", encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            num_samples_p = num_samples_p + 1
            item_scores = json.loads(line.strip())["data"]
            for j, x in enumerate(item_scores):
                item = x[0]
                if item not in item_dict:
                    item_dict[item] = len(item_dict)
                row_id_p.append(i)
                col_id_p.append(item_dict[item])
                val_p.append(1.0 / (j + 1))

    assert num_samples_t == num_samples_p
    Y_true = smat.csr_matrix((val_t, (row_id_t, col_id_t)), shape=(num_samples_t, len(item_dict)))
    Y_pred = smat.csr_matrix((val_p, (row_id_p, col_id_p)), shape=(num_samples_p, len(item_dict)))

    metric = smat_util.Metrics.generate(Y_true, Y_pred, topk=args.topk)
    print("==== evaluation results ====")
    print(metric)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    do_evaluation(args)
