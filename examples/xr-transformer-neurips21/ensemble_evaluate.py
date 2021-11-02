#!/usr/bin/env python3 -u

import argparse

from pecos.utils.smat_util import sorted_csr, CsrEnsembler, load_matrix

def parse_arguments():
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
        nargs="*",
        metavar="PATH",
        help="path to the file of predicted output (CSR: nr_insts * nr_items)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        required=True,
        nargs="*",
        metavar="PATH",
        help="tags attached to each prediction",
    )
    parser.add_argument(
        "--ens-method",
        type=str,
        metavar="STR",
        default="rank_average",
        help="prediction ensemble method",
    )

    return parser


def do_evaluation(args):
    """ Evaluate xlinear predictions """
    assert len(args.tags) == len(args.pred_path)
    Y_true = sorted_csr(load_matrix(args.truth_path).tocsr())
    Y_pred = [sorted_csr(load_matrix(pp).tocsr()) for pp in args.pred_path]
    print("==== evaluation results ====")
    CsrEnsembler.print_ens(Y_true, Y_pred, args.tags, ens_method=args.ens_method)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    do_evaluation(args)
