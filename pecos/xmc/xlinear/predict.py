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
from pecos.xmc import PostProcessor

from .model import XLinearModel


def parse_arguments():
    """Parse prediction arguments"""

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "-x",
        "--inst-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the npz file of the feature matrix (CSR, nr_insts * nr_feats)",
    )

    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        required=True,
        metavar="DIR",
        help="path to the model folder.",
    )

    # Optional
    parser.add_argument(
        "-k",
        "--only-topk",
        type=int,
        default=None,
        metavar="INT",
        help="override the only topk specified in the model (default None to disable overriding)",
    )

    parser.add_argument(
        "-b",
        "--beam-size",
        type=int,
        default=None,
        metavar="INT",
        help="override the beam size specified in the model (default None to disable overriding)",
    )

    parser.add_argument(
        "-pp",
        "--post-processor",
        type=str,
        choices=PostProcessor.valid_list(),
        default=None,
        metavar="STR",
        help="override the post processor specified in the model (default None to disable overriding)",
    )

    parser.add_argument(
        "-y",
        "--label-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the npz file of the label matrix (CSR, nr_insts * nr_labels)",
    )

    parser.add_argument(
        "-o",
        "--save-pred-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to save the predictions (sorted CSR, nr_insts * nr_labels)",
    )

    parser.add_argument(
        "-B",
        "--max-pred-chunk",
        default=10 ** 7,
        metavar="INT",
        type=int,
        help="Max number of instances to predict on at once, set to avoid OOM. Set to None to predict on all instances at once. Default 10^7",
    )

    parser.add_argument(
        "-n",
        "--threads",
        type=int,
        default=-1,
        metavar="THREADS",
        help="number of threads to use (default -1 to denote all the CPUs)",
    )

    parser.add_argument(
        "-so",
        "--selected-output",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the npz file of the selected output matrix (CSR, nr_insts * nr_labels), only-topk and beam-size are ignored if given",
    )
    return parser


def do_predict(args):
    """Predict and Evaluate for xlinear model

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """

    # Load data
    Xt = XLinearModel.load_feature_matrix(args.inst_path)

    if args.selected_output is not None:
        # Selected Output
        selected_outputs_csr = XLinearModel.load_feature_matrix(args.selected_output)
        xlinear_model = XLinearModel.load(
            args.model_folder, is_predict_only=True, weight_matrix_type="CSC"
        )
    else:
        # TopK
        selected_outputs_csr = None
        xlinear_model = XLinearModel.load(args.model_folder, is_predict_only=True)

    # Model Predicting
    Yt_pred = xlinear_model.predict(
        Xt,
        selected_outputs_csr=selected_outputs_csr,
        only_topk=args.only_topk,
        beam_size=args.beam_size,
        post_processor=args.post_processor,
        threads=args.threads,
        max_pred_chunk=args.max_pred_chunk,
    )

    # Save prediction
    if args.save_pred_path:
        smat_util.save_matrix(args.save_pred_path, Yt_pred)

    # Evaluate
    if args.label_path:
        Yt = XLinearModel.load_label_matrix(args.label_path)
        metric = smat_util.Metrics.generate(Yt, Yt_pred, topk=10)
        print("==== evaluation results ====")
        print(metric)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    do_predict(args)
