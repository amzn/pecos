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
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for prediction (default None)",
    )

    parser.add_argument(
        "-n",
        "--threads",
        type=int,
        default=-1,
        metavar="THREADS",
        help="number of threads to use (default -1 to denote all the CPUs)",
    )
    return parser


def do_predict(args):
    """Predict and Evaluate for xlinear model

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """

    # Load data
    Xt = XLinearModel.load_feature_matrix(args.inst_path)

    # Model Predicting
    xlinear_model = XLinearModel.load(args.model_folder, is_predict_only=True)

    if args.batch_size is not None:
        Yts = []
        for i in range(0, Xt.shape[0], args.batch_size):
            Yte = xlinear_model.predict(
                Xt[i : i + args.batch_size, :],
                only_topk=args.only_topk,
                beam_size=args.beam_size,
                post_processor=args.post_processor,
                threads=args.threads,
            )
            Yts.append(Yte)
        # vstack_csr will retain indices order
        Yt_pred = smat_util.vstack_csr(Yts)
    else:
        Yt_pred = xlinear_model.predict(
            Xt,
            only_topk=args.only_topk,
            beam_size=args.beam_size,
            post_processor=args.post_processor,
            threads=args.threads,
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
