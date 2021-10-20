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
import itertools
import sys

from pecos.xmc import PostProcessor

from .model import Text2Text


def parse_arguments():
    """Parse Text2Text model prediction arguments"""

    parser = argparse.ArgumentParser(description="Text2Text: online prediction or batch prediction")

    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        required=True,
        metavar="MODEL_FOLDER",
        help="model folder name",
    )

    parser.add_argument(
        "-i",
        "--input-text-path",
        type=str,
        default="-",
        metavar="INPUT_TEXT_PATH",
        help='path to text input file name. (default "-" to denote stdin). We assume utf-8 encoding for text.',
    )

    parser.add_argument(
        "-o",
        "--predicted-output-item-path",
        type=str,
        default="-",
        metavar="OUTPUT_PATH",
        help='path to the predicted output item file name. (default "-" to denote stdout). We assume utf-8 encoding for text.',
    )

    parser.add_argument(
        "--output-to-keep-path",
        type=str,
        default=None,
        metavar="OUTPUT_TO_KEEP_PATH",
        help="path to valid output texts file name. (default None to keep all output items). We assume utf-8 encoding for text.",
    )

    parser.add_argument(
        "-k",
        "--only-topk",
        type=int,
        default=20,
        help="Output top-k items for each input (default 20)",
    )

    parser.add_argument(
        "-b",
        "--beam-size",
        type=int,
        default=None,
        metavar="INT",
        help="Override the beam size specified in the model (default None to disable overriding)",
    )

    parser.add_argument(
        "-pp",
        "--post-processor",
        type=str,
        choices=PostProcessor.valid_list(),
        default=None,
        metavar="STR",
        help="Override the post processor specified in the model (default None to disable overriding)",
    )

    parser.add_argument(
        "-B",
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for prediction (default 2048)",
    )

    parser.add_argument(
        "-T",
        "--threshold",
        type=float,
        default=None,
        help="Drop output items with scores less than this threshold among top-k items (default None)",
    )

    parser.add_argument(
        "--meta-info-path",
        type=str,
        default=None,
        help="Path to the file of meta info for each line in input_text_path. (default None)",
    )

    return parser


def predict(args):
    """Predict labels for given text inputs using Text2Text model

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """

    t2t_model = Text2Text.load(
        args.model_folder, is_predict_only=(args.output_to_keep_path is None)
    )

    if args.output_to_keep_path is not None:
        outputs_to_keep = []
        with open(args.output_to_keep_path, "r", encoding="utf-8") as f:
            for line in f:
                outputs_to_keep += [line.strip()]
        t2t_model.set_output_constraint(outputs_to_keep)

    if args.predicted_output_item_path != "-":
        fout = open(args.predicted_output_item_path, "w", encoding="utf-8")
    else:
        fout = sys.stdout

    if args.input_text_path != "-":
        if args.meta_info_path is not None:
            fmeta = open(args.meta_info_path, "r", encoding="utf-8")
        else:
            fmeta = []

        with open(args.input_text_path, "r", encoding="utf-8") as fin:
            # Divide the test file into small batches to avoid out-of-memory issue.
            corpus = []
            meta_info = []
            for input_line, meta_line in itertools.zip_longest(fin, fmeta, fillvalue=""):
                corpus += [input_line.strip("\n").split("\t")[-1]]
                meta_info += [meta_line.strip("\n")]
                if len(corpus) == args.batch_size:
                    Y = t2t_model.predict(
                        corpus,
                        beam_size=args.beam_size,
                        only_topk=args.only_topk,
                        threshold=args.threshold,
                    )
                    if args.meta_info_path is None:
                        t2t_model.print_predictions(Y, fout)
                    else:
                        t2t_model.print_predictions(Y, fout, meta_info=meta_info)
                    corpus = []
                    meta_info = []

            if len(corpus) > 0:
                Y = t2t_model.predict(
                    corpus,
                    only_topk=args.only_topk,
                    beam_size=args.beam_size,
                    post_processor=args.post_processor,
                    threshold=args.threshold,
                )
                if args.meta_info_path is None:
                    t2t_model.print_predictions(Y, fout)
                else:
                    t2t_model.print_predictions(Y, fout, meta_info=meta_info)
    else:
        fin = sys.stdin
        for line in fin:
            Y = t2t_model.predict(
                [line.strip()],
                only_topk=args.only_topk,
                beam_size=args.beam_size,
                post_processor=args.post_processor,
                threshold=args.threshold,
            )
            t2t_model.print_predictions(Y, fout)

    fin.close()
    fout.close()


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    predict(args)
