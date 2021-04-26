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

from .model import SentencePieceModel


def predict(args):
    """Tokenize text file using trained SentencePiece model

    Args:
        args (Namespace): Namespace object of prediction arguments from cli below.
    """

    model = SentencePieceModel.load(args.model_folder)
    to_ids = True if args.output_type == "ids" else False
    model.predict_file(args.input_path, args.output_path, to_ids)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SentencePiece: tokenize text")

    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        required=True,
        metavar="MODEL_FOLDER",
        help="path to model folder",
    )

    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        default="-",
        metavar="PATH",
        help="path to input text file name. (default '-' to denote stdin)",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="-",
        metavar="PATH",
        help="path to output encoded file name. (default '-' to denote stdout)",
    )

    parser.add_argument(
        "-t",
        "--output-type",
        type=str,
        default="pieces",
        choices=["pieces", "ids"],
        metavar="OUTPUT_TYPE",
        help="encode text to: pieces or ids. (default is pieces)",
    )

    args = parser.parse_args()

    predict(args)
