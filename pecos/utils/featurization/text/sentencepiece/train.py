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

from pecos.utils import cli

from .model import SentencePieceModel


def train(args):
    """Train SentencePiece model

    Args:
        args (Namespace): Namespace object of training arguments from cli below.
    """

    model = SentencePieceModel.train(
        input_paths=args.input_paths,
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=args.shuffle_input_sentence,
        max_sentence_length=args.max_sentence_length,
        char_coverage=args.char_coverage,
    )

    model.save(args.model_folder)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SentencePiece: train tokenization model")

    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        required=True,
        metavar="MODEL_FOLDER",
        help="folder to save trained model and vocabulary",
    )

    parser.add_argument(
        "-i",
        "--input-paths",
        type=cli.comma_separated_type(str),
        required=True,
        metavar="PATHS",
        help="Paths to input text files. Multiple input files (separated by comma) are supported and will lead to a model trained using all input files. Input files are loaded in sequence.",
    )

    parser.add_argument(
        "-t",
        "--model-type",
        type=str,
        default="unigram",
        choices=["unigram", "bpe", "word", "char"],
        metavar="MODEL_TYPE",
        help="model algorithm: unigram, bpe, word for char (default is unigram)",
    )

    parser.add_argument(
        "-v",
        "--vocab-size",
        type=int,
        default=8000,
        metavar="VOCAB_SIZE",
        help="vocabulary size (default is 8000)",
    )

    parser.add_argument(
        "--char-coverage",
        type=float,
        default=0.9995,
        metavar="CHAR_COVERAGE",
        help="character coverage to determine the minimum symbols (default is 0.9995)",
    )

    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=4192,
        metavar="MAX_SENT_LENGTH",
        help="maximum length of sentence in byte (default is 4192)",
    )

    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=None,
        metavar="INPUT_SENT_SIZE",
        help="maximum size of sentences the trainer loads",
    )

    parser.add_argument(
        "--shuffle-input-sentence",
        type=cli.str2bool,
        default=True,
        metavar="SHUFFLE_INPUT_SENT",
        help="Randomly sample input sentences in advance (default is True)",
    )

    args = parser.parse_args()

    train(args)
