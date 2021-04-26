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
import contextlib
import logging
import os
import shutil
import sys
import tempfile

import sentencepiece as spm


class SentencePieceModel(object):
    """SentencePiece Model

    This is a wrapper for Sentencepiece text tokenizer.

    For complete list of training and prediction arguments,
    see https://github.com/google/sentencepiece
    """

    def __init__(self, model, model_folder):
        """
        Args:
            model (spm.SentencePieceProcessor): Trained SentencePiece model.
            model_folder (tempfile.TemporaryDirectory): Temporary directory object.

        Attributes:
            model (spm.SentencePieceProcessor): Trained SentencePiece model.
            model_folder (tempfile.TemporaryDirectory): Temporary directory object.
        """

        assert isinstance(model, spm.SentencePieceProcessor)
        assert isinstance(model_folder, tempfile.TemporaryDirectory)
        self.model = model
        self.model_folder = model_folder

    def save(self, save_folder):
        """Save trained model and vocabulary.

        Args:
            save_folder (str): Path to folder to save trained model and vocabulary.
        """

        logger = logging.getLogger(__name__)
        os.makedirs(save_folder, exist_ok=True)
        model_path = os.path.join(save_folder, "spm.model")
        vocab_path = os.path.join(save_folder, "spm.vocab")
        if os.path.exists(model_path):
            logger.warning(f"Overwrite existing Sentencepiece model at {model_path}")
        if os.path.exists(vocab_path):
            logger.warning(f"Overwrite existing Sentencepiece vocabulary at {vocab_path}")

        curr_model_path = os.path.join(self.model_folder.name, "spm.model")
        shutil.copyfile(curr_model_path, model_path)
        curr_vocab_path = os.path.join(self.model_folder.name, "spm.vocab")
        if os.path.exists(curr_vocab_path):
            shutil.copyfile(curr_vocab_path, vocab_path)

    @classmethod
    def load(cls, model_folder):
        """Load saved model and vocabulary.

        Args:
            model_folder (str): Path to saved folder with trained model and vocabulary (optional).

        Returns:
            SentencePieceModel object.
        """

        model_path = os.path.join(model_folder, "spm.model")
        vocab_path = os.path.join(model_folder, "spm.vocab")
        model = spm.SentencePieceProcessor()
        model.Load(model_path)

        tmp_folder = tempfile.TemporaryDirectory()
        tmp_model_path = os.path.join(tmp_folder.name, "spm.model")
        shutil.copyfile(model_path, tmp_model_path)

        if os.path.exists(vocab_path):
            tmp_vocab_path = os.path.join(tmp_folder.name, "spm.vocab")
            shutil.copyfile(vocab_path, tmp_vocab_path)
        return cls(model, tmp_folder)

    @classmethod
    def train(
        cls,
        input_paths,
        model_type="unigram",
        input_sentence_size=None,
        shuffle_input_sentence=True,
        max_sentence_length=4192,
        vocab_size=8000,
        char_coverage=0.9995,
        num_threads=None,
        user_defined_symbols=[],
        extra_args=[],
    ):
        """Train SentencePiece model.

        Args:
            input_paths (list): Paths to input files used for training sentencepiece model.
            input_sentence_size (int, optional): Sentencepiece trainer only loads the first `input_sentence_size` sentences.
            shuffle_input_sentence (bool, optional): Randomly sample input sentences in advance. Valid when `input_sentence_size` > 0.
            max_sentence_length (int, optional): Maximum length of sentence in byte.
            vocab_size (int, optional): Vocabulary size.
            char_coverage (float, optional): Character coverage to determine the minimum symbols.
            num_threads (int, optional): Number of threads for training.
            user_defined_symbols (list, optional): A list of user defined symbols, each enclosed in angle brackets e.g.<symbol-0>.
                User defined symbols are handled as one piece in any context, i.e. if it's included in the input text,
                it's always extracted as one piece.
            extra_args (list, optional): List of strings of extra arguments that can be passed directly to Sentencepiece.

        Returns:
            SentencePieceModel object.
        """

        tmp_folder = tempfile.TemporaryDirectory()
        model_prefix = os.path.join(tmp_folder.name, "spm")

        if isinstance(input_paths, str):
            input_paths = [input_paths]

        assert model_type in ["unigram", "bpe", "char", "word"]

        cmd = []
        cmd += ["--pad_id=0"]
        cmd += ["--bos_id=1"]
        cmd += ["--eos_id=2"]
        cmd += ["--unk_id=3"]
        cmd += ["--input={}".format(",".join(input_paths))]
        cmd += ["--model_prefix={}".format(model_prefix)]
        cmd += ["--model_type={}".format(model_type)]
        cmd += ["--max_sentence_length={}".format(int(max_sentence_length))]
        cmd += ["--minloglevel=1"]
        cmd += ["--vocab_size={}".format(int(vocab_size))]
        cmd += ["--character_coverage={}".format(float(char_coverage))]

        if input_sentence_size is not None:
            cmd += ["--input_sentence_size={}".format(int(input_sentence_size))]

        if shuffle_input_sentence:
            cmd += ["--shuffle_input_sentence=true"]

        if num_threads is not None:
            cmd += ["--num_threads={}".format(int(num_threads))]

        if len(user_defined_symbols) > 0:
            cmd += ["--user_defined_symbols=" + ",".join(user_defined_symbols)]

        if isinstance(extra_args, str):
            if len(extra_args) > 0:
                extra_args = [extra_args]
            else:
                extra_args = []
        cmd += extra_args

        cmd = " ".join(cmd)
        SentencePieceModel._train_raw(cmd)

        return SentencePieceModel.load(tmp_folder.name)

    def predict(self, input_lines, to_ids=False):
        """Tokenize lines of text

        Args:
            input_lines (list): List of texts to tokenize.
            to_ids (bool, optional): Whether to return IDs or text pieces of tokens.

        Returns:
            list: Each item is a list of the tokenization of the  corresponding text in input_lines.
        """

        return self.encode(input_lines, to_ids)

    def predict_file(self, input_file="-", output_file="-", to_ids=False):
        """Tokenize file

        Args:
            input_file (str, optional): Path to input text file. '-' denotes stdin.
            output_file (str, optional): Path to output file. Each line in the output file is the tokenization of the corresponding line in the input file. Within a line, predicted pieces/IDs are separated by single white space.
            to_ids (bool, optional): Whether to output IDs or text pieces of tokens.
        """

        return self.encode_file(input_file, output_file, to_ids)

    def encode_file(self, input_file="-", output_file="-", to_ids=False):
        """Tokenize file

        Args:
            input_file (str, optional): Path to input text file. '-' denotes stdin.
            output_file (str, optional): Path to output file. Each line in the output file is the tokenization of the corresponding line in the input file. Within a line, predicted pieces/IDs are separated by single white space.
            to_ids (bool, optional): Whether to output IDs or text pieces of tokens.
        """

        if to_ids:
            encode = self._encode_to_ids
        else:
            encode = self._encode_to_pieces

        with contextlib.ExitStack() as stack:
            if input_file == "-":
                fin = sys.stdin
            else:
                fin = stack.enter_context(open(input_file, "r", encoding="utf-8"))
            if output_file == "-":
                fout = sys.stdout
            else:
                fout = stack.enter_context(open(output_file, "w", encoding="utf-8"))

            for line in fin:
                encoded_output = encode(line.strip())
                output_line = "{}\n".format(" ".join(map(str, encoded_output)))
                fout.write(output_line)

    def encode(self, input_lines, to_ids=False):
        """Tokenize lines of text

        Args:
            input_lines (list): List of texts to tokenize.
            to_ids (bool, optional): Whether to return IDs or text pieces of tokens.

        Returns:
            list: Each item is a list of the tokenization of the  corresponding text in input_lines.
        """

        if to_ids:
            encode = self._encode_to_ids
        else:
            encode = self._encode_to_pieces
        if isinstance(input_lines, str):
            input_lines = [input_lines]
        outputs = []
        for line in input_lines:
            outputs.append(encode(line.strip()))
        return outputs

    def decode_file(self, input_file="-", output_file="-", from_ids=False):
        """De-tokenize file

        Args:
            input_file (str, optional): Path to input text file of encoding. Encoded pieces/IDs within a line are separated by single whitespace.
            output_file (str, optional): Path to output file. Each line in the output file is the decoding of the corresponding line in the input file.
            from_ids (bool, optional): Whether encoded file contains IDs/text pieces of tokens.
        """

        if from_ids:
            decode = self._decode_from_ids
        else:
            decode = self._decode_from_pieces

        with contextlib.ExitStack() as stack:
            if input_file == "-":
                fin = sys.stdin
            else:
                fin = stack.enter_context(open(input_file, "r", encoding="utf-8"))
            if output_file == "-":
                fout = sys.stdout
            else:
                fout = stack.enter_context(open(output_file, "w", encoding="utf-8"))

            for line in fin:
                encoded_input = line.strip().split(" ")
                if from_ids:
                    encoded_input = list(map(int, encoded_input))
                decoded_output = decode(encoded_input)
                output_line = "{}\n".format(decoded_output)
                fout.write(output_line)

    def decode(self, inputs, from_ids=False):
        """De-tokenize list of encoded IDs/text pieces of tokens

        Args:
            inputs (list): List of lists, each inner list is the encoded pieces/IDs of one raw string
            from_ids (bool, optional): Whether encoded list contains IDs or text pieces.

        Returns:
            list: List of string, each string is the decoding of one inner list.
        """

        if from_ids:
            decode = self._decode_from_ids
        else:
            decode = self._decode_from_pieces
        output_lines = []
        for one_input in inputs:
            output_lines.append(decode(one_input))
        return output_lines

    @staticmethod
    def _train_raw(train_args):
        """Call SentencePiece's SentencePieceTrainer

        Args:
            train_args (str): SentencePieceTrainer arguments.
        """

        assert isinstance(train_args, str)
        spm.SentencePieceTrainer.Train(train_args)

    def _encode_to_pieces(self, raw):
        """Call SentencePiece's EncodeAsPieces to tokenize into text pieces.

        Args:
            raw (str): Text to tokenize into pieces.

        Returns:
            list: Tokenized text pieces.
        """

        return self.model.EncodeAsPieces(raw)

    def _encode_to_ids(self, raw):
        """Call SentencePiece's EncodeAsIds to tokenize into IDs.

        Args:
            raw (str): Text to tokenize into IDs.

        Returns:
            list: Tokenized integer IDs.
        """

        return self.model.EncodeAsIds(raw)

    def _decode_from_pieces(self, pieces):
        """Call SentencePiece's DecodePieces to de-tokenize from text pieces.

        Args:
            pieces (list): List of text pieces.

        Returns:
            str: De-tokenized text.
        """

        return self.model.DecodePieces(pieces)

    def _decode_from_ids(self, ids):
        """Call SentencePiece's DecodePieces to de-tokenize from IDs.

        Args:
            ids (list): List of integer IDs.

        Returns:
            str: De-tokenized text.
        """

        return self.model.DecodeIds(ids)
