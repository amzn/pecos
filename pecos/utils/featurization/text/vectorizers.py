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
import json
import logging
import os
import pathlib
import pickle
from abc import ABCMeta

import numpy as np
import torch
from pecos.core import clib
from pecos.utils import torch_util
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import (
    AlbertConfig,
    AlbertModel,
    AlbertTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaModel,
    XLMRobertaTokenizer,
    XLMConfig,
    XLMModel,
    XLMTokenizer,
    XLNetConfig,
    XLNetModel,
    XLNetTokenizer,
)

vectorizer_dict = {}

LOGGER = logging.getLogger(__name__)


class VectorizerMeta(ABCMeta):
    """Metaclass for keeping track of all `Vectorizer` subclasses."""

    def __new__(cls, name, bases, attr):
        cls = super().__new__(cls, name, bases, attr)
        if cls.__name__ != "Vectorizer":
            vectorizer_dict[cls.__name__.lower()] = cls
        return cls


class Vectorizer(metaclass=VectorizerMeta):
    """Wrapper class for all vectorizers."""

    def __init__(self, config, model):
        """Initialization

        Args:
            config (dict): Dict with key `"type"` and value being the lower-cased name of the specific vectorizer class to use.
                Also contains keyword arguments to pass to the specified vectorizer.
            model (Vectorizer): Trained vectorizer.
        """

        self.config = config
        self.model = model

    def save(self, vectorizer_folder):
        """Save trained vectorizer to disk.

        Args:
            vectorizer_folder (str): Folder to save to.
        """

        os.makedirs(vectorizer_folder, exist_ok=True)
        with open(os.path.join(vectorizer_folder, "config.json"), "w", encoding="utf-8") as fout:
            fout.write(json.dumps(self.config))
        self.model.save(vectorizer_folder)

    @classmethod
    def load(cls, vectorizer_folder):
        """Load a saved vectorizer from disk.

        Args:
            vectorizer_folder (str): Folder where `Vectorizer` was saved to using `Vectorizer.save`.

        Returns:
            Vectorizer: The loaded object.
        """

        config_path = os.path.join(vectorizer_folder, "config.json")
        if not os.path.exists(config_path):
            # to maintain compatibility with previous versions of pecos models
            config = {"type": "tfidf", "kwargs": {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())
        vectorizer_type = config.get("type", None)
        assert vectorizer_type is not None, f"{vectorizer_folder} is not a valid vectorizer folder"
        assert vectorizer_type in vectorizer_dict, f"invalid vectorizer type {config['type']}"
        model = vectorizer_dict[vectorizer_type].load(vectorizer_folder)
        return cls(config, model)

    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list or str): Training corpus in the form of a list of strings or path to text file.
            config (dict, optional): Dict with key `"type"` and value being the lower-cased name of the specific vectorizer class to use.
                Also contains keyword arguments to pass to the specified vectorizer. Default behavior is to use tfidf vectorizer with default arguments.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            Vectorizer: Trained vectorizer.
        """

        config = config if config is not None else {"type": "tfidf", "kwargs": {}}
        LOGGER.debug(f"Train Vectorizer with config: {json.dumps(config, indent=True)}")
        vectorizer_type = config.get("type", None)
        assert (
            vectorizer_type is not None
        ), f"config {config} should contain a key 'type' for the vectorizer type"
        assert vectorizer_type in vectorizer_dict, f"invalid vectorizer type {config['type']}"
        assert (
            isinstance(trn_corpus, list) or vectorizer_type == "tfidf"
        ), "only tfidf support from file training"
        model = vectorizer_dict[vectorizer_type].train(
            trn_corpus, config=config["kwargs"], dtype=dtype
        )
        return cls(config, model)

    def predict(self, corpus, **kwargs):
        """Vectorize a corpus.

        Args:
            corpus (list or str): List of strings to vectorize or path to text file.
            **kwargs: Keyword arguments to pass to the trained vectorizer.

        Returns:
            numpy.ndarray or scipy.sparse.csr.csr_matrix: Matrix of features.
        """

        if isinstance(corpus, str) and self.config["type"] != "tfidf":
            raise ValueError("Iterable over raw text expected for vectorizer other than tfidf.")
        return self.model.predict(corpus, **kwargs)

    @staticmethod
    def load_config_from_args(args):
        """Parse config from a `argparse.Namespace` object.

        Args:
            args (argparse.Namespace): Contains either a `vectorizer_config_path` (path to a json file) or `vectorizer_config_json` (a json object in string form).

        Returns:
            dict: The dict resulting from loading the json file or json object.

        Raises:
            Exception: If json object cannot be loaded.
        """

        if args.vectorizer_config_path is not None:
            with open(args.vectorizer_config_path, "r", encoding="utf-8") as fin:
                vectorizer_config_json = fin.read()
        else:
            vectorizer_config_json = args.vectorizer_config_json

        try:
            vectorizer_config = json.loads(vectorizer_config_json)
        except json.JSONDecodeError as jex:
            raise Exception(
                "Failed to load vectorizer config json from {} ({})".format(
                    vectorizer_config_json, jex
                )
            )
        return vectorizer_config


class Tfidf(Vectorizer):
    """Multithreaded tfidf vectorizer with C++ backend.

    Supports 'word', 'char' and 'char_wb' tokenization.
    """

    def __init__(self, model=None):
        """Initialization

        Args:
            model (ctypes.c_void_p): pointer to C instance tfidf::Vectorizer
        """
        self.model = model

    def __del__(self):
        """Destruct self model instance"""
        clib.tfidf_destruct(self.model)

    def save(self, save_dir):
        """Save trained tfidf vectorizer to disk.

        Args:
            save_dir (str): Folder to save the model.
        """
        os.makedirs(save_dir, exist_ok=True)
        clib.tfidf_save(self.model, save_dir)

    @classmethod
    def load(cls, load_dir):
        """Load a Tfidf vectorizer from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            Tfidf: The loaded object.
        """
        if not os.path.exists(load_dir):
            raise ValueError(f"tfidf model not exist at {load_dir}")
        return cls(clib.tfidf_load(load_dir))

    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list of str or str): Training corpus in the form of a list of strings or path to corpus file/folder.
            config (dict): Dict with keyword arguments to pass to C++ class tfidf::Vectorizer.
                The keywords are:
                    ngram_range (tuple of int): (min_ngram, max_ngram)
                    truncate_length (int): sequence truncation length, set to negative to disable
                    max_feature (int): maximum number of features allowed, set to 0 to disable
                    min_df_ratio (float, [0, max_df_ratio)): min ratio for document frequency truncation
                    max_df_ratio (float, (min_df_ratio, 1]): max ratio for document frequency truncation
                    min_df_cnt (int, [0, max_df_cnt)): min count for document frequency truncation
                    max_df_cnt (int, (min_df_cnt, Inf)): max count for document frequency truncation. Default -1 to disable.
                    binary (bool): whether to binarize term frequency, default False
                    use_idf (bool): whether to use inverse document frequency, default True
                    smooth_idf (bool): whether to smooth IDF by adding 1 to all DF counts, default True
                    add_one_idf (bool): whether to smooth IDF by adding 1 to all IDF scores, default False
                    sublinear_tf (bool): whether to use sublinear mapping (log) on term frequency, default False
                    keep_frequent_feature (bool): if max_feature > 0, will only keep max_feature features by
                                    ignoring features with low document frequency (if True, default),
                                    ignoring features with high document frequency (if False)
                    norm (str, 'l1' or 'l2'): feature vector will have unit l1 or l2 norm
                    analyzer (str, 'word', 'char' or 'char_wb'): Whether to use word or character n-grams.
                                    Option ‘char_wb’ creates character n-grams only from text inside word boundaries,
                                    n-grams at the edges of words are padded with single space.
                    buffer_size (int): if train from file, number of bytes allocated for file I/O. Set to 0 to use default value.
                    threads (int): number of threads to use, set to negative to use all
            dtype (np.dtype): The data type to use. Default to `np.float32`.

        Note:
            stop word removal: simultaneously satisfy count and ratio constraint.
                i.e. will use max(min_df_cnt, min_df_ratio * nr_doc) as final min_df_cnt
                and min(max_df_cnt, max_df_ratio * nr_doc) as final max_df_cnt

        Returns:
            Tfidf: Trained vectorizer.
        """
        DEFAULTS = {
            "ngram_range": (1, 1),
            "truncate_length": -1,
            "max_feature": 0,
            "min_df_ratio": 0.0,
            "max_df_ratio": 1.0,
            "min_df_cnt": 0,
            "max_df_cnt": -1,
            "binary": False,
            "use_idf": True,
            "smooth_idf": True,
            "add_one_idf": False,
            "sublinear_tf": False,
            "keep_frequent_feature": True,
            "norm": "l2",
            "analyzer": "word",
            "buffer_size": 0,
            "threads": -1,
        }

        DEFAULTS_META = {
            "norm_p": 2,
            "buffer_size": 0,
            "threads": -1,
            "base_vect_configs": [DEFAULTS],
        }

        def check_base_config_key(base_config):
            unexpected_keys = []
            for key in base_config:
                if key not in DEFAULTS:
                    unexpected_keys.append(key)
            if len(unexpected_keys) > 0:
                raise ValueError(f"Unknown argument: {unexpected_keys}")
            return {**DEFAULTS, **base_config}

        if "base_vect_configs" not in config:
            config = check_base_config_key(config)
        else:
            for idx, base_config in enumerate(config["base_vect_configs"]):
                base_config = check_base_config_key(base_config)
                config["base_vect_configs"][idx] = base_config
            config = {**DEFAULTS_META, **config}

        cmodel = clib.tfidf_train(trn_corpus, config)

        return cls(cmodel)

    def predict(self, corpus, **kwargs):
        """Vectorize a corpus.

        Args:
            corpus (list): List of strings to vectorize.
            **kwargs:
                threads (int, default -1): number of threads to use for predict, set to negative to use all

        Returns:
            scipy.sparse.csr.csr_matrix: Matrix of features.
        """
        return clib.tfidf_predict(
            self.model,
            corpus,
            buffer_size=kwargs.get("buffer_size", 0),
            threads=kwargs.get("threads", -1),
        )


class SklearnTfidf(Vectorizer):
    """Sklearn tfidf vectorizer"""

    def __init__(self, model=None):
        """Initialization

        Args:
            model (sklearn.feature_extraction.text.TfidfVectorizer, optional): The trained tfidf vectorizer. Default is `None`.
        """

        self.model = model

    def save(self, vectorizer_folder):
        """Save trained sklearn Tfidf vectorizer to disk.

        Args:
            vectorizer_folder (str): Folder to store serialized object in.
        """

        os.makedirs(vectorizer_folder, exist_ok=True)
        with open(os.path.join(vectorizer_folder, "vectorizer.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, vectorizer_folder):
        """Load a saved sklearn Tfidf vectorizer from disk.

        Args:
            vectorizer_folder (str): Folder where `SklearnTfidf` object was saved to using `SklearnTfidf.save`.

        Returns:
            SklearnTfidf: The loaded object.
        """

        vectorizer_path = os.path.join(vectorizer_folder, "vectorizer.pkl")
        assert os.path.exists(vectorizer_path), "vectorizer path {} does not exist".format(
            vectorizer_path
        )
        with open(vectorizer_path, "rb") as fvec:
            return cls(pickle.load(fvec))

    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's TfidfVectorizer.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            Tfidf: Trained vectorizer.

        Raises:
            Exception: If `config` contains keyword arguments that the tfidf vectorizer does not accept.
        """
        defaults = {
            "encoding": "utf-8",
            "strip_accents": "unicode",
            "stop_words": None,
            "ngram_range": (1, 1),
            "min_df": 1,
            "lowercase": True,
            "norm": "l2",
            "dtype": dtype,
        }
        try:
            model = TfidfVectorizer(**{**defaults, **config})
        except TypeError:
            raise Exception(
                f"vectorizer config {config} contains unexpected keyword arguments for TfidfVectorizer"
            )
        model.fit(trn_corpus)
        return cls(model)

    def predict(self, corpus, **kwargs):
        """Vectorize a corpus.

        Args:
            corpus (list): List of strings to vectorize.
            **kwargs: Ignored.

        Returns:
            scipy.sparse.csr.csr_matrix: Matrix of features.
        """
        result = self.model.transform(corpus)
        # Indices must be sorted for C++ batch code to work
        result.sort_indices()
        return result


class SklearnHashing(Vectorizer):
    """Sklearn hashing vectorizer"""

    def __init__(self, model=None):
        """Initialization

        Args:
            model (sklearn.feature_extraction.text.HashingVectorizer, optional): The trained hashing vectorizer. Default is `None`.
        """
        self.model = model

    def save(self, vectorizer_folder):
        """Save trained sklearn hashing vectorizer to disk.

        Args:
            vectorizer_folder (str): Folder to store serialized object in.
        """

        os.makedirs(vectorizer_folder, exist_ok=True)
        with open(os.path.join(vectorizer_folder, "vectorizer.pkl"), "wb") as fout:
            pickle.dump(self.model, fout)

    @classmethod
    def load(cls, vectorizer_folder):
        """Load a saved sklearn hashing vectorizer from disk.

        Args:
            vectorizer_folder (str): Folder where `SklearnHashing` object was saved to using `SklearnHashing.save`.

        Returns:
            SklearnHashing: The loaded object.
        """

        vectorizer_path = os.path.join(vectorizer_folder, "vectorizer.pkl")
        assert os.path.exists(vectorizer_path), "vectorizer path {} does not exist".format(
            vectorizer_path
        )
        with open(vectorizer_path, "rb") as fvec:
            return cls(pickle.load(fvec))

    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dict with keyword arguments to pass to sklearn's HashingVectorizer.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            Hashing: Trained vectorizer.

        Raises:
            Exception: If `config` contains keyword arguments that the hashing vectorizer does not accept.
        """

        defaults = {
            "encoding": "utf-8",
            "strip_accents": "unicode",
            "stop_words": None,
            "ngram_range": (1, 2),
            "lowercase": True,
            "norm": "l2",
            "dtype": dtype,
            "n_features": 1048576,  # default number in HashingVectorizer
        }
        try:
            model = HashingVectorizer(**{**defaults, **config})
        except TypeError:
            raise Exception(
                f"vectorizer config {config} contains unexpected keyword arguments for HashingVectorizer"
            )
        model.fit(trn_corpus)
        return cls(model)

    def predict(self, corpus, **kwargs):
        """Vectorize a corpus.

        Args:
            corpus (list): List of strings to vectorize.
            **kwargs: Ignored.

        Returns:
            scipy.sparse.csr.csr_matrix: Matrix of features.
        """
        result = self.model.transform(corpus)
        # Indices must be sorted for C++ batch code to work
        result.sort_indices()
        return result


class TransformerModelClass(object):
    """Utility class for representing a Transformer and tokenizer."""

    def __init__(self, config_class, model_class, tokenizer_class):
        """Initialization

        Args:
            config_class (transformers.configuration_utils.PretrainedConfig)
            model_class (transformers.modeling_utils.PreTrainedModel)
            tokenizer_class (transformers.tokenization_utils.PreTrainedTokenizer)
        """
        self.config_class = config_class
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class


class PretrainedTransformer(Vectorizer):
    """Vectorizer with a variety of Transformer models."""

    MODEL_CLASSES = {
        "albert": TransformerModelClass(AlbertConfig, AlbertModel, AlbertTokenizer),
        "bert": TransformerModelClass(BertConfig, BertModel, BertTokenizer),
        "distilbert": TransformerModelClass(
            DistilBertConfig,
            DistilBertModel,
            DistilBertTokenizer,
        ),
        "roberta": TransformerModelClass(RobertaConfig, RobertaModel, RobertaTokenizer),
        "xlm-roberta": TransformerModelClass(
            XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer
        ),
        "xlm": TransformerModelClass(XLMConfig, XLMModel, XLMTokenizer),
        "xlnet": TransformerModelClass(XLNetConfig, XLNetModel, XLNetTokenizer),
    }

    def __init__(self, model=None, tokenizer=None, transformer_options=None):
        """Initialization

        Args:
            model (transformers.modeling_utils.PreTrainedModel)
            tokenizer (transformers.tokenization_utils.PreTrainedTokenizer)
            transformer_options (dict)
        """

        self.tokenizer = tokenizer
        self.model = model
        self.transformer_options = transformer_options

    def save(self, vectorizer_folder):
        """Save the pre-trained model and tokenizer to disk.

        Args:
            vectorizer_folder (str): Folder to save to.
        """

        os.makedirs(vectorizer_folder, exist_ok=True)
        model_folder = os.path.join(vectorizer_folder, "model")
        tokenizer_folder = os.path.join(vectorizer_folder, "tokenizer")
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(tokenizer_folder, exist_ok=True)
        # this creates transformer_options.json
        with open(
            os.path.join(vectorizer_folder, "transformer_options.json"),
            "w",
            encoding="utf-8",
        ) as fout:
            fout.write(json.dumps(self.transformer_options))
        # this creates config.json, pytorch_model.bin
        self.model.save_pretrained(model_folder)
        # this creates added_tokens.json, special_tokens_map.json, tokenizer_config.json, vocab.txt
        self.tokenizer.save_pretrained(tokenizer_folder)

    @classmethod
    def load(cls, vectorizer_folder):
        """Load the pre-trained model and tokenizer from disk.

        Args:
            vectorizer_folder (str): Folder which was saved to using `PretrainedTransformer.save`.

        Returns:
            PretrainedTransformer: Loaded object.
        """

        model_folder = os.path.join(vectorizer_folder, "model")
        tokenizer_folder = os.path.join(vectorizer_folder, "tokenizer")

        assert os.path.isdir(model_folder), "pre-trained model folder {} does not exist".format(
            model_folder
        )
        assert os.path.isdir(
            tokenizer_folder
        ), "pre-trained tokenizer folder {} does not exist".format(tokenizer_folder)

        # Load from transformer_options.json
        with open(
            os.path.join(vectorizer_folder, "transformer_options.json"),
            "r",
            encoding="utf-8",
        ) as fin:
            transformer_options = json.loads(fin.read())
        dnn_type = cls.MODEL_CLASSES[transformer_options["transformer_type"]]
        # Load from config.json
        model_config = dnn_type.config_class.from_pretrained(model_folder)
        # Load from vocab.txt [,added_tokens.json, special_tokens_map.json, tokenizer_config.json]
        tokenizer = dnn_type.tokenizer_class.from_pretrained(tokenizer_folder)
        # Load from pytorch_model.bin
        model = dnn_type.model_class.from_pretrained(model_folder, config=model_config)
        return cls(model, tokenizer, transformer_options)

    @classmethod
    def train(cls, trn_corpus, config=None, dtype=np.float32):
        """Download a Transformer model.

        Args:
            trn_corpus (list): Training corpus in the form of a list of strings.
            config (dict): Dictionary containing keywords/value for training. The keywords are:
                transformer_type (str): Transformer model type (default 'bert').

                norm (str): embedding normalization method, set to None to skip {'l1','l2'} (default 'l2')

                pooling (str): pooling method {'mean','mask-mean','first','last','cls'} (default 'mean')

                model_shortcut: can be either
                        [1] str (default): pre-trained transformer model name shortcut (default 'bert-base-uncased')
                                model configuration & tokenizer & model weights will be downloaded or load from
                                cache_dir (if previously downloaded).
                        [2] dictionary: a dictionary containing paths to load model configuration, vocabulary
                                for tokenizer and pre-trained model weights
                                (e.g. {"config_path": "path/to/config.json",
                                        "vocab_path": "path/to/vocab.txt",
                                        "weight_path": "path/to/pytorch_model.bin" })

                do_fine_tune (bool): whether to fine tune the pre-trained Transformer (default False). Not yet implemented.

                cache_dir (str): cache directory to save downloaded model, set to None (default) to use a temporary folder.

        Returns:
            PretrainedTransformer: Pre-trained model.
        """

        defaults = {
            "transformer_type": "bert",
            "norm": "l2",
            "pooling": "mean",
            "model_shortcut": "bert-base-uncased",
            "do_fine_tune": False,
            "cache_dir": None,
        }
        config = {**defaults, **config}
        transformer_type = config["transformer_type"]
        model_shortcut = config["model_shortcut"]
        cache_dir = config["cache_dir"]

        assert (
            transformer_type in cls.MODEL_CLASSES
        ), "Unsupported model type [{}], available Transformer model types: {}".format(
            transformer_type, cls.MODEL_CLASSES.keys()
        )
        dnn_type = cls.MODEL_CLASSES[transformer_type]

        if isinstance(model_shortcut, str):  # download everything
            # download model config
            model_config = dnn_type.config_class.from_pretrained(
                model_shortcut, cache_dir=cache_dir
            )
            # download tokenizer
            tokenizer = dnn_type.tokenizer_class.from_pretrained(
                model_shortcut, cache_dir=cache_dir
            )
            # download model weights
            model = dnn_type.model_class.from_pretrained(
                model_shortcut, config=model_config, cache_dir=cache_dir
            )
        else:  # load everything locally
            model_config_path = model_shortcut["config_path"]
            vocab_path = model_shortcut["vocab_path"]
            weight_path = model_shortcut["weight_path"]
            assert pathlib.Path(model_config_path).is_file(), "model config not found at {}".format(
                model_config_path
            )
            model_config = dnn_type.config_class.from_pretrained(model_config_path)
            assert pathlib.Path(vocab_path).is_file(), "vocabulary not found at {}".format(
                vocab_path
            )
            tokenizer = dnn_type.tokenizer_class.from_pretrained(vocab_path)
            assert pathlib.Path(weight_path).is_file(), "model weights not found at {}".format(
                weight_path
            )
            model = dnn_type.model_class.from_pretrained(weight_path, config=model_config)

        if config["do_fine_tune"]:
            # train the model!
            raise NotImplementedError("Training for Transformer vectorizer not implemented!")

        return cls(model, tokenizer, config)

    def predict(
        self,
        corpus,
        batch_size=8,
        truncate_length=300,
        use_gpu_if_available=True,
        **kwargs,
    ):
        """Vectorizer a corpus.

        Args:
            corpus (list): List of strings to vectorize.
            batch_size (int, optional): Default is 8.
            truncate_length (int, optional): Default is 300.
            use_gpu_if_available (bool, optional): Default is True.

        Returns:
            numpy.ndarray: Matrix of features.
        """

        if self.model.config.max_position_embeddings > 0:
            truncate_length = min(truncate_length, self.model.config.max_position_embeddings)

        # generate feature batches
        feature_tensors = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=corpus,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
            add_special_tokens=True,
            max_length=truncate_length,
            truncation=True,
            padding="longest",
        )
        # setup device
        device, n_active_gpu = torch_util.setup_device(use_gpu_if_available=use_gpu_if_available)
        # start eval
        transformer_type = self.transformer_options["transformer_type"]
        norm = self.transformer_options["norm"]
        pooling = self.transformer_options["pooling"]

        batch_size = batch_size * max(1, n_active_gpu)
        data = TensorDataset(
            feature_tensors["input_ids"],
            feature_tensors["attention_mask"],
            feature_tensors["token_type_ids"],
        )

        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, num_workers=4)

        # multi-gpu eval
        if n_active_gpu > 1 and not isinstance(self.model, torch.nn.parallel.DataParallel):
            model = torch.nn.parallel.DataParallel(self.model)
        else:
            model = self.model

        model.eval()
        model.to(device)
        embeddings = []
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                if transformer_type == "distilbert":
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                elif transformer_type in [
                    "bert",
                    "roberta",
                    "xlm-roberta",
                    "albert",
                    "xlm",
                    "xlnet",
                ]:
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        token_type_ids=inputs["token_type_ids"],
                    )
                else:
                    raise NotImplementedError(
                        "Unsupported transformer_type {}".format(transformer_type)
                    )

                # get the embeddings from model output
                # REF: https://huggingface.co/transformers/v2.3.0/model_doc/bert.html#bertmodel
                # For bert,roberta,xlm-roberta,albert:  outputs = last_hidden_states, pooled_output, (hidden_states)
                # For xlm,xlnet,distilbert: outputs = last_hidden_states, (hidden_states), (attentions)
                if pooling == "mean":
                    pooled_output = outputs[0].mean(dim=1)
                elif pooling == "mask-mean":
                    last_hidden_states = torch_util.apply_mask(outputs[0], inputs["attention_mask"])
                    pooled_output = last_hidden_states.sum(dim=1)
                    masked_length = inputs["attention_mask"].sum(dim=1)
                    pooled_output = pooled_output / masked_length.unsqueeze(1).float()
                elif pooling == "first":
                    pooled_output = outputs[0][:, 0, :]
                elif pooling == "last":
                    pooled_output = outputs[0][:, -1, :]
                elif pooling == "cls":
                    assert transformer_type in [
                        "bert",
                        "roberta",
                        "xlm-roberta",
                        "albert",
                    ], "Only {} models have [CLS] token.".format(
                        ["bert", "roberta", "xlm-roberta", "albert"]
                    )
                    # get the [CLS] embedding for the document
                    pooled_output = outputs[1]
                else:
                    raise NotImplementedError("Unsupported pooling method {}".format(pooling))

                embeddings.append(pooled_output.cpu().numpy())

        # construct dense output
        embeddings = np.concatenate(embeddings, axis=0)
        if norm is not None:
            embeddings = normalize(embeddings, norm=norm, axis=1, copy=False)
        return embeddings
