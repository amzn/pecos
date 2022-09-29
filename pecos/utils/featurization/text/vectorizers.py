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
import pickle
from abc import ABCMeta

import numpy as np
from pecos.core import clib
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


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
