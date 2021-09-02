import os
import sys
import json
import array
import pickle
import logging
import pathlib
import numpy as np
from collections import defaultdict

import scipy.sparse as smat
from pure_sklearn.map import convert_estimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils import _IS_32BIT

import pecos.utils.featurization.text.vectorizers as pecos_vects


logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def _make_float_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array("f")


class TfidfQueryPrefix(pecos_vects.Vectorizer):
    """Vectorizer that processes Query and Prefix joined by a delim"""

    def __init__(
        self,
        model_query=None,
        model_prefix=None,
        delim="<@@>",
        max_prefix_len=None,
    ):
        """
        Parameters:
        ----------
        model_query: TfidfVectorizer object
            query encoder
        model_prefix: CountVectorizer object
            prefix encoder
        delim: str
            delim between query and prefix
        max_prefix_len: int
            if supplied then part of the prefix till this length
            is used for training and prediction. Default is None, in which case the full
            prefix is used.
        """
        self.model_query = model_query
        self.model_prefix = model_prefix
        self.delim = delim
        self.query_vocab_len = len(self.model_query.vocabulary_)
        self.prefix_vocab_len = len(self.model_prefix.vocabulary_)
        self.max_prefix_len = max_prefix_len

    def save(self, vectorizer_folder):
        """
        Save objects in pkl format.

        Parameters
        ----------
        vectorizer_folder: str
            Folder inside which to store the serialized objects as `vectorizer_<use_Case>.pkl`

        """
        os.makedirs(vectorizer_folder, exist_ok=True)
        with open(pathlib.Path(vectorizer_folder, "vectorizer_query.pkl"), "wb") as pfile:
            pickle.dump(self.model_query, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        with open(pathlib.Path(vectorizer_folder, "vectorizer_prefix.pkl"), "wb") as pfile:
            pickle.dump(self.model_prefix, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        with open(pathlib.Path(vectorizer_folder, "delim.json"), "w") as jfile:
            json.dump({"delim": self.delim}, jfile)
        with open(pathlib.Path(vectorizer_folder, "max_prefix_len.json"), "w") as jfile:
            json.dump({"max_prefix_len": self.max_prefix_len}, jfile)

    @classmethod
    def load(cls, vectorizer_folder, realtime=False):
        """
        Load a saved object.

        Parameters
        ----------
        vectorizer_folder: str
            Folder to load the model from
        realtime: bool
            if true then it is loaded for realtime inference

        """
        with open(pathlib.Path(vectorizer_folder, "vectorizer_query.pkl"), "rb") as pfile:
            model_query = pickle.load(pfile)
        with open(pathlib.Path(vectorizer_folder, "vectorizer_prefix.pkl"), "rb") as pfile:
            model_prefix = pickle.load(pfile)
        with open(pathlib.Path(vectorizer_folder, "delim.json"), "r") as jfile:
            delim = json.load(jfile)["delim"]
        try:
            with open(pathlib.Path(vectorizer_folder, "max_prefix_len.json"), "r") as jfile:
                max_prefix_len = json.load(jfile)["max_prefix_len"]
        except Exception:
            LOGGER.warning("max_prefix_len.json file not found. Max Prefix Len set to null")
            max_prefix_len = None
        if realtime:
            model_query = convert_estimator(model_query)
            model_prefix = convert_estimator(model_prefix)  # convert to predict only faster version
        return cls(
            model_query=model_query,
            model_prefix=model_prefix,
            delim=delim,
            max_prefix_len=max_prefix_len,
        )

    def predict(self, corpus):
        """Predict on corpus

        Parameters:
        ----------
        corpus: list/iterator
            corpus where each eliment is of the form <query><delim><prefix>

        Returns:
        -------
        encoded corpus
        """
        query_features = self.model_query.transform(
            [sample.split(self.delim)[0] for sample in corpus]
        )
        prefix_features = self.model_prefix.transform(
            [sample.split(self.delim)[1][: self.max_prefix_len] for sample in corpus]
        )
        query_features = self._convert_to_csr(query_features, self.query_vocab_len)
        prefix_features = self._convert_to_csr(prefix_features, self.prefix_vocab_len)
        return normalize(smat.hstack([query_features, prefix_features]), "l2", axis=1)

    def _convert_to_csr(self, features, dimension):
        """
        Helper function to convert dictionary of features to sparse csr_matrix

        Parameters:
        ----------
        features: list(dictionary) (or sparse csr matrix)
            a sparse matrix represented as list of dictionary, each element of the list is a row.
            Each row's dictionary has indices mapped to values
        dimension: int
            dimension 1 of the sparse csr matrix that is required

        Returns:
        -------
        sparse csr matrix.
        """
        if isinstance(features, smat.csr_matrix):
            return features
        data = []
        indices = []
        ptr = 0
        indptr = [ptr]
        for f in features:
            data += list(f.values())
            indices += list(f.keys())
            ptr += len(f)
            indptr.append(ptr)

        return smat.csr_matrix(
            (data, indices, indptr),
            shape=(len(features), dimension),
            dtype=np.float32,
        )


class PositionProductTfidf(TfidfVectorizer):
    """
    Tfidf vectorizer first creates a
    position discounted CountVectorizer where an ngram
    at position i counts for 1/(i+1) instead of 1 unit
    and then multiplies the discounted count vector with tfidf vector
    """

    def __init__(self, **kwargs):
        super(PositionProductTfidf, self).__init__(**kwargs)

    def fit_transform(self, raw_documents, y=None):
        """Fit and transform method for vectorizer.

        Parameters:
        -----------
        raw_documents: lst(str)
            list of raw documents for that need to be vectorized
        y: None
            this parameter is not needed here but kept for the sake of consistency
            with scikit learn.
            Similar to https://github.com/scikit-learn/
            scikit-learn/blob/0fb307bf3/sklearn/feature_extraction/text.py#L1808

        Returns:
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        self.fit(raw_documents=raw_documents, y=y)
        return self.transform(raw_documents)

    def _count_vocab_w_pos(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False"""
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = []
        indptr = []

        values = _make_float_array()
        indptr.append(0)
        for doc in raw_documents:
            feature_counter = {}
            curr_ngram_len = 0
            ngram_offset = 0
            # the analyze function breaks the doc into list of n-grams
            # starting with 1-gram, 2-gram and so on till n.
            # so if we want till 2-grams, then
            # "iphone" will become [i, p, h, o, n, e, ip, ph, ho, on, ne]
            # note that the position of ip is 0 and not its position in the above
            # list, so the extra logic is needed for position weighting
            for feat_pos, feature in enumerate(analyze(doc)):
                if len(feature) > curr_ngram_len:
                    curr_ngram_len = len(feature)
                    ngram_offset = feat_pos
                try:
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1 / (feat_pos - ngram_offset + 1)
                    else:
                        feature_counter[feature_idx] += 1 / (feat_pos - ngram_offset + 1)
                except KeyError:
                    # keyerror can occur only if fixed_vocab is true
                    # but it does not mean that it will necessarily occur
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behavior
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError(
                    "empty vocabulary; perhaps the documents only" " contain stop words"
                )

        if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
            if _IS_32BIT:
                raise ValueError(
                    (
                        "sparse CSR array has {} non-zero "
                        "elements and requires 64 bit indexing, "
                        "which is unsupported with 32 bit Python."
                    ).format(indptr[-1])
                )
            indices_dtype = np.int64

        else:
            indices_dtype = np.int32
        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.single)

        X = smat.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(vocabulary)),
            dtype=self.dtype,
        )
        X.sort_indices()
        return vocabulary, X

    def transform(self, corpus, **kwargs):
        """Transform documents to document-term matrix.
        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).

        Parameters:
        ----------
        corpus : iterable
            An iterable which yields either str, unicode or file objects.

        Returns:
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """

        self._check_vocabulary()
        # use the same matrix-building strategy as fit_transform
        _, X = self._count_vocab_w_pos(corpus, fixed_vocab=True)

        all_tfidf_vecs = self._tfidf.transform(X, copy=False)

        return all_tfidf_vecs


class TfidfQueryOnly(pecos_vects.Vectorizer):
    """Vectorizer that processes takes Query and Prefix joined by a delim
    as input but generates a vector using query ONLY"""

    def __init__(self, model_query=None, delim="<@@>"):
        """
        Parameters:
        ----------
        model_query: TfidfVectorizer object
            query encoder
        delim: str
            delim between query and prefix
        """
        self.model_query = model_query
        self.delim = delim

    def save(self, vectorizer_folder):
        """
        Save objects in pkl format.

        Parameters
        ----------
        vectorizer_folder: str
            Folder inside which to store the serialized objects as `vectorizer_<use_Case>.pkl`

        """
        os.makedirs(vectorizer_folder, exist_ok=True)
        with open(pathlib.Path(vectorizer_folder, "vectorizer_query.pkl"), "wb") as pfile:
            pickle.dump(self.model_query, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        with open(pathlib.Path(vectorizer_folder, "delim.json"), "w") as jfile:
            json.dump({"delim": self.delim}, jfile)

    @classmethod
    def load(cls, vectorizer_folder):
        """
        Load a saved object.

        Parameters
        ----------
        vectorizer_folder: str
            Folder to load the model from

        """
        with open(pathlib.Path(vectorizer_folder, "vectorizer_query.pkl"), "rb") as pfile:
            model_query = pickle.load(pfile)
        with open(pathlib.Path(vectorizer_folder, "delim.json"), "r") as jfile:
            delim = json.load(jfile)["delim"]
        return cls(model_query=model_query, delim=delim)

    @classmethod
    def train(cls, trn_corpus, config={}, dtype=np.float32):
        """Train vectorizer from corpus.

        Paraneters:
        ----------
        trn_corpus: list/iterator
            training corpus where each eliment is of the form <query><delim><prefix>
        config: dict
            config file for training with keys:
                config_query: configuration to set params for query encoder
                delim: delimiter to be used (default = <@@>)
        dtype: dtype object
            datatype for encoding returned

        Returns:
        -------
        trained object of cls
        """
        defaults_query = {
            "encoding": "utf-8",
            "strip_accents": "unicode",
            "stop_words": None,
            "ngram_range": (1, 1),
            "min_df": 1,
            "lowercase": True,
            "norm": "l2",
            "dtype": dtype,
        }
        delim = "<@@>"
        config_query = config.get("config_query", {})

        try:
            model_query = TfidfVectorizer(**{**defaults_query, **config_query})
        except TypeError:
            raise Exception(
                f"vectorizer config {config} contains unexpected keyword arguments for TfidfVectorizer"
            )
        if "delim" in config:
            delim = config["delim"]
        model_query.fit(list(set([sample.split(delim)[0] for sample in trn_corpus])))
        return cls(model_query, delim)

    def predict(self, corpus, **kwargs):
        """Predict on corpus

        Parameters:
        ----------
        corpus: list/iterator
            corpus where each eliment is of the form <query><delim><prefix>

        Returns:
        -------
        encoded corpus
        """
        query_features = self.model_query.transform(
            [sample.split(self.delim)[0] for sample in corpus]
        )
        query_features.sort_indices()
        return query_features
