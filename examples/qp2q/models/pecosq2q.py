"""The module contains a class to train pecos qp2q models."""
import copy
import json
import logging
import os
import pathlib
import gc
import glob

import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse as smat
from tqdm import tqdm
from abc import ABCMeta
from sklearn.base import BaseEstimator


from pecos.apps.text2text import Text2Text
from pecos.xmc import Indexer
from pecos.xmc import LabelEmbeddingFactory
from pecos.xmc.xlinear.model import XLinearModel
from pecos.utils.featurization.text.vectorizers import vectorizer_dict

from qp2q.models.indices import (
    TrieIndexer,
    HybridIndexer,
    HKMeans_w_MLC,
    build_prefix_mlc_mat,
    reduce_chain_len,
)
from qp2q.models.vectorizers import *

LOGGER = logging.getLogger(__name__)
MODEL_DICT = {}


class ModelMeta(ABCMeta):
    """Collects all tacos models in model_dict"""

    def __new__(cls, name, bases, attr):
        cls = super().__new__(cls, name, bases, attr)
        if cls.__name__ != "BaseModel":
            MODEL_DICT[cls.__name__.lower()] = cls
        return cls


class BaseModel(BaseEstimator, metaclass=ModelMeta):
    """Base class for TACOS models"""

    def __init__(self, **kwargs):
        super(BaseEstimator, self).__init__()
        pass

    def fit(self, X, y, **kwargs):
        pass

    @classmethod
    def load(cls, model_path):
        pass

    def save(self, model_path):
        pass

    def predict(self, X, **kwargs):
        pass


class PecosQP2QModel(BaseModel):
    """Class to train a pecos next query prediction model given input sparse-dataframe
    with prev_query,prefix as rows and next query as cols"""

    def __init__(
        self,
        model=None,
        vectorizer="SklearnTfidf",
        cluster_matrix=None,
        label_universe=None,
        query_feature_cols=None,
        vectorizer_class=None,
        vectorizer_config={"token_pattern": r"(?u)\b\w+\b", "tokenizer": None},
        load_cluster_matrix=False,
        load_trained_vectorizer=False,
        weighted_pifa=False,
        indexer_type="hierarchicalkmeans",
        spherical_clustering=True,
        fitted_=False,
        query_prefix_delimiter="<@@>",
    ):
        """
        Parameters
        ----------
        model: XLinearModel object, optional
            core pecos model used for training, prediction
        vectorizer: str
            vectorizer type to be used in the model
        cluster_matrix: list[csc matrix]/ cluster_chain from pecos.util/ None
            clustering matrix, the output of the pecos clustering step
        label_universe: list[str], optional
            list of queries to restrict label universe for pecos prediction
        query_feature_cols : list[str], optional
            list of query features to be used in training,
            should correspond to columns in query_features_df
        vectorizer_class : str, optional
            name of vectorizer class used in training, default is tfidf
        vectorizer_config : dict
            non-default vectorizer parameters loaded into vectorizer
        load_cluster_matrix : bool
            true if cluster matrix is loaded directly for training, default is false
        load_trained_vectorizer : bool
            true if trained vectorizer is loaded directly for training, default is false
        weighted_pifa : bool
            true if label embedding is weighted by score
        indexer_type : str
            indexer algorithm to be used during clustering, default is hierarchicalkmeans.
            This value must be present in indexers.Indexer.indexer_dict
        spherical_clustering : bool
            True if cluster centers are to be l2 normalized
        fitted_: bool
            if the model is fitted or not. a model is fitted if
            we have previously run the fit method.
        query_prefix_delimiter: str
            delimiter to separate query and prefix while
            representing them as rows of sparse_data_frame

        Notes:
        -----

        When load_trained_vectorizer is True,
        you must supply a trained vectorizer for the vectorizer argument.
        Similarly, when load_cluster_matrix is True, you must provide a cluster_matrix.

        Otherwise these objects are generated during fit and transform.
        """
        super().__init__()
        self.model = model
        self.vectorizer = vectorizer
        self.cluster_matrix = cluster_matrix
        self.label_universe = label_universe
        self.query_feature_cols = query_feature_cols
        self.vectorizer_class = vectorizer_class
        self.vectorizer_config = vectorizer_config
        self.load_cluster_matrix = load_cluster_matrix
        self.load_trained_vectorizer = load_trained_vectorizer
        self.weighted_pifa = weighted_pifa
        self.indexer_type = indexer_type
        self.spherical_clustering = spherical_clustering
        self.fitted_ = fitted_
        self.query_prefix_delimiter = query_prefix_delimiter

    @classmethod
    def load(cls, folder_path, realtime=False, query_prefix_delimiter="<@@>"):
        """
        Class method to load a pre-trained model from given path

        Parameters
        ----------
        folder_path : str
            path to the folder to load the model
        realtime: bool, default - False
            load in realtime inference mode or not
        query_prefix_delimiter: str
            delimiter will be placed between previous query and prefix during inference

        Returns
        -------
        PecosQP2QModel object
        """
        model = Text2Text.load(folder_path, is_predict_only=realtime)
        return cls(model=model, query_prefix_delimiter=query_prefix_delimiter)

    def save(self, folder_path):
        """
        Method to save a trained model and its attributes

        Parameters
        ------------
        folder_path : str
            path to the folder to save the model
        """
        os.makedirs(folder_path, exist_ok=True)

        self.model.save(pathlib.Path(folder_path, "xlinear_ensemble/0"))
        self.vectorizer.save(pathlib.Path(folder_path, "preprocessor"))
        self.cluster_matrix.save(pathlib.Path(folder_path, "cluster"))

        np.save(
            pathlib.Path(folder_path, "query_feature_cols"), np.asarray(self.query_feature_cols)
        )

        with open(pathlib.Path(folder_path, "output_items.json"), "w") as f:
            json.dump(self.label_universe, f)

        parameter_dict = self.get_params(deep=False)
        vectorizer_cls = type(self.vectorizer).__name__.lower()
        for key in list(parameter_dict):
            if not isinstance(parameter_dict[key], (bool, str)):
                parameter_dict.pop(key)
        parameter_dict["vectorizer_class"] = vectorizer_cls
        with open(pathlib.Path(folder_path, "model_config.json"), "w") as f:
            json.dump(parameter_dict, f)

        with open(pathlib.Path(folder_path, "preprocessor/config.json"), "w") as f:
            json.dump({"type": vectorizer_cls, "kwargs": {}}, f)

        with open(pathlib.Path(folder_path, "preprocessor/max_prefix_len.json"), "w") as f:
            json.dump({"max_prefix_len": None}, f)

        other_params = {
            "nr_ensembles": 1,
            "kwargs": [
                {
                    "bias": 1.0,
                    "Cp": 1.0,
                    "Cn": 1.0,
                    "solver_type": 1,
                    "threshold": 0.1,
                    "negative_sampling_scheme": "tfn",
                    "pred_kwargs": {"beam_size": 10, "only_topk": 20, "post_processor": "l3-hinge"},
                    "indexer_algo": "hierarchicalkmeans",
                    "imbalanced_ratio": 0.0,
                    "imbalanced_depth": 100,
                    "spherical": True,
                    "seed": 0,
                    "max_iter": 20,
                    "max_leaf_size": 100,
                    "label_embed_type": "pifa",
                }
            ],
        }
        with open(pathlib.Path(folder_path, "xlinear_ensemble/config.json"), "w") as f:
            json.dump(other_params, f)

        LOGGER.info(f"Saved model and its attributes to {folder_path}")

    def get_output_items(self, indices):
        """Get labels given indices.
        Parameters:
        ----------
        indices: list/ array
            list of indices in [0,1,...,#labels]

        Returns:
        -------
        labels corresponding to those indices
        """
        return [self.model.get_output_item(i) for i in indices]

    def get_suggestions(
        self,
        prev_query,
        prefix,
        topk=10,
        beam_size=10,
        max_prefix_len=None,
        max_query_tokens=100,
        n_threads=1,
    ):
        """
        Return predicted queries given prefix

        Parameters:
        ----------
        prev_query: str
            last query
        prefix: str
            prefix to be matched
        topk: int
            num. entries to get
        beam_size: int
            pecos beam size
        max_beam_size: int
            maximum pecos beam size
        max_prefix_len: int with default None
            skip inference on longer prefixes
        max_query_tokens: int
            do not attempt to run inference on a larger number of query tokens
        n_threads: int
            how many threads to use during inference

        Returns:
        -------
        suggested queries, list

        """
        split_prev_query = prev_query.split()
        if (
            sum(len(x) for x in split_prev_query) == 0
            or len(split_prev_query) > max_query_tokens
            or (max_prefix_len is not None and len(prefix) > max_prefix_len)
        ):
            return []

        text = [self.query_prefix_delimiter.join([prev_query, prefix])]
        params = {
            "beam_size": beam_size,
            "topk": beam_size * self.model.xlinear_models[0][1]["max_leaf_size"],
            "threads": n_threads,
        }

        try:
            results = self.model.predict(text, **params)
        except Exception as exc:
            LOGGER.exception("Unexpected exception, returning empty response.", exc_info=exc)
            return []

        out = []
        for i, idx in enumerate(results.indices):
            item = self.model.get_output_item(idx)
            if (
                not item.startswith(prefix) or len(item) == 0
            ):  # Filter-out suggestions that do not match prefix
                continue
            out.append((item, results.data[i]))
            if len(out) == topk:
                break
        return out

    def _build_clusters(
        self,
        label_features,
        max_leaf_size,
        max_iterations,
        seed,
        depth=-1,
        imbalanced_ratio=0.0,
        imbalanced_depth=100,
        nr_splits=2,
    ):
        """
        Method builds clusters using pifa embeddings of the labels.

        Parameters
        ----------
        label_features : csr matrix
            pifa embeddings of the labels
        indexer_algorithm: str
            option of clustering algorithm to use ('SKMEANS' or 'KMEANS')
        seed : int
            random seed
        max_leaf_size : int
            max size of leaf nodes in clustering
        max_iterations : int
            max iterations for the indexer
        depth : int
            Depth of trie: Useful only for TrieIndexer and HybridIndexer
        imbalanced_depth : int
            Parameter for Hierarchical k-means indexing
        imbalanced_ratio : float
            Parameter for Hierarchical k-means indexing
        nr_splits : int
            Parameter for Hierarchical k-means indexing
        Notes
        -----
        Computes and sets the cluster_matrix needed for training.

        """
        LOGGER.info("Creating index for training.")
        if (
            self.indexer_type not in Indexer.indexer_dict
            and self.indexer_type.lower() not in Indexer.indexer_dict
        ):
            raise ValueError(f"{self.indexer_type} is not supported in PECOS.")

        if self.indexer_type.lower() == "trieindexer":

            LOGGER.info("Depth of cluster matrix is = {}".format(depth))
            self.cluster_matrix = TrieIndexer.gen(
                feat_mat=None, label_strs=self.label_universe, depth=depth
            )

            LOGGER.info("Created cluster matrix")

        elif self.indexer_type.lower() == "hybridindexer":

            LOGGER.info("Depth of trie in hybrid index is = {}".format(depth))
            self.cluster_matrix = HybridIndexer.gen(
                feat_mat=smat.csr_matrix(label_features, dtype=sp.float32),
                label_strs=self.label_universe,
                depth=depth,
                max_leaf_size=max_leaf_size,
                seed=seed,
                max_iter=max_iterations,
                spherical=self.spherical_clustering,
            )
            LOGGER.info("Created cluster matrix")

        elif self.indexer_type.lower() == "hkmeans_w_mlc":
            LOGGER.info("Depth of constraint cluster matrix is = {}".format(depth))
            mlc_mats = build_prefix_mlc_mat(label_strs=self.label_universe, max_pref_len=depth)
            LOGGER.info("Created constraint cluster matrix")

            self.cluster_matrix = HKMeans_w_MLC.gen(
                feat_mat=smat.csr_matrix(label_features, dtype=sp.float32),
                mlc_mats=mlc_mats,
                use_freq=True,
                max_leaf_size=max_leaf_size,
                seed=seed,
                max_iter=max_iterations,
                spherical=self.spherical_clustering,
                imbalanced_depth=imbalanced_depth,
                imbalanced_ratio=imbalanced_ratio,
                nr_splits=nr_splits,
            )
            LOGGER.info("Length of cluster matrix = {}".format(len(self.cluster_matrix)))

        else:
            self.cluster_matrix = Indexer.gen(
                smat.csr_matrix(label_features, dtype=sp.float32),
                indexer_type=self.indexer_type.lower(),
                max_leaf_size=max_leaf_size,
                seed=seed,
                max_iter=max_iterations,
                spherical=self.spherical_clustering,
                imbalanced_depth=imbalanced_depth,
                imbalanced_ratio=imbalanced_ratio,
                nr_splits=nr_splits,
            )
            if depth != -1:
                LOGGER.info(
                    "Reducing length of cluster matrix from {} to {}".format(
                        len(self.cluster_matrix), depth
                    )
                )
                self.cluster_matrix = reduce_chain_len(
                    cluster_chain=self.cluster_matrix, max_depth=depth
                )

        try:
            LOGGER.info("Length of cluster matrix = {}".format(len(self.cluster_matrix)))
        except Exception as e:
            LOGGER.info("Error raised : {} ".format(str(e)))
            pass

    def fit(
        self,
        X,
        y,
        seed=121,
        Cp=1.0,
        Cn=1.0,
        n_jobs=16,
        threshold=0.1,
        max_iterations=20,
        max_leaf_size=100,
        dim_for_PIFA=None,
        label_text_features=None,
        depth=0,
        imbalanced_ratio=0.0,
        imbalanced_depth=100,
        nr_splits=2,
    ):
        """
        Class method to train the q2a model by breaking down the process into two steps: i. Clustering ii. Training.

        Parameters
        ----------
        X : csr matrix, dtype=np.float32
            training feature matrix
        y : csc matrix, dtype=np.float32
            training label matrix
        n_jobs: int
            number of threads to spawn, default = 16
        Cp : float
            coefficient for the positive class in the loss function, default = 1.0
        Cn : float
            coefficient for the negative class in the loss function, default = 1.0
        threshold : 0.1
            threshold to sparsify the model weights, default = 1.0
        seed : int
            random seed, default = 121
        max_leaf_size : int
            max size of leaf nodes in clustering, default = 100
        dim_for_PIFA : int or None
            If label_features is not None, then label embedding created using PIFA embedding upto dim = dim_for_PIFA
            concatenated with label_features
        max_iterations : int
            max iterations for the indexer, default = 20
        label_text_features:
            dense or sparse matrix with #rows = Number of labels
        depth : int
            Depth of trie: Useful only for TrieIndexer, HybridIndexer, and hkmeans_w_mlc
        imbalanced_depth : int
            Parameter for Hierarchical k-means indexing
        imbalanced_ratio : float
            Parameter for Hierarchical k-means indexing
        nr_splits : float
            Parameter for Hierarchical k-means indexing
        Returns
        -------
        fitted PecosQP2QModel object
        """

        LOGGER.info("Starting model training")
        LOGGER.info(f"Type(X) = {type(X)}, type(y) = {type(y)}")
        if not X.shape[0] == y.shape[0]:
            raise ValueError(
                f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) don't match"
            )

        if self.load_cluster_matrix:
            LOGGER.info("Using cluster matrix loaded during init")
        else:
            if self.indexer_type.lower() == "trieindexer":
                LOGGER.info("Generating index structure using Trie")
                self._build_clusters(
                    label_features=None,
                    max_leaf_size=max_leaf_size,
                    max_iterations=0,
                    seed=0,
                    depth=depth,
                )
            else:
                if self.weighted_pifa:
                    label_features = LabelEmbeddingFactory.pifa(X=X, Y=y)
                    y[y > 0] = 1
                else:
                    y[y > 0] = 1
                    label_features = LabelEmbeddingFactory.pifa(X=X, Y=y)

                if label_text_features is not None:
                    # Concat PIFA features with some features extracted from label text
                    if label_features.shape[0] != label_text_features.shape[0]:
                        raise ValueError(
                            "Some labels in the label_text_features do not have features in label_features or the other way"
                        )

                    LOGGER.info("Beginning to append label text feature to PIFA embedding")
                    label_features = smat.csr_matrix(
                        smat.hstack([label_features[:, :dim_for_PIFA], label_text_features])
                    )
                    LOGGER.info(
                        "Appending features from label text to PIFA features. Final feature mat dim is {}".format(
                            label_features.shape
                        )
                    )

                self._build_clusters(
                    label_features=label_features,
                    max_leaf_size=max_leaf_size,
                    max_iterations=max_iterations,
                    seed=seed,
                    depth=depth,
                    imbalanced_depth=imbalanced_depth,
                    imbalanced_ratio=imbalanced_ratio,
                    nr_splits=nr_splits,
                )
                del label_features
                gc.collect()

        LOGGER.info("Training XLinear model")
        self.model = XLinearModel.train(
            X,
            y,
            self.cluster_matrix,
            threads=n_jobs,
            Cp=Cp,
            Cn=Cn,
            threshold=threshold,
        )
        LOGGER.info("Training complete")
        self.fitted_ = True
        return self

    def predict(
        self,
        samples,
        query_features_df=None,
        topk=10,
        beam_size=10,
        batch_size=10000,
        labels_to_keep=[],
    ):
        """
        Method to generate candidates for a list of input samples (queries).

        Parameters
        ----------
        samples : list[str]
            list of samples (queries)
        query_features_df : pandas dataframe object
            dataframe indexed by query with columns as query features, default = None
        topk: int
            number of outputs to be considered for each sample, default = 10
        beam_size: int
            beam width for prediction, default = 10
        batch_size: int
            size of batches to be predicted sequentially, default = 10000
        labels_to_keep : list[str]
            list of asins to restrict the label universe for prediction, default = None

        Returns
        -------
        Y : csr matrix
            sparse Q-A matrix with pecos scores
        """

        if not self.fitted_:
            raise ValueError("Error: model has not been trained")
        if labels_to_keep:
            self.model.set_output_constraint(labels_to_keep)
            LOGGER.info(f"Restricted labels universe to {len(labels_to_keep)} asins")

        X = self.vectorizer.predict(samples)
        if isinstance(query_features_df, pd.DataFrame):
            X = self._combine_query_features(X, query_features_df)
            LOGGER.info(
                f"Added {len(self.query_feature_cols)} query features to the prediction feature matrix"
            )
        num_queries = X.shape[0]
        LOGGER.info(f"Inferring pecos scores for {num_queries} samples.")
        batch_indices = np.array_split(np.arange(num_queries), max(num_queries // batch_size, 1))
        X = smat.csr_matrix(X, dtype=np.float32)
        gc.collect()
        Y_list = []
        for i in tqdm(range(len(batch_indices))):
            X_current = X[batch_indices[i], :]
            X_current.sort_indices()
            Y_list.append(
                self.model.predict(
                    X_current,
                    beam_size=beam_size,
                    only_topk=topk,
                )
            )
        Y = smat.csr_matrix(smat.vstack(Y_list))
        LOGGER.info("Model prediction complete")
        return Y

    def transform(self, sparse_data_frame, query_features_df=None, inplace=False):
        """
        Method to featurize input sparse data and create X, y matrices for training.

        Parameters
        ----------
        sparse_data_frame : SparseDataFrame object
            query-asin sparse matrix used for model training
        query_features_df: pandas dataframe object
            dataframe must be indexed by query, default = None
        inplace: bool
            if false: a copy of sparse_data_frame data matrix is made for constructing y
            if true: the changes are done in place

        Returns
        -------
        X : csr matrix
            feature matrix
        y : csc matrix
            label matrix

        Notes
        -----
        There is an option to load a pretrained vectorizer to featurize the Q-A matrix and use non-text query features if available.
        """
        self.label_universe = list(sparse_data_frame.i2c.values())
        LOGGER.info(f"Number of asins in label universe: {len(self.label_universe)}")

        if self.load_trained_vectorizer:
            trained_vectorizer = self.vectorizer
            LOGGER.info("Loaded pre-trained vectorizer")
        else:
            if self.vectorizer.lower() not in vectorizer_dict:
                raise ValueError(
                    f"Invalid vectorizer type {self.vectorizer}, ensure vectorizer class inherits PECOS vectorizer"
                )
            self.vectorizer = vectorizer_dict[self.vectorizer.lower()].train(
                sparse_data_frame.i2r.values(), self.vectorizer_config
            )
            trained_vectorizer = self.vectorizer
            LOGGER.info(f"Finished training {self.vectorizer}...")

        X = trained_vectorizer.predict(list(sparse_data_frame.i2r.values()))
        if isinstance(query_features_df, pd.DataFrame):
            X = self._combine_query_features(X, query_features_df)
            LOGGER.info(
                f"Added {len(self.query_feature_cols)} query features to the training feature matrix"
            )
        LOGGER.info(f"Shape of feature matrix (X) : {X.shape}")
        if inplace:
            y = sparse_data_frame.data_matrix
        else:
            y = copy.deepcopy(sparse_data_frame.data_matrix)
        X = smat.csr_matrix(X, dtype=np.float32)
        y = smat.csc_matrix(y, dtype=np.float32)
        gc.collect()
        return X, y
