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
import gc
import hashlib
import itertools
import json
import logging
import pathlib
import tempfile
from os import makedirs, path

import numpy as np
from pecos.utils import smat_util
from pecos.utils.cluster_util import ClusterChain
from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.xmc import Indexer, LabelEmbeddingFactory
from pecos.xmc.xlinear import XLinearModel

LOGGER = logging.getLogger(__name__)


class CachedWorkspace(object):
    """Generate a folder name for caching intermediate variables"""

    def __init__(self, ws=None):
        """Initialization

        Args:
            ws (str, optional): Workspace folder
                If not given, will use a temporary folder.
        """
        if ws is None:
            self.tmp_folder = tempfile.TemporaryDirectory()
            ws = self.tmp_folder.name
        self.ws = ws

    def get_path_for_name_and_kwargs(self, name, kwargs):
        """Generate a hashed path in the workspace (self.ws)

        Args:
            name (str): a basename (e.g., the intermediate variable's name) for this folder.
            kwargs (dict): args that generates the variable

        Returns:
            A str: a hashed path for the given name and kwargs
        """
        folder = path.join(self.ws, name)
        makedirs(folder, exist_ok=True)
        # mapping.json is for debug purpose
        mapping_path = path.join(folder, "mapping.json")
        if path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f_map:
                mapping = json.loads(f_map.read())
        else:
            mapping = {}

        key = hashlib.sha224(json.dumps(kwargs, sort_keys=True).encode("utf-8")).hexdigest()
        mapping[key] = kwargs

        with open(mapping_path, "w", encoding="utf-8") as fout:
            fout.write(json.dumps(mapping, sort_keys=True, indent=2))
        return path.join(folder, key)


class Text2Text(object):
    """Given an input text, generate a subset of items relevant to this input from a fixed set of output items.

    The input should be a text sequence, while the output items can be either text-based or symbol-based
    (although the symbols are usually represented in text format).
    """

    def __init__(self, preprocessor=None, xlinear_models=None, output_items=None):
        """Initialization

        Args:
            preprocessor (Preprocessor): Text preprocessor.
            xlinear_models (list): List of XLinear models.
            output_items (list): List of output items name.
        """
        self.preprocessor = preprocessor
        self.xlinear_models = xlinear_models
        self.output_items = output_items

    def save(self, model_folder):
        """Save the Text2Text model

        Args:
            model_folder (str): folder name to save
        """

        LOGGER.info("Saving the model...")
        self.preprocessor.save(path.join(model_folder, "preprocessor"))
        xlinear_folder = path.join(model_folder, "xlinear_ensemble")
        ensemble_config = {"nr_ensembles": len(self.xlinear_models), "kwargs": []}
        for i, (m, kwargs) in enumerate(self.xlinear_models):
            ensemble_config["kwargs"] += [kwargs]
            folder = path.join(xlinear_folder, "{}".format(i))
            m.save(folder)
        with open(path.join(xlinear_folder, "config.json"), "w", encoding="utf-8") as fout:
            fout.write(json.dumps(ensemble_config, indent=True))
        with open(path.join(model_folder, "output_items.json"), "w", encoding="utf-8") as fp:
            json.dump(self.output_items, fp)

    @classmethod
    def load(cls, model_folder, is_predict_only=False, **kwargs):
        """Load the Text2Text model

        Args:
            model_folder (str): folder name to load
            is_predict_only (bool): if the loaded model will be used for prediction only in the batch mode.

        Returns:
            A Text2Text object
        """

        preprocessor = Preprocessor.load(path.join(model_folder, "preprocessor"))
        xlinear_folder = path.join(model_folder, "xlinear_ensemble")
        with open(path.join(xlinear_folder, "config.json"), "r", encoding="utf-8") as fin:
            ensemble_config = json.loads(fin.read())
        xlinear_models = []
        for i, model_kwargs in enumerate(ensemble_config["kwargs"]):
            folder = path.join(xlinear_folder, "{}".format(i))
            xlinear_models += [(XLinearModel.load(folder, is_predict_only, **kwargs), model_kwargs)]
        output_items = None
        folder_path = pathlib.Path(model_folder)
        json_output_items_filepath = folder_path / "output_items.json"
        with open(str(json_output_items_filepath), "r", encoding="utf-8") as fin:
            output_items = json.load(fin)
        if not output_items:
            raise ValueError("Could not read output items saved in json format")

        return cls(preprocessor, xlinear_models, output_items)

    @classmethod
    def train(
        cls,
        input_text_path,
        output_text_path,
        vectorizer_config=None,
        dtype=np.float32,
        label_embed_type=["pifa"],
        indexer_algo=["hierarchicalkmeans"],
        imbalanced_ratio=0.0,
        imbalanced_depth=100,
        spherical=True,
        nr_splits=2,
        max_leaf_size=[100],
        seed=[0],
        max_iter=[20],
        solver_type=["L2R_L2LOSS_SVC_DUAL"],
        Cp=[1.0],
        Cn=[1.0],
        bias=1.0,
        threshold=[0.1],
        negative_sampling_scheme="tfn",
        pred_kwargs=None,
        threads=-1,
        workspace_folder=None,
    ):
        """Train a Text2Text model

        Args:

            input_text_path (str): Text input file name.
                Format: in each line, OUTPUT_ID1,OUTPUT_ID2,OUTPUT_ID3,...\t INPUT_TEXT
                where OUTPUT_IDs are the zero-based output item indices
                corresponding to the line numbers of OUTPUT_ITEM_PATH.
                We assume utf-8 encoding for text.
            output_text_path (str): The file path for output text items.
                Format: each line corresponds to a representation
                of the output item. We assume utf-8 encoding for text.
            vectorizer_config_json (str): Json_format string for vectorizer config (default None)
            dtype (float32 | float64): data type (default float32)
            label_embed_type (list of str): Label embedding types. (default pifa).
                Multiple values will lead to different individual models for ensembling.
            indexer_algo (list of str): Indexer algorithm (default ["hierarchicalkmeans"]).
            imbalanced_ratio (float): Value between 0.0 and 0.5 (inclusive). Indicates how relaxed the balancedness
                constraint of 2_means can be. Specifically, if an iteration of 2_means is clustering L labels,
                the size of the output 2 clusters will be within approx imbalanced_ratio * 2 * L of each other.
                (default 0.0)
            imbalanced_depth (int): After hierarchical 2_means clustering has reached this depth,
                it will continue clustering as if imbalanced_ratio is set to 0.0. (default 100)
            spherical (bool): Do l2_normalize cluster centers while clustering (default True).
            nr_splits (int): number of splits used to construct hierarchy (a power of 2 is recommended, default 2)
            max_leaf_size (list of int): The max size of the leaf nodes of hierarchical 2_means clustering.
                Multiple values (separated by comma) are supported and will lead to different
                individual models for ensembling. (default [100])
            seed (list of int): Random seeds (default [0]). Multiple values will lead to different individual
                models for ensembling.
            max_iter (int): The max iteration for indexing (default 20)
            solver_type (list of string): solver type for ranking (default ["L2R_L2LOSS_SVC_DUAL"])
            Cp (float): Coefficient for positive class in the loss function (default 1.0)
            Cn (float): Coefficient for negative class in the loss function (default 1.0)
            bias (float): bias for the ranking model (default=1.0)
            threshold (float): Threshold to sparsify the model weights (default 0.1)
            negative_sampling (str, choices=[tfn, man, tfn+man]): Negative Sampling Schemes (default tfn)
            pred_kwargs (dict): kwargs for prediction used in matching-aware training
                only_topk (int): the default number of top labels used in the prediction
                beam_size (int): the default size of beam search used in the prediction
                post_processor (str): the default post processor used in the prediction
            workspace_folder: (str, default=None): A folder name for storing intermediate
                variables during training

        Returns:
            A Text2Text object
        """

        ws = CachedWorkspace(workspace_folder)

        # Train Preprocessor and obtain X, Y
        XY_kwargs = dict(
            input_text_path=input_text_path,
            output_text_path=output_text_path,
            vectorizer_config=vectorizer_config,
            dtype=str(dtype),
        )

        # Prepare Preprocessor
        preprocessor_path = ws.get_path_for_name_and_kwargs("preprocessor", XY_kwargs)
        if path.exists(preprocessor_path):
            LOGGER.info("Loading existing preprocessor...")
            preprocessor = Preprocessor.load(preprocessor_path)
        else:
            LOGGER.info("Parsing text files...")
            Y, corpus = Preprocessor.load_data_from_file(input_text_path, output_text_path)
            LOGGER.info(
                f"Training {vectorizer_config['type']} vectorizer on {len(corpus)} input texts..."
            )
            preprocessor = Preprocessor.train(corpus, vectorizer_config, dtype=dtype)
            preprocessor.save(preprocessor_path)

        # Prepare X, X could be dense or sparse
        X_path = ws.get_path_for_name_and_kwargs("X", XY_kwargs)

        if path.exists(X_path):
            X = XLinearModel.load_feature_matrix(X_path)
        else:
            if "corpus" not in locals():
                Y, corpus = Preprocessor.load_data_from_file(input_text_path, output_text_path)
            LOGGER.info(f"Vectorizing {len(corpus)} texts...")
            X = preprocessor.predict(corpus)
            XLinearModel.save_feature_matrix(X_path, X)
        LOGGER.info(
            f"{vectorizer_config['type']} input X loaded: {X.shape[0]} samples with {X.shape[1]} features."
        )

        # Prepare Y, Y is always sparse
        Y_path = ws.get_path_for_name_and_kwargs("Y", XY_kwargs) + ".npz"
        if path.exists(Y_path):
            Y = smat_util.load_matrix(Y_path)
        else:
            if "Y" not in locals():
                Y, corpus = Preprocessor.load_data_from_file(input_text_path, output_text_path)
            smat_util.save_matrix(Y_path, Y)
        LOGGER.info(f"Output label Y loaded: {Y.shape[0]} samples with {Y.shape[1]} labels.")

        # Grid Parameters for XLinearModel
        ranker_param_names = [
            "bias",
            "Cp",
            "Cn",
            "solver_type",
            "threshold",
            "negative_sampling_scheme",
            "pred_kwargs",
        ]

        ranker_grid_params = {}
        for name in ranker_param_names:
            tmp = locals()[name]
            ranker_grid_params[name] = tmp if isinstance(tmp, (list, tuple)) else [tmp]

        indexer_param_names = [
            "indexer_algo",
            "imbalanced_ratio",
            "imbalanced_depth",
            "spherical",
            "seed",
            "max_iter",
            "max_leaf_size",
            "nr_splits",
            "label_embed_type",
        ]

        indexer_grid_params = {}
        for name in indexer_param_names:
            tmp = locals()[name]
            indexer_grid_params[name] = tmp if isinstance(tmp, (list, tuple)) else [tmp]

        # Generate various label features
        label_feat_set = {}
        for embed_type in indexer_grid_params["label_embed_type"]:
            label_embed_kwargs = dict(
                input_text_path=input_text_path,
                output_text_path=output_text_path,
                dtype=str(dtype),
                vectorizer_config=vectorizer_config,
                embed_type=embed_type,
            )
            label_embed_path = ws.get_path_for_name_and_kwargs("L", label_embed_kwargs)
            if path.exists(label_embed_path):
                LOGGER.info(f"Loading existing {embed_type} features for {Y.shape[1]} labels...")
                label_feat_set[embed_type] = XLinearModel.load_feature_matrix(label_embed_path)
            else:
                LOGGER.info(f"Generating {embed_type} features for {Y.shape[1]} labels...")
                # Create label features
                label_feat_set[embed_type] = LabelEmbeddingFactory.create(Y, X, method=embed_type)
                XLinearModel.save_feature_matrix(label_embed_path, label_feat_set[embed_type])

        for indexer_values in itertools.product(
            *[indexer_grid_params[k] for k in indexer_param_names]
        ):
            # Indexing
            indexer_kwargs = dict(zip(indexer_param_names, indexer_values))
            indexer_kwargs_local = indexer_kwargs.copy()
            C_path = ws.get_path_for_name_and_kwargs("C", indexer_kwargs_local)
            if path.exists(C_path):
                LOGGER.info(f"Loading existing clustering code with params {indexer_kwargs_local}")
                C = ClusterChain.load(C_path)
            else:
                label_embed_type = indexer_kwargs.pop(
                    "label_embed_type", None
                )  # as label_embed_type is not a valid argument for XLinearModel.train
                LOGGER.info(f"Clustering with params {indexer_kwargs_local}...")
                C = Indexer.gen(
                    label_feat_set[indexer_kwargs_local["label_embed_type"]],
                    indexer_kwargs.pop("indexer_algo"),
                    threads=threads,
                    **indexer_kwargs,
                )
                LOGGER.info(f"Created {C[-1].shape[1]} clusters.")
                C.save(C_path)

            # Ensemble Models
            for ranker_values in itertools.product(
                *[ranker_grid_params[k] for k in ranker_param_names]
            ):
                ranker_kwargs = dict(zip(ranker_param_names, ranker_values))
                ranker_kwargs_local = ranker_kwargs.copy()
                # Model Training
                ranker_kwargs_local.update(indexer_kwargs_local)

                model_path = ws.get_path_for_name_and_kwargs("model", ranker_kwargs_local)
                if path.exists(model_path):
                    LOGGER.info(f"Model with params {ranker_kwargs_local} exists")
                else:
                    LOGGER.info(f"Training model with params {ranker_kwargs_local}...")
                    m = XLinearModel.train(
                        X,
                        Y,
                        C,
                        threads=threads,
                        **ranker_kwargs,
                    )
                    m.save(model_path)
                    del m
                    gc.collect()

            del C
            gc.collect()

        del X, Y, label_feat_set
        gc.collect()

        xlinear_models = []
        for indexer_values in itertools.product(
            *[indexer_grid_params[k] for k in indexer_param_names]
        ):
            indexer_kwargs = dict(zip(indexer_param_names, indexer_values))
            indexer_kwargs_local = indexer_kwargs.copy()
            for ranker_values in itertools.product(
                *[ranker_grid_params[k] for k in ranker_param_names]
            ):
                ranker_kwargs = dict(zip(ranker_param_names, ranker_values))
                ranker_kwargs_local = ranker_kwargs.copy()
                ranker_kwargs_local.update(indexer_kwargs_local)
                model_path = ws.get_path_for_name_and_kwargs("model", ranker_kwargs_local)
                xlinear_models += [(XLinearModel.load(model_path), ranker_kwargs_local)]

        # Load output items
        with open(output_text_path, "r", encoding="utf-8") as f:
            output_items = [q.strip() for q in f]

        return cls(preprocessor, xlinear_models, output_items)

    def predict(
        self, corpus, topk=10, beam_size=None, post_processor=None, threshold=None, **kwargs
    ):
        """Predict labels for given inputs

        Args:
            corpus (list of strings): input strings.
            topk (int, optional): override the only topk specified in the model
                Default None to disable overriding
            beam_size (int, optional): override the beam size specified in the model
                Default None to disable overriding
            post_processor (str, optional):  override the post_processor specified in the model
                Default None to disable overriding
            threshold (float, optional): Drop output items with scores less than this threshold among top-k items
                Default None to not threshold
            kwargs:
                post_processor (str, optional):  override the post_processor specified in the model
                    Default None to disable overriding
                threads (int, optional): the number of threads to use for predicting.
                    Default to -1 to use all.
        Returns:
            csr_matrix: predicted label matrix (num_samples x num_labels)
        """

        X = self.preprocessor.predict(corpus)
        Y_pred = smat_util.CsrEnsembler.average(
            *[
                m.predict(
                    X, only_topk=topk, beam_size=beam_size, post_processor=post_processor, **kwargs
                )
                for m, _ in self.xlinear_models
            ]
        )

        if threshold is not None:
            Y_pred.data[Y_pred.data <= threshold] = 0
            Y_pred.eliminate_zeros()

        return smat_util.sorted_csr(Y_pred, topk)

    def set_output_constraint(self, output_items_to_keep):
        """Prune the tree

        Args:
            output_items_to_keep (list of strings): A list of output items to be kept in the tree. The rest will be pruned.
        """
        output_items = dict()
        for i, item in enumerate(self.output_items):
            output_items[item] = i
        output_labels_to_keep = set()
        for item in output_items_to_keep:
            if item in output_items:
                output_labels_to_keep.add(output_items[item])
        for xlm, _ in self.xlinear_models:
            xlm.set_output_constraint(output_labels_to_keep)

    def get_output_item(self, output_id):
        """Get output item given an output id

        Args:
            output_id (int): output index

        Returns:
            A string: the output item corresponds to the input index
        """
        return self.output_items[output_id]

    def print_predictions(self, Y, fout, meta_info=None):
        """Dump predicted items

        Args:
            Y (csr_matrix): predicted label matrix
            fout (a file object): The file (or stdout) to dump
            meta_info (list of string, optional): meta_info to be incorporated into the output for each input.
                Default None

        """
        if meta_info is not None:
            assert Y.shape[0] == len(meta_info), "meta_info and Y should have the same sample size"

        for i in range(Y.shape[0]):
            output_data = {}
            output_data["schema"] = ["output_item", "score"]
            item_score = []
            idx = slice(Y.indptr[i], Y.indptr[i + 1])
            for output_id, score in zip(Y.indices[idx], Y.data[idx]):
                item_score.append(
                    [self.get_output_item(output_id), float(format(float(score), ".5g"))]
                )
            output_data["data"] = item_score
            if meta_info is not None:
                output_data["meta_info"] = meta_info[i]
            ss = json.dumps(output_data)
            fout.write(ss + "\n")
            fout.flush()
