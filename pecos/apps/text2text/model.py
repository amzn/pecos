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
import dataclasses as dc
import gc
import hashlib
import json
import logging
import pathlib
import tempfile
from os import makedirs, path

import numpy as np
import pecos
from pecos.utils import smat_util
from pecos.utils.cluster_util import ClusterChain
from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.xmc import Indexer, LabelEmbeddingFactory
from pecos.xmc.xlinear import XLinearModel
from pecos.xmc.base import HierarchicalKMeans

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

    @dc.dataclass
    class TrainParams(pecos.BaseParams):
        """Training Parameters of Text2Text.

        indexer_params (HierarchicalKMeans.TrainParams): params to generate hierarchial label tree.
        xlinear_params (XLinearModel.TrainParams): train params for XLinearModel
        """

        indexer_params: HierarchicalKMeans.TrainParams = None  # type: ignore
        xlinear_params: XLinearModel.TrainParams = None  # type: ignore

    @dc.dataclass
    class PredParams(pecos.BaseParams):
        """Pred Parameters of XTransformer.

        xlinear_params (XLinearModel.PredParams): pred params for linear ranker
        """

        xlinear_params: XLinearModel.PredParams = None  # type: ignore

        def override_with_kwargs(self, pred_kwargs):
            """override pred_params with kwargs.

            Args:
                pred_kwargs:
                    beam_size (int): the beam search size.
                        Overrides only_topk for all models except for the bottom one.
                    only_topk (int): the final topk predictions to generate.
                        Overrides only_topk for bottom model.
                    post_processor (str): post processor scheme for prediction.
                        Overrides post_processor for all models.
            """
            self.xlinear_params.override_with_kwargs(pred_kwargs)
            return self

    def get_pred_params(self):
        """get pred_params saved in the model"""
        ret_pred_params = self.PredParams(
            xlinear_params=self.xlinear_models[0][0].get_pred_params(),
        )
        return ret_pred_params

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

        if len(xlinear_models) > 1:
            LOGGER.warning(
                f"Deprecation warning: loaded {len(xlinear_models)} xlinear models, multi-model ensemble prediction will be deprecated in future releases."
            )

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
        label_embed_type="pifa",
        vectorizer_config=None,
        train_params=None,
        pred_params=None,
        workspace_folder=None,
        **kwargs,
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
            label_embed_type (list of str): Label embedding types. (default pifa).
                We support pifa, pifa_lf_concat::Z=path, and pifa_lf_convex_combine::Z=path::alpha=scalar_value.
                Multiple values will lead to different individual models for ensembling.
            vectorizer_config_json (str): Json_format string for vectorizer config (default None)
            train_params (Text2Text.TrainParams): params to train Text2Text model
            pred_params (Text2Text.PredParams): params to predict Text2Text model
            workspace_folder: (str, default=None): A folder name for storing intermediate
                variables during training
            kwargs:
                {"beam_size": INT, "only_topk": INT, "post_processor": STR},
                    Default None to use HierarchicalMLModel.PredParams defaults

        Returns:
            A Text2Text object
        """

        ws = CachedWorkspace(workspace_folder)
        dtype = np.float32

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
            parsed_result = Preprocessor.load_data_from_file(input_text_path, output_text_path)
            Y = parsed_result["label_matrix"]
            R = parsed_result["label_relevance"]
            corpus = parsed_result["corpus"]

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
                parse_result = Preprocessor.load_data_from_file(input_text_path, output_text_path)
                Y = parse_result["label_matrix"]
                R = parse_result["label_relevance"]
                corpus = parse_result["corpus"]
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
                parsed_result = Preprocessor.load_data_from_file(input_text_path, output_text_path)
                Y = parsed_result["label_matrix"]
                R = parsed_result["label_relevance"]
            smat_util.save_matrix(Y_path, Y)
        LOGGER.info(f"Output label Y loaded: {Y.shape[0]} samples with {Y.shape[1]} labels.")

        # Prepare R, R should have same sparsity pattern as Y
        R_path = ws.get_path_for_name_and_kwargs("R", XY_kwargs) + ".npz"
        if path.exists(R_path):
            R = smat_util.load_matrix(R_path)
        else:
            if "R" not in locals():
                parsed_result = Preprocessor.load_data_from_file(input_text_path, output_text_path)
                R = parsed_result["label_relevance"]
            if R is not None:
                smat_util.save_matrix(R_path, R)
        if R is not None:
            LOGGER.info(f"Relevance matrix R loaded, cost sensitive learning enabled.")

        # construct indexing, training and prediction params
        if train_params is None:
            # fill all BaseParams class with their default value
            train_params = cls.TrainParams.from_dict(dict(), recursive=True)
        else:
            train_params = cls.TrainParams.from_dict(train_params)

        # construct pred_params
        if pred_params is None:
            # fill all BaseParams with their default value
            pred_params = cls.PredParams.from_dict(dict(), recursive=True)
        else:
            pred_params = cls.PredParams.from_dict(pred_params)
        pred_params = pred_params.override_with_kwargs(kwargs)

        # 1. Generate label features
        label_embed_kwargs = dict(
            input_text_path=input_text_path,
            output_text_path=output_text_path,
            dtype=str(dtype),
            vectorizer_config=vectorizer_config,
            embed_type=label_embed_type,
        )
        label_embed_path = ws.get_path_for_name_and_kwargs("L", label_embed_kwargs)
        if path.exists(label_embed_path):
            LOGGER.info(f"Loading existing {label_embed_type} features for {Y.shape[1]} labels...")
            label_feat = XLinearModel.load_feature_matrix(label_embed_path)
        else:
            LOGGER.info(f"Generating {label_embed_type} features for {Y.shape[1]} labels...")
            # parse embed_type string, expect either the following three cases:
            # (1) pifa
            # (2) pifa_lf_concat::Z=path
            # (3) pifa_lf_convex_combine::Z=path::alpha=value
            lemb_key_val_list = label_embed_type.split("::")
            lemb_type = lemb_key_val_list[0]
            lemb_kwargs = {}
            for key_val_str in lemb_key_val_list[1:]:
                key, val = key_val_str.split("=")
                if key == "Z":
                    Z = smat_util.load_matrix(val)
                    lemb_kwargs.update({"Z": Z})
                elif key == "alpha":
                    alpha = float(val)
                    lemb_kwargs.update({"alpha": alpha})
                else:
                    raise ValueError(f"key={key}, val={val} is not supported!")
            if "lf" in lemb_type and lemb_kwargs.get("Z", None) is None:
                raise ValueError(
                    "pifa_lf_concat/pifa_lf_convex_combine must provide external path for Z."
                )
            # Create label features
            label_feat = LabelEmbeddingFactory.create(
                Y,
                X,
                method=lemb_type,
                **lemb_kwargs,
            )
            XLinearModel.save_feature_matrix(label_embed_path, label_feat)

        # 2. Indexing
        indexer_kwargs_dict = train_params.indexer_params.to_dict()
        C_path = ws.get_path_for_name_and_kwargs("C", indexer_kwargs_dict)
        if path.exists(C_path):
            LOGGER.info(f"Loading existing clustering code with params {indexer_kwargs_dict}")
            C = ClusterChain.load(C_path)
        else:
            C = Indexer.gen(label_feat, train_params=train_params.indexer_params)
            LOGGER.info("Hierarchical label tree: {}".format([cc.shape[0] for cc in C]))
            C.save(C_path)

        del label_feat
        gc.collect()

        # Ensemble Models
        m = XLinearModel.train(
            X,
            Y,
            C=C,
            R=R,
            train_params=train_params.xlinear_params,
            pred_params=pred_params.xlinear_params,
            pred_kwargs=kwargs,
        )

        xlinear_models = [[m, train_params.to_dict()]]

        # Load output items
        with open(output_text_path, "r", encoding="utf-8") as f:
            output_items = [q.strip() for q in f]

        return cls(preprocessor, xlinear_models, output_items)

    def predict(self, corpus, threshold=None, **kwargs):
        """Predict labels for given inputs

        Args:
            corpus (list of strings): input strings.
            threshold (float, optional): Drop output items with scores less than this threshold among top-k items
                Default None to not threshold
            kwargs:
                only_topk (int, optional): override the only topk specified in the model
                    Default None to disable overriding
                beam_size (int, optional): override the beam size specified in the model
                    Default None to disable overriding
                post_processor (str, optional):  override the post_processor specified in the model
                    Default None to disable overriding
                threads (int, optional): the number of threads to use for predicting.
                    Default to -1 to use all.
        Returns:
            csr_matrix: predicted label matrix (num_samples x num_labels)
        """

        X = self.preprocessor.predict(corpus)
        Y_pred = [m.predict(X, **kwargs) for m, _ in self.xlinear_models]

        if len(Y_pred) > 1:
            Y_pred = smat_util.CsrEnsembler.average(*Y_pred)
        else:
            Y_pred = Y_pred[0]

        if threshold is not None:
            Y_pred.data[Y_pred.data <= threshold] = 0
            Y_pred.eliminate_zeros()

        return smat_util.sorted_csr(Y_pred, only_topk=kwargs.get("only_topk", None))

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
