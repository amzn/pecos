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
from copy import deepcopy
from glob import glob

import dataclasses as dc
import numpy as np
import pecos
import scipy.sparse as smat
import torch
from pecos.core import clib
from pecos.utils import smat_util, torch_util
from pecos.utils.cluster_util import ClusterChain
from pecos.xmc.base import HierarchicalMLModel
from pecos.xmc.xlinear.model import XLinearModel
from sklearn.preprocessing import normalize as sk_normalize

from .matcher import TransformerMatcher
from .module import MLProblemWithText

LOGGER = logging.getLogger(__name__)


class XTransformer(pecos.BaseClass):
    """Hierarchical-XTransformer for XMC.
    Consists of one or more TransformerMatcher models followed by an optional XLinearModel.

    See more details in Section 5 of PECOS paper (Yu et al., 2020) and X-Transformer paper (Chang et al., 2020).

    PECOS: Prediction for Enormous and Correlated Output Spaces
        Hsiang-Fu Yu, Kai Zhong, Inderjit S. Dhillon
        https://arxiv.org/abs/2010.05878

    Taming Pre-trained Transformers for Extreme Multi-label Text Classification (KDD 2020)
        Wei-Cheng Chang, Hsiang-Fu Yu, Kai Zhong, Yiming Yang, Inderjit Dhillon
        https://arxiv.org/abs/1905.02331
    """

    @dc.dataclass
    class TrainParams(pecos.BaseParams):
        """Training Parameters of XTransformer.

        model_dir (str, optional): the i-th trained matcher will be saved to 'model_dir/{i}.model'
        ranker_level (int, optional): number of ranker levels. Default 1 to use a single layer ranker
        matcher_params_chain (TransformerMatcher.TrainParams or list): chain of params for TransformerMatchers.
        ranker_params (XLinearModel.TrainParams): train params for linear ranker
        """

        model_dir: str = ""
        ranker_level: int = 1
        matcher_params_chain: TransformerMatcher.TrainParams = None  # type: ignore
        ranker_params: XLinearModel.TrainParams = None  # type: ignore

    @dc.dataclass
    class PredParams(pecos.BaseParams):
        """Pred Parameters of XTransformer.

        matcher_params_chain (TransformerMatcher.PredParams or list): chain of params for TransformerMatchers
        ranker_params (XLinearModel.PredParams): pred params for linear ranker
        """

        matcher_params_chain: TransformerMatcher.PredParams = None  # type: ignore
        ranker_params: XLinearModel.PredParams = None  # type: ignore

        def override_with_kwargs(self, pred_kwargs, no_ranker=False):
            """override pred_params with kwargs.

            Args:
                pred_kwargs:
                    beam_size (int): the beam search size.
                        Overrides only_topk for all models except for the bottom one.
                    only_topk (int): the final topk predictions to generate.
                        Overrides only_topk for bottom model.
                    post_processor (str): post processor scheme for prediction.
                        Overrides post_processor for all models.
                no_ranker (bool, optional): if there is a linear ranker in the
                    model list. Used to decide which one is the bottom model.
            """
            if pred_kwargs is not None:
                if not isinstance(pred_kwargs, dict):
                    raise TypeError("type(pred_kwargs) must be dict")
                overridden_beam_size = pred_kwargs.get("beam_size", None)
                overridden_only_topk = pred_kwargs.get("only_topk", None)
                overridden_post_processor = pred_kwargs.get("post_processor", None)
                depth = len(self.matcher_params_chain)
                for d in range(depth):
                    if overridden_beam_size:
                        if no_ranker and d == depth - 1:
                            continue
                        self.matcher_params_chain[d].only_topk = overridden_beam_size
                    if overridden_post_processor:
                        self.matcher_params_chain[d].post_processor = overridden_post_processor
                if no_ranker:
                    if overridden_only_topk:
                        self.matcher_params_chain[-1].only_topk = overridden_only_topk
                else:
                    self.ranker_params.override_with_kwargs(pred_kwargs)
            return self

    def __init__(self, model_list):
        """Initialization

        Args:
            model_list (list of TransformerMatcher or XLinearModel): List of model.
        """
        self.model_list = model_list

    @property
    def depth(self):
        """Get the depth of the hierarchical model

        Returns:
            depth (int): Depth of model.
        """
        return len(self.model_list)

    @property
    def nr_labels(self):
        """Get the number of labels

        Returns:
            nr_labels (int): Number of labels.
        """
        return self.model_list[-1].nr_labels

    def save(self, save_dir):
        """Save the X-Transformer model to file.

        Args:
            save_dir (str): dir to save the model, will be created if not exist
                save params to save_dir/param.json
                save model_list[i] to save_dir/{i}.model
        """
        os.makedirs(save_dir, exist_ok=True)
        params = {
            "model": self.__class__.__name__,
            "depth": self.depth,
            "nr_labels": self.nr_labels,
        }
        params = self.append_meta(params)
        param_dir = os.path.join(save_dir, "param.json")
        with open(param_dir, "w", encoding="utf-8") as fpa:
            fpa.write(json.dumps(params, indent=True))
        LOGGER.info("Parameters saved to {}".format(param_dir))

        for i, model in enumerate(self.model_list):
            model_dir = os.path.join(save_dir, "{}.model".format(i))
            model.save(model_dir)
            LOGGER.info("Model {}({}) saved to {}".format(i, type(model), model_dir))

    @classmethod
    def load(cls, load_dir):
        """Load X-Transformer model from file

        Args:
            load_dir (str): dir to load the model

        Returns:
            XTransformer
        """
        if not os.path.isdir(load_dir):
            raise ValueError(f"load dir does not exist at: {load_dir}")

        param_dir = os.path.join(load_dir, "param.json")
        with open(param_dir, "r", encoding="utf-8") as fpa:
            params = json.loads(fpa.read())
        LOGGER.info("Params loaded from {}".format(param_dir))

        depth = int(params.get("depth", len(glob("{}/*.model".format(load_dir)))))
        model_list = []
        for i in range(depth):
            model_dir = os.path.join(load_dir, "{}.model".format(i))
            # load params for single model and get model type
            with open(os.path.join(model_dir, "param.json"), "r", encoding="utf-8") as fin:
                model_params = json.loads(fin.read())
            model_class = eval(model_params["__meta__"]["class_fullname"].split("###")[-1])
            cur_model = model_class.load(model_dir)
            LOGGER.info("Model {}({}) loaded from {}".format(i, type(cur_model), model_dir))
            model_list.append(cur_model)

        return cls(model_list)

    def get_pred_params(self):
        """Get model's pred_params for creating the XTransformer.PredParams instance"""
        ret = self.PredParams()

        ret.matcher_params_chain = []
        for m in self.model_list:
            if isinstance(m, TransformerMatcher):
                ret.matcher_params_chain.append(m.get_pred_params())
            elif isinstance(m, XLinearModel):
                ret.ranker_params = m.get_pred_params()
            else:
                raise TypeError("Unsupported model type: {type(m)}")
        return ret

    @classmethod
    def train(
        cls,
        prob,
        clustering,
        val_prob=None,
        train_params=None,
        pred_params=None,
        **kwargs,
    ):
        """Train the X-transformer model with the given input data.

        Args:
            prob (MLProblemWithText): ML problem to solve.
            clustering (ClusterChain): cluster-chain for the ranker, matcher will be applied on its ranker_level clustering.
                        i.e. matcher.nr_match_labels = clustering[-ranker_level].shape[1]
            val_prob (MLProblemWithText, optional): ML problem for validation.
            train_params (XTransformer.TrainParams): training parameters for XTransformer
            pred_params (XTransformer.pred_params): pred parameters for XTransformer
            kwargs:
                matmul_threads (int, optional): number of threads to use for
                    constructing label tree. Default to use at most 32 threads
                beam_size (int, optional): overrides only_topk for models except
                    bottom layer one

        Returns:
            XTransformer
        """
        # assert cluster chain in clustering is valid
        clustering = ClusterChain(clustering)
        if clustering[-1].shape[0] != prob.nr_labels:
            raise ValueError("nr_labels mismatch!")
        nr_levels = len(clustering)

        if nr_levels <= train_params.ranker_level:
            raise ValueError(f"Expect ranker_level < depth, got {train_params.ranker_level}")

        nr_transformers = nr_levels - train_params.ranker_level
        nr_linears = train_params.ranker_level

        # construct train_params
        train_params = cls.TrainParams.from_dict(train_params)
        train_params.ranker_params.mode = "ranker"

        train_params = HierarchicalMLModel._duplicate_fields_with_name_ending_with_chain(
            train_params, cls.TrainParams, nr_transformers
        )
        if nr_linears > 0:
            train_params.ranker_params.hlm_args = (
                HierarchicalMLModel._duplicate_fields_with_name_ending_with_chain(
                    train_params.ranker_params.hlm_args,
                    HierarchicalMLModel.TrainParams,
                    nr_linears,
                )
            )

        # construct pred_params
        pred_params = cls.PredParams.from_dict(pred_params)
        pred_params = HierarchicalMLModel._duplicate_fields_with_name_ending_with_chain(
            pred_params, cls.PredParams, nr_transformers
        )
        if nr_linears > 0:
            pred_params.ranker_params.hlm_args = (
                HierarchicalMLModel._duplicate_fields_with_name_ending_with_chain(
                    pred_params.ranker_params.hlm_args,
                    HierarchicalMLModel.PredParams,
                    nr_linears,
                )
            )
        pred_params = pred_params.override_with_kwargs(
            kwargs,
            no_ranker=(nr_linears == 0),
        )

        LOGGER.debug(
            f"XTransformer train_params: {json.dumps(train_params.to_dict(), indent=True)}"
        )
        LOGGER.debug(f"XTransformer pred_params: {json.dumps(pred_params.to_dict(), indent=True)}")

        def get_negative_samples(mat_true, mat_pred, scheme):
            if scheme == "tfn":
                result = smat_util.binarized(mat_true)
            elif scheme == "man":
                result = smat_util.binarized(mat_pred)
            elif "tfn" in scheme and "man" in scheme:
                result = smat_util.binarized(mat_true) + smat_util.binarized(mat_pred)
            else:
                raise ValueError("Unrecognized negative sampling method {}".format(scheme))
            LOGGER.debug(
                f"Construct {scheme} with shape={result.shape} avr_M_nnz={result.nnz/result.shape[0]}"
            )
            return result

        # construct label chain for training and validation set
        # avoid large matmul_threads to prevent overhead in Y.dot(C) and save memory
        matmul_threads = kwargs.get("threads", os.cpu_count())
        matmul_threads = min(32, matmul_threads)
        YC_list = [prob.Y]
        for cur_C in reversed(clustering[1:]):
            Y_t = clib.sparse_matmul(YC_list[-1], cur_C, threads=matmul_threads).tocsr()
            YC_list.append(Y_t)
        YC_list.reverse()

        if val_prob is not None:
            val_YC_list = [val_prob.Y]
            for cur_C in reversed(clustering[1:]):
                Y_t = clib.sparse_matmul(val_YC_list[-1], cur_C, threads=matmul_threads).tocsr()
                val_YC_list.append(Y_t)
            val_YC_list.reverse()

        model_list = []
        M, val_M = None, None
        M_pred, val_M_pred = None, None
        bootstrapping, inst_embeddings = None, None
        for i in range(nr_transformers):
            cur_train_params = train_params.matcher_params_chain[i]
            cur_pred_params = pred_params.matcher_params_chain[i]
            cur_train_params.model_dir = os.path.join(train_params.model_dir, "{}.model".format(i))

            cur_ns = cur_train_params.negative_sampling
            if i > 0:
                M = get_negative_samples(YC_list[i - 1], M_pred, cur_ns)

            cur_prob = MLProblemWithText(
                prob.X_text,
                prob.X,
                YC_list[i],
                C=clustering[i],
                M=M,
            )
            if val_prob is not None:
                if i > 0:
                    val_M = get_negative_samples(val_YC_list[i - 1], val_M_pred, cur_ns)
                cur_val_prob = MLProblemWithText(
                    val_prob.X_text,
                    val_prob.X,
                    val_YC_list[i],
                    C=clustering[i],
                    M=val_M,
                )
            else:
                cur_val_prob = None

            avr_trn_labels = (
                float(cur_prob.M.nnz) / YC_list[i].shape[0]
                if cur_prob.M is not None
                else YC_list[i].shape[1]
            )
            LOGGER.info(
                "Training Hierarchical-XTransformer with {} at level {}, nr_labels={}, avr_M_nnz={}".format(
                    cur_ns, i, YC_list[i].shape[1], avr_trn_labels
                )
            )

            # bootstrapping with previous text_encoder and instance embeddings
            if len(model_list) > 0:
                init_encoder = deepcopy(model_list[-1].text_encoder)
                bootstrapping = (init_encoder, inst_embeddings)

            res_dict = TransformerMatcher.train(
                cur_prob,
                csr_codes=M_pred,
                val_prob=cur_val_prob,
                val_csr_codes=val_M_pred,
                train_params=cur_train_params,
                pred_params=cur_pred_params,
                bootstrapping=bootstrapping,
                return_dict=True,
            )
            cur_model = res_dict["matcher"]
            M_pred = res_dict["trn_pred"]
            val_M_pred = res_dict["val_pred"]
            inst_embeddings = res_dict["trn_embeddings"]

            model_list.append(cur_model)

        # Train the subsequent layers with XLinearModel
        if train_params.ranker_level > 0:
            inst_embeddings = sk_normalize(inst_embeddings, axis=1, copy=False)
            if isinstance(prob.X, smat.csr_matrix):
                inst_embeddings = smat_util.dense_to_csr(inst_embeddings)
                prob.X = smat_util.hstack_csr([prob.X, inst_embeddings], dtype=np.float32)
            else:
                prob.X = np.hstack([prob.X, inst_embeddings])
            del inst_embeddings
            LOGGER.info("Constructed instance feature matrix with shape={}".format(prob.X.shape))

            # train the ranker
            LOGGER.info("Start training ranker...")

            # getting the top model negative sampling scheme
            # and add user supplied negatives to all subsequent ranker layers
            cur_ns = train_params.ranker_params.hlm_args.neg_mining_chain
            if isinstance(cur_ns, list):
                cur_ns = cur_ns[0]
                train_params.ranker_params.hlm_args.neg_mining_chain = [
                    v + "+usn" for v in train_params.ranker_params.hlm_args.neg_mining_chain
                ]
            else:
                train_params.ranker_params.hlm_args.neg_mining_chain += "+usn"

            M = get_negative_samples(YC_list[-train_params.ranker_level - 1], M_pred, cur_ns)

            ranker = XLinearModel.train(
                prob.X,
                prob.Y,
                C=clustering,
                user_supplied_negatives={train_params.ranker_level: M},
                train_params=train_params.ranker_params,
                pred_params=pred_params.ranker_params,
            )
            model_list.append(ranker)

        return cls(model_list)

    def predict(
        self,
        X_text,
        X_feat=None,
        pred_params=None,
        **kwargs,
    ):
        """Use the X-Transformer model to predict on given data.

        Args:
            X_text (iterable over str): instance text input to predict on
            X_feat (csr_matrix or ndarray): instance feature matrix (nr_insts, feature_dim)
            pred_kwargs (XTransformer.PredParams, optional): instance of
                XTransformer.PredParams. Default None to use pred_params stored
                during model training.
            kwargs:
                beam_size (int, optional): override the beam size specified in the model.
                    Default None to disable overriding
                only_topk (int, optional): override the only topk specified in the model
                    Default None to disable overriding
                post_processor (str, optional):  override the post_processor specified in the model
                    Default None to disable overriding
                saved_pt (str, optional): if given, will try to load encoded tensors and skip text encoding
                embeddings_save_path (str, optional): if given, will save the instance embeddings matrix
                batch_size (int, optional): per device batch size for transformer evaluation. Default 8
                batch_gen_workers (int, optional): number of CPUs to use for batch generation. Default 4
                disable_gpu (bool, optional): not use GPU even if available. Default False
                threads (int, optional): the number of threads to use for linear model prediction.

        Returns:
            P_matrix (csr_matrix): instance to label prediction (csr_matrix, nr_insts * nr_labels)
        """
        saved_pt = kwargs.get("saved_pt", None)
        batch_size = kwargs.get("batch_size", 8)
        batch_gen_workers = kwargs.get("batch_gen_workers", 4)
        disable_gpu = kwargs.get("disable_gpu", False)
        embeddings_save_path = kwargs.get("embeddings_save_path", None)
        device, n_gpu = torch_util.setup_device(not disable_gpu)

        # get the override pred_params
        nr_transformers = sum([isinstance(m, TransformerMatcher) for m in self.model_list])
        if pred_params is None:
            # copy stored params
            pred_params = self.get_pred_params()
        else:
            pred_params = self.PredParams.from_dict(pred_params)
            pred_params = HierarchicalMLModel._duplicate_fields_with_name_ending_with_chain(
                pred_params,
                self.PredParams,
                nr_transformers,
            )
        pred_params.override_with_kwargs(kwargs, no_ranker=(nr_transformers == self.depth))
        LOGGER.debug(f"XTransformer prediction with pred_params: {pred_params.to_dict()}")

        # generate instance-to-cluster prediction
        if saved_pt and os.path.isfile(saved_pt):
            text_tensors = torch.load(saved_pt)
            LOGGER.info("Predict tensors loaded_from {}".format(saved_pt))
        else:
            text_tensors = self.model_list[0].text_to_tensor(
                X_text,
                num_workers=batch_gen_workers,
                max_length=self.model_list[0].pred_params.truncate_length,
            )

        pred_csr = None
        for d, cur_model in enumerate(self.model_list):
            if isinstance(cur_model, TransformerMatcher):
                cur_model.to_device(device, n_gpu=n_gpu)
                pred_csr, embeddings = cur_model.predict(
                    text_tensors,
                    X_feat=X_feat,
                    csr_codes=pred_csr,
                    pred_params=pred_params.matcher_params_chain[d],
                    batch_size=batch_size * max(1, n_gpu),
                    batch_gen_workers=batch_gen_workers,
                )
                cur_model.to_device(torch.device("cpu"))
                torch.cuda.empty_cache()

            elif isinstance(cur_model, XLinearModel):
                # concatenate instance feature matrix with embeddings
                cat_embeddings = sk_normalize(embeddings, axis=1, copy=True)
                if isinstance(X_feat, smat.csr_matrix):
                    cat_embeddings = smat_util.dense_to_csr(cat_embeddings)
                    cat_embeddings = smat_util.hstack_csr(
                        [X_feat, cat_embeddings], dtype=np.float32
                    )
                else:
                    cat_embeddings = np.hstack([X_feat, cat_embeddings])
                LOGGER.info(
                    "Constructed instance feature matrix with shape={}".format(X_feat.shape)
                )
                pred_csr = cur_model.predict(
                    cat_embeddings,
                    csr_codes=pred_csr,
                    pred_params=None if pred_params is None else pred_params.ranker_params,
                    threads=kwargs.get("threads", -1),
                )
            else:
                raise ValueError("Unknown model type {}".format(type(cur_model)))

        if embeddings_save_path:
            smat_util.save_matrix(embeddings_save_path, embeddings)
            LOGGER.info("Saved embeddings({}) to {}".format(embeddings.shape, embeddings_save_path))

        return pred_csr
