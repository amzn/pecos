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
import tempfile

import dataclasses as dc
import pecos
import torch
from pecos.core import clib
from pecos.utils import smat_util, torch_util
from pecos.utils.cluster_util import ClusterChain
from pecos.xmc import Indexer, LabelEmbeddingFactory
from pecos.xmc.base import HierarchicalMLModel, HierarchicalKMeans
from pecos.xmc.xlinear.model import XLinearModel

from .matcher import TransformerMatcher
from .module import MLProblemWithText

LOGGER = logging.getLogger(__name__)


class XTransformer(pecos.BaseClass):
    """Hierarchical-XTransformer for XMC.
    Consists of one or more TransformerMatcher models followed by an optional XLinearModel.

    See more details in Section 5 of PECOS paper (Yu et al., 2020) and XR-Transformer paper (Zhang et al., 2021).

    PECOS: Prediction for Enormous and Correlated Output Spaces
        Hsiang-Fu Yu, Kai Zhong, Inderjit S. Dhillon
        https://arxiv.org/abs/2010.05878

    Fast Multi-Resolution Transformer Fine-tuning for Extreme Multi-label Text Classification
        Jiong Zhang, Wei-Cheng Chang, Hsiang-Fu Yu, Inderjit Dhillon
        https://arxiv.org/pdf/2110.00685
    """

    @dc.dataclass
    class TrainParams(pecos.BaseParams):
        """Training Parameters of XTransformer.

        preliminary_indexer_params (HierarchicalKMeans.TrainParams): params to generate preliminary hierarchial label tree.
            ignored if clustering is given
        refined_indexer_params (HierarchicalKMeans.TrainParams): params to generate refined hierarchial label tree.
            ignored if fix_clustering is True
        matcher_params_chain (TransformerMatcher.TrainParams or list): chain of params for TransformerMatchers.
        ranker_params (XLinearModel.TrainParams): train params for linear ranker

        do_fine_tune (bool, optional): if False, skip fine-tuning steps and directly use pre-trained transformer models.
            Default True
        only_encoder (bool, optional): if True, skip linear ranker training. Default False
        fix_clustering (bool, optional): if True, use the same hierarchial label tree for fine-tuning and final prediction. Default false.
        max_match_clusters (int, optional): max number of clusters on which to fine-tune transformer. Default 32768
        save_emb_dir (str): dir to save instance embeddings. Default None to ignore
        """

        preliminary_indexer_params: HierarchicalKMeans.TrainParams = None  # type: ignore
        refined_indexer_params: HierarchicalKMeans.TrainParams = None  # type: ignore
        matcher_params_chain: TransformerMatcher.TrainParams = None  # type: ignore
        ranker_params: XLinearModel.TrainParams = None  # type: ignore

        do_fine_tune: bool = True
        only_encoder: bool = False
        fix_clustering: bool = False
        max_match_clusters: int = 32768
        save_emb_dir: str = None  # type: ignore

    @dc.dataclass
    class PredParams(pecos.BaseParams):
        """Pred Parameters of XTransformer.

        matcher_params_chain (TransformerMatcher.PredParams or list): chain of params for TransformerMatchers
        ranker_params (XLinearModel.PredParams): pred params for linear ranker
        """

        matcher_params_chain: TransformerMatcher.PredParams = None  # type: ignore
        ranker_params: XLinearModel.PredParams = None  # type: ignore

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
            if pred_kwargs is not None:
                if not isinstance(pred_kwargs, dict):
                    raise TypeError("type(pred_kwargs) must be dict")
                overridden_beam_size = pred_kwargs.get("beam_size", None)
                overridden_only_topk = pred_kwargs.get("only_topk", None)
                overridden_post_processor = pred_kwargs.get("post_processor", None)
                depth = len(self.matcher_params_chain)
                for d in range(depth):
                    if overridden_beam_size:
                        if d == depth - 1:
                            continue
                        self.matcher_params_chain[d].only_topk = overridden_beam_size
                    if overridden_post_processor:
                        self.matcher_params_chain[d].post_processor = overridden_post_processor

                if overridden_only_topk:
                    self.matcher_params_chain[-1].only_topk = overridden_only_topk

                if self.ranker_params is not None:
                    self.ranker_params.override_with_kwargs(pred_kwargs)
            return self

    def __init__(self, text_encoder, concat_model):
        """Initialization

        Args:
            text_encoder (TransformerMatcher): Transformer model to encode input text
            concat_model (XLinearModel): linear models predicting on concatenated features
        """
        self.text_encoder = text_encoder
        self.concat_model = concat_model

    @property
    def depth(self):
        """Get the depth of the hierarchical model

        Returns:
            depth (int): Depth of model.
        """
        if self.concat_model:
            return self.concat_model.model.depth
        else:
            return 1

    @property
    def nr_labels(self):
        """Get the number of labels

        Returns:
            nr_labels (int): Number of labels.
        """
        if self.concat_model:
            return self.concat_model.nr_labels
        else:
            return self.text_encoder.nr_labels

    def save(self, save_dir):
        """Save the X-Transformer model to file.

        Args:
            save_dir (str): dir to save the model, will be created if not exist
                save params to save_dir/param.json
                save text_encoder to save_dir/text_encoder
                save concat_model to save_dir/concat_model if concat_model exist
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

        self.text_encoder.save(os.path.join(save_dir, "text_encoder"))
        if self.concat_model:
            self.concat_model.save(os.path.join(save_dir, "concat_model"))
        LOGGER.info("Model saved to {}".format(save_dir))

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
        text_encoder = TransformerMatcher.load(os.path.join(load_dir, "text_encoder"))
        try:
            concat_model = XLinearModel.load(os.path.join(load_dir, "concat_model"))
            LOGGER.info("Full model loaded from {}".format(load_dir))
        except FileNotFoundError:
            concat_model = None
            LOGGER.info("Concat model not exist, text encoder loaded from {}".format(load_dir))
        return cls(text_encoder, concat_model)

    def get_pred_params(self):
        """Get model's pred_params for creating the XTransformer.PredParams instance"""
        ret = self.PredParams()
        ret.matcher_params_chain = [self.text_encoder.get_pred_params()]
        if self.concat_model is not None:
            ret.ranker_params = self.concat_model.get_pred_params()
        else:
            ret.ranker_params = None
        return ret

    @classmethod
    def train(
        cls,
        prob,
        clustering=None,
        val_prob=None,
        train_params=None,
        pred_params=None,
        **kwargs,
    ):
        """Train the XR-Transformer model with the given input data.

        Args:
            prob (MLProblemWithText): ML problem to solve.
            clustering (ClusterChain, optional): preliminary hierarchical label tree,
                where transformer is fine-tuned on.
            val_prob (MLProblemWithText, optional): ML problem for validation.
            train_params (XTransformer.TrainParams): training parameters for XTransformer
            pred_params (XTransformer.pred_params): pred parameters for XTransformer
            kwargs:
                label_feat (ndarray or csr_matrix, optional): label features on which to generate preliminary HLT
                saved_trn_pt (str, optional): path to save the tokenized trn text. Use a tempdir if not given
                saved_val_pt (str, optional): path to save the tokenized val text. Use a tempdir if not given
                matmul_threads (int, optional): number of threads to use for
                    constructing label tree. Default to use at most 32 threads
                beam_size (int, optional): overrides only_topk for models except
                    bottom layer one

        Returns:
            XTransformer
        """
        # tempdir to save tokenized text
        temp_dir = tempfile.TemporaryDirectory()
        saved_trn_pt = kwargs.get("saved_trn_pt", "")
        if not saved_trn_pt:
            saved_trn_pt = f"{temp_dir.name}/X_trn.pt"

        saved_val_pt = kwargs.get("saved_val_pt", "")
        if not saved_val_pt:
            saved_val_pt = f"{temp_dir.name}/X_val.pt"

        # construct train_params
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

        if not train_params.do_fine_tune:
            if isinstance(train_params.matcher_params_chain, list):
                matcher_train_params = train_params.matcher_params_chain[-1]
            else:
                matcher_train_params = train_params.matcher_params_chain

            if isinstance(train_params.matcher_params_chain, list):
                matcher_pred_params = pred_params.matcher_params_chain[-1]
            else:
                matcher_pred_params = pred_params.matcher_params_chain

            device, n_gpu = torch_util.setup_device(matcher_train_params.use_gpu)

            if matcher_train_params.init_model_dir:
                parent_model = cls.load(train_params.init_model_dir)
                LOGGER.info("Loaded encoder from {}.".format(matcher_train_params.init_model_dir))
            else:
                parent_model = TransformerMatcher.download_model(
                    matcher_train_params.model_shortcut,
                )
                LOGGER.info(
                    "Downloaded encoder from {}.".format(matcher_train_params.model_shortcut)
                )

            parent_model.to_device(device, n_gpu=n_gpu)
            _, inst_embeddings = parent_model.predict(
                prob.X_text,
                pred_params=matcher_pred_params,
                batch_size=matcher_train_params.batch_size * max(1, n_gpu),
                batch_gen_workers=matcher_train_params.batch_gen_workers,
                only_embeddings=True,
            )
            if val_prob:
                _, val_inst_embeddings = parent_model.predict(
                    val_prob.X_text,
                    pred_params=matcher_pred_params,
                    batch_size=matcher_train_params.batch_size * max(1, n_gpu),
                    batch_gen_workers=matcher_train_params.batch_gen_workers,
                    only_embeddings=True,
                )
        else:
            # 1. Constructing primary Hierarchial Label Tree
            if clustering is None:
                label_feat = kwargs.get("label_feat", None)
                if label_feat is None:
                    if prob.X_feat is None:
                        raise ValueError(
                            "Instance features are required to generate label features!"
                        )
                    label_feat = LabelEmbeddingFactory.pifa(prob.Y, prob.X_feat)

                clustering = Indexer.gen(
                    label_feat,
                    train_params=train_params.preliminary_indexer_params,
                )
            else:
                # assert cluster chain in clustering is valid
                clustering = ClusterChain(clustering)
                if clustering[-1].shape[0] != prob.nr_labels:
                    raise ValueError("nr_labels mismatch!")
            prelim_hierarchiy = [cc.shape[0] for cc in clustering]
            LOGGER.info("Hierarchical label tree: {}".format(prelim_hierarchiy))

            # get the fine-tuning task numbers
            nr_transformers = sum(i <= train_params.max_match_clusters for i in prelim_hierarchiy)

            LOGGER.info(
                "Fine-tune Transformers with nr_labels={}".format(
                    [cc.shape[0] for cc in clustering[:nr_transformers]]
                )
            )

            steps_scale = kwargs.get("steps_scale", None)
            if steps_scale is None:
                steps_scale = [1.0] * nr_transformers
            if len(steps_scale) != nr_transformers:
                raise ValueError(f"steps-scale length error: {len(steps_scale)}!={nr_transformers}")

            # construct fields with chain now we know the depth
            train_params = HierarchicalMLModel._duplicate_fields_with_name_ending_with_chain(
                train_params, cls.TrainParams, nr_transformers
            )

            LOGGER.debug(
                f"XTransformer train_params: {json.dumps(train_params.to_dict(), indent=True)}"
            )

            pred_params = HierarchicalMLModel._duplicate_fields_with_name_ending_with_chain(
                pred_params, cls.PredParams, nr_transformers
            )
            pred_params = pred_params.override_with_kwargs(kwargs)

            LOGGER.debug(
                f"XTransformer pred_params: {json.dumps(pred_params.to_dict(), indent=True)}"
            )

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

            parent_model = None
            M, val_M = None, None
            M_pred, val_M_pred = None, None
            bootstrapping, inst_embeddings = None, None
            for i in range(nr_transformers):
                cur_train_params = train_params.matcher_params_chain[i]
                cur_pred_params = pred_params.matcher_params_chain[i]
                cur_train_params.max_steps = steps_scale[i] * cur_train_params.max_steps
                cur_train_params.num_train_epochs = (
                    steps_scale[i] * cur_train_params.num_train_epochs
                )

                cur_ns = cur_train_params.negative_sampling

                # construct train and val problem for level i
                # note that final layer do not need X_feat
                if i > 0:
                    M = get_negative_samples(YC_list[i - 1], M_pred, cur_ns)

                cur_prob = MLProblemWithText(
                    prob.X_text,
                    YC_list[i],
                    X_feat=None if i == nr_transformers - 1 else prob.X_feat,
                    C=clustering[i],
                    M=M,
                )
                if val_prob is not None:
                    if i > 0:
                        val_M = get_negative_samples(val_YC_list[i - 1], val_M_pred, cur_ns)
                    cur_val_prob = MLProblemWithText(
                        val_prob.X_text,
                        val_YC_list[i],
                        X_feat=None if i == nr_transformers - 1 else val_prob.X_feat,
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
                    "Fine-tuning XR-Transformer with {} at level {}, nr_labels={}, avr_M_nnz={}".format(
                        cur_ns, i, YC_list[i].shape[1], avr_trn_labels
                    )
                )

                # bootstrapping with previous text_encoder and instance embeddings
                if parent_model is not None:
                    init_encoder = deepcopy(parent_model.text_encoder)
                    init_text_model = deepcopy(parent_model.text_model)
                    bootstrapping = (init_encoder, inst_embeddings, init_text_model)

                # determine whether train prediction and instance embeddings are needed
                return_train_pred = (
                    i + 1 < nr_transformers
                ) and "man" in train_params.matcher_params_chain[i + 1].negative_sampling
                return_train_embeddings = (
                    i + 1 == nr_transformers
                ) or "linear" in cur_train_params.bootstrap_method

                res_dict = TransformerMatcher.train(
                    cur_prob,
                    csr_codes=M_pred,
                    val_prob=cur_val_prob,
                    val_csr_codes=val_M_pred,
                    train_params=cur_train_params,
                    pred_params=cur_pred_params,
                    bootstrapping=bootstrapping,
                    return_dict=True,
                    return_train_pred=return_train_pred,
                    return_train_embeddings=return_train_embeddings,
                    saved_trn_pt=saved_trn_pt,
                    saved_val_pt=saved_val_pt,
                )
                parent_model = res_dict["matcher"]
                M_pred = res_dict["trn_pred"]
                val_M_pred = res_dict["val_pred"]
                inst_embeddings = res_dict["trn_embeddings"]
                val_inst_embeddings = res_dict["val_embeddings"]

        if train_params.save_emb_dir:
            os.makedirs(train_params.save_emb_dir, exist_ok=True)
            if inst_embeddings is not None:
                smat_util.save_matrix(
                    os.path.join(train_params.save_emb_dir, "X.trn.npy"),
                    inst_embeddings,
                )
                LOGGER.info(f"Trn embeddings saved to {train_params.save_emb_dir}/X.trn.npy")
            if val_inst_embeddings is not None:
                smat_util.save_matrix(
                    os.path.join(train_params.save_emb_dir, "X.val.npy"),
                    val_inst_embeddings,
                )
                LOGGER.info(f"Val embeddings saved to {train_params.save_emb_dir}/X.val.npy")

        ranker = None
        if not train_params.only_encoder:
            # construct X_concat
            X_concat = TransformerMatcher.concat_features(
                prob.X_feat,
                inst_embeddings,
                normalize_emb=True,
            )
            del inst_embeddings
            LOGGER.info("Constructed instance feature matrix with shape={}".format(X_concat.shape))

            # 3. construct refined HLT
            if train_params.fix_clustering:
                clustering = clustering
            else:
                clustering = Indexer.gen(
                    LabelEmbeddingFactory.pifa(prob.Y, X_concat),
                    train_params=train_params.refined_indexer_params,
                )
            LOGGER.info(
                "Hierarchical label tree for ranker: {}".format([cc.shape[0] for cc in clustering])
            )

            # the HLT could have changed depth
            train_params.ranker_params.hlm_args = (
                HierarchicalMLModel._duplicate_fields_with_name_ending_with_chain(
                    train_params.ranker_params.hlm_args,
                    HierarchicalMLModel.TrainParams,
                    len(clustering),
                )
            )
            pred_params.ranker_params.hlm_args = (
                HierarchicalMLModel._duplicate_fields_with_name_ending_with_chain(
                    pred_params.ranker_params.hlm_args,
                    HierarchicalMLModel.PredParams,
                    len(clustering),
                )
            )
            pred_params.ranker_params.override_with_kwargs(kwargs)

            # train the ranker
            LOGGER.info("Start training ranker...")

            ranker = XLinearModel.train(
                X_concat,
                prob.Y,
                C=clustering,
                train_params=train_params.ranker_params,
                pred_params=pred_params.ranker_params,
            )

        return cls(parent_model, ranker)

    def predict(
        self,
        X_text,
        X_feat=None,
        pred_params=None,
        **kwargs,
    ):
        """Use the XR-Transformer model to predict on given data.

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
                batch_size (int, optional): per device batch size for transformer evaluation. Default 8
                batch_gen_workers (int, optional): number of CPUs to use for batch generation. Default 4
                use_gpu (bool, optional): use GPU if available. Default True
                max_pred_chunk (int, optional): max number of instances to predict at once.
                    Set to None to ignore. Default 10^7
                threads (int, optional): the number of threads to use for linear model prediction.

        Returns:
            pred_csr (csr_matrix): instance to label prediction (csr_matrix, nr_insts * nr_labels)
        """
        if not isinstance(self.concat_model, XLinearModel):
            raise TypeError("concat_model is not present in current XTransformer model!")

        saved_pt = kwargs.get("saved_pt", None)
        batch_size = kwargs.get("batch_size", 8)
        batch_gen_workers = kwargs.get("batch_gen_workers", 4)
        use_gpu = kwargs.get("use_gpu", True)
        max_pred_chunk = kwargs.get("max_pred_chunk", 10 ** 7)
        device, n_gpu = torch_util.setup_device(use_gpu)

        # get the override pred_params
        if pred_params is None:
            pred_params = self.get_pred_params()
        else:
            pred_params = self.PredParams.from_dict(pred_params)
        pred_params.override_with_kwargs(kwargs)

        LOGGER.debug(
            f"Prediction with pred_params: {json.dumps(pred_params.to_dict(), indent=True)}"
        )
        if isinstance(pred_params.matcher_params_chain, list):
            encoder_pred_params = pred_params.matcher_params_chain[-1]
        else:
            encoder_pred_params = pred_params.matcher_params_chain

        # generate instance-to-cluster prediction
        if saved_pt and os.path.isfile(saved_pt):
            text_tensors = torch.load(saved_pt)
            LOGGER.info("Text tensors loaded_from {}".format(saved_pt))
        else:
            text_tensors = self.text_encoder.text_to_tensor(
                X_text,
                num_workers=batch_gen_workers,
                max_length=encoder_pred_params.truncate_length,
            )

        pred_csr = None
        self.text_encoder.to_device(device, n_gpu=n_gpu)
        _, embeddings = self.text_encoder.predict(
            text_tensors,
            pred_params=encoder_pred_params,
            batch_size=batch_size * max(1, n_gpu),
            batch_gen_workers=batch_gen_workers,
            max_pred_chunk=max_pred_chunk,
            only_embeddings=True,
        )

        cat_embeddings = TransformerMatcher.concat_features(
            X_feat,
            embeddings,
            normalize_emb=True,
        )
        LOGGER.debug(
            "Constructed instance feature matrix with shape={}".format(cat_embeddings.shape)
        )
        pred_csr = self.concat_model.predict(
            cat_embeddings,
            pred_params=None if pred_params is None else pred_params.ranker_params,
            max_pred_chunk=max_pred_chunk,
            threads=kwargs.get("threads", -1),
        )
        return pred_csr

    def encode(
        self,
        X_text,
        pred_params=None,
        **kwargs,
    ):
        """Use the Transformer text encoder to generate embeddings for input data.

        Args:
            X_text (iterable over str): instance text input to predict on
            pred_kwargs (XTransformer.PredParams, optional): instance of
                XTransformer.PredParams. Default None to use pred_params stored
                during model training.
            kwargs:
                saved_pt (str, optional): if given, will try to load encoded tensors and skip text encoding
                batch_size (int, optional): per device batch size for transformer evaluation. Default 8
                batch_gen_workers (int, optional): number of CPUs to use for batch generation. Default 4
                use_gpu (bool, optional): use GPU if available. Default True
                max_pred_chunk (int, optional): max number of instances to predict at once.
                    Set to None to ignore. Default 10^7

        Returns:
            embeddings (ndarray): instance embedding on training data, shape = (nr_inst, hidden_dim).
        """
        saved_pt = kwargs.get("saved_pt", None)
        batch_size = kwargs.get("batch_size", 8)
        batch_gen_workers = kwargs.get("batch_gen_workers", 4)
        use_gpu = kwargs.get("use_gpu", True)
        max_pred_chunk = kwargs.get("max_pred_chunk", 10 ** 7)
        device, n_gpu = torch_util.setup_device(use_gpu)

        # get the override pred_params
        if pred_params is None:
            pred_params = self.get_pred_params()
        else:
            pred_params = self.PredParams.from_dict(pred_params)
        pred_params.override_with_kwargs(kwargs)

        LOGGER.debug(f"Encode with pred_params: {json.dumps(pred_params.to_dict(), indent=True)}")
        if isinstance(pred_params.matcher_params_chain, list):
            encoder_pred_params = pred_params.matcher_params_chain[-1]
        else:
            encoder_pred_params = pred_params.matcher_params_chain

        # generate instance-to-cluster prediction
        if saved_pt and os.path.isfile(saved_pt):
            text_tensors = torch.load(saved_pt)
            LOGGER.info("Text tensors loaded_from {}".format(saved_pt))
        else:
            text_tensors = self.text_encoder.text_to_tensor(
                X_text,
                num_workers=batch_gen_workers,
                max_length=encoder_pred_params.truncate_length,
            )

        self.text_encoder.to_device(device, n_gpu=n_gpu)
        _, embeddings = self.text_encoder.predict(
            text_tensors,
            pred_params=encoder_pred_params,
            batch_size=batch_size * max(1, n_gpu),
            batch_gen_workers=batch_gen_workers,
            max_pred_chunk=max_pred_chunk,
            only_embeddings=True,
        )
        return embeddings
