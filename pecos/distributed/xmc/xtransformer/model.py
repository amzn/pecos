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
import gc
import tempfile

import numpy as np
import scipy.sparse as smat
import dataclasses as dc
import pecos
from pecos.core import clib
from pecos.utils.cluster_util import ClusterChain
from pecos.xmc.base import HierarchicalMLModel
from pecos.xmc.xtransformer.model import XTransformer

from pecos.xmc.xtransformer.matcher import TransformerMatcher
from pecos.xmc.xtransformer.module import MLProblemWithText
from pecos.xmc.xtransformer.network import TransformerLinearXMCHead
from . import dist_trainer
from .module import DeepSpeedUtils as ds_utils


LOGGER = logging.getLogger(__name__)


class XTransformerDistTrainer(XTransformer):
    """Distributed Trainer for Hierarchical-XTransformer for XMC."""

    @dc.dataclass
    class DistParams(pecos.BaseParams):
        """Distributed Parameters of XTransformer.

        fp16 (bool): whether to use half-precision in model fine-tuning
        max_shard_size (int): max number of instances within a data shard. Default 10 ** 7
        shard_scheme (str): data shard access scheme. [synchronized, ordered].
            Default synchronized that every worker loads the same shard
        hostfile (str): path to the hostfile for distributed training.
        shared_workdir (str): a shared workdir by all workers to cache and read data/params.
        """

        fp16: bool = False
        max_shard_size: int = 10**7
        shard_scheme: str = "synchronized"
        hostfile: str = ""
        shared_workdir: str = "."

    @staticmethod
    def get_label_hierarchy(Y, clustering):
        """Get label hierarchy for multi-resolution training

        Args:
            Y (csr_matrix): the label matrix with shape = (nr_inst, nr_labels)
            clustering (ClusterChain): Hierarchical label tree with depth D

        Returns:
            YC_list (list): list of length D. YC_list[d] is the label matrix at level d, shape=(nr_inst, K^(d))
        """

        YC_list = [Y]
        for cur_C in reversed(clustering[1:]):
            Y_t = clib.sparse_matmul(YC_list[-1], cur_C, threads=min(32, os.cpu_count())).tocsr()
            YC_list.append(Y_t)
        YC_list.reverse()
        return YC_list

    @staticmethod
    def get_pretrained(model_shortcut, num_labels=None, hidden_dropout_prob=0.1):
        """Get pretrained transformer model for fine-tuning

        Args:
            model_shortcut (str, TransformerMatcher):
                if TransformerMatcher, do nothing
                if local path to serialized TransformerMatcher, load from disk
                else if model identifier, download pre-trained model frim huggingface repo.
            num_labels (int): the number of label embeddings the model is expected to have.
                If existing num_labels is inconsistent with given value, new label embeddings will be randomly initialized.
            hidden_dropout_prob (float): training dropout probabilities for the hidden vectors.
                Only used when downloading the model from huggingface repo.

        Returns:
            TransformerMatcher with num_labels label embeddings

        """
        if isinstance(model_shortcut, TransformerMatcher):
            parent_model = model_shortcut
        elif os.path.exists(model_shortcut):
            # load from local
            parent_model = TransformerMatcher.load(model_shortcut)
            LOGGER.info("Loaded model from {}.".format(model_shortcut))
        else:
            # download from huggingface repo
            parent_model = TransformerMatcher.download_model(
                model_shortcut,
                num_labels,
                hidden_dropout_prob=hidden_dropout_prob,
            )
            LOGGER.info("Downloaded {} model from s3.".format(model_shortcut))

        if num_labels is not None and num_labels != parent_model.nr_labels:
            LOGGER.warning(
                f"Got mismatch nr_labels (expected {num_labels} but got {parent_model.nr_labels}), text_model reinitialized!"
            )
            parent_model.text_model = TransformerLinearXMCHead(
                parent_model.text_encoder.config.hidden_size, num_labels
            )
            parent_model.text_encoder.config.num_labels = num_labels
        return parent_model

    @classmethod
    def train(
        cls,
        prob,
        clustering=None,
        train_params=None,
        pred_params=None,
        dist_params=None,
        **kwargs,
    ):
        """Train the XR-Transformer model with the given input data.

        Args:
            prob (MLProblemWithText): ML problem to solve.
            clustering (ClusterChain, optional): preliminary hierarchical label tree,
                where transformer is fine-tuned on.
            train_params (XTransformer.TrainParams): training parameters for XTransformer
            pred_params (XTransformer.PredParams): pred parameters for XTransformer
            dist_params (XTransformerDistTrainer.DistParams): distributed parameters
                for XTransformerDistTrainer
            kwargs:
                beam_size (int, optional): overrides only_topk for models except
                    bottom layer one

        Returns:
            XTransformer
        """

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
        # construct dist_params
        if dist_params is None:
            # fill all BaseParams with their default value
            dist_params = cls.DistParams.from_dict(dict(), recursive=True)
        else:
            dist_params = cls.DistParams.from_dict(dist_params)

        if not train_params.only_encoder:
            LOGGER.warning(
                f"Distributed fine-tuning is only for encoder, fall back to only_encoder=True"
            )
            train_params.only_encoder = True

        if not train_params.do_fine_tune:
            LOGGER.warning(f"do_fine_tune is set to False, override to do_fine_tune=True..")
            train_params.do_fine_tune = True

        # 1. Constructing primary Hierarchial Label Tree
        if clustering is None:
            clustering = smat.csc_matrix(np.ones((prob.nr_labels, 1)), dtype=np.float32)

        clustering = ClusterChain(clustering)
        if clustering[-1].shape[0] != prob.nr_labels:
            raise ValueError("nr_labels mismatch!")

        prelim_hierarchiy = [cc.shape[0] for cc in clustering]
        LOGGER.info("Hierarchical label tree: {}".format(prelim_hierarchiy))

        # 1.1 Get the fine-tuning task numbers
        nr_transformers = sum(i <= train_params.max_match_clusters for i in prelim_hierarchiy)

        LOGGER.info(
            "Fine-tune Transformers with nr_labels={}".format(
                [cc.shape[0] for cc in clustering[:nr_transformers]]
            )
        )

        # 1.2 construct fields with chain now we know the depth
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

        LOGGER.debug(f"XTransformer pred_params: {json.dumps(pred_params.to_dict(), indent=True)}")
        LOGGER.debug(
            f"XTransformerDistTrainer dist_params: {json.dumps(dist_params.to_dict(), indent=True)}"
        )
        # construct label chain for training and validation set
        YC_list = cls.get_label_hierarchy(prob.Y, clustering)

        parent_model = train_params.matcher_params_chain[0].model_shortcut
        if train_params.matcher_params_chain[0].init_model_dir:
            parent_model = train_params.matcher_params_chain[0].init_model_dir

        prev_head = None
        for i in range(nr_transformers):
            cur_train_params = train_params.matcher_params_chain[i]
            cur_pred_params = pred_params.matcher_params_chain[i]

            # only support TFN
            M = YC_list[i - 1] if i > 0 else None

            cur_prob = MLProblemWithText(
                prob.X_text,
                YC_list[i],
                X_feat=None,
                C=clustering[i],
                M=M,
            )

            avr_trn_labels = (
                float(cur_prob.M.nnz) / YC_list[i].shape[0]
                if cur_prob.M is not None
                else YC_list[i].shape[1]
            )
            LOGGER.info(
                "Fine-tuning XR-Transformer with {} at level {}, nr_labels={}, avr_M_nnz={}".format(
                    "tfn", i, YC_list[i].shape[1], avr_trn_labels
                )
            )

            # construct TransformerMatcher instance for this layer
            parent_model = cls.get_pretrained(
                parent_model,
                num_labels=cur_prob.Y.shape[1],
                hidden_dropout_prob=train_params.matcher_params_chain[0].hidden_dropout_prob,
            )
            parent_model.C = cur_prob.C
            parent_model.train_params = cur_train_params
            parent_model.pred_params = cur_pred_params

            if cur_train_params.bootstrap_method == "no-bootstrap" or i == 0:
                parent_model.text_model.random_init(sparse=False)
                LOGGER.info("Randomly initialized transformer text_model!")
            else:
                if cur_train_params.bootstrap_method != "inherit":
                    LOGGER.warning(
                        f"bootstrap_method={cur_train_params.bootstrap_method} not supported in distributed training. Fall back to inherit"
                    )
                parent_model.text_model.inherit(prev_head, cur_prob.C, sparse=False)
                LOGGER.info("Initialized transformer text_model from parent layer!")

            if cur_train_params.pre_tokenize:
                if not prob.is_tokenized:
                    prob.X_text = parent_model.text_to_tensor(
                        prob.X_text,
                        max_length=cur_pred_params.truncate_length,
                    )

            # temp folder in workdir
            temp_dir = tempfile.TemporaryDirectory(dir=dist_params.shared_workdir)
            temp_data_dir = temp_dir.name
            temp_data_path = f"{temp_data_dir}/train_data"
            temp_params_path = f"{temp_data_dir}/param.json"

            if cur_train_params.checkpoint_dir:
                temp_encoder_path = cur_train_params.checkpoint_dir
            else:
                temp_encoder_path = f"{temp_data_path}/encoder"

            # construct dataset and save into shards
            train_data = TransformerMatcher.prepare_data(
                cur_prob,
                label_padding_idx=parent_model.text_model.label_padding_idx,
                pre_tensorize_labels=cur_train_params.pre_tensorize_labels,
                input_transform=None if cur_prob.is_tokenized else parent_model._tokenize,
            )
            num_shards = (
                len(train_data) + dist_params.max_shard_size - 1
            ) // dist_params.max_shard_size
            train_data.save(temp_data_path, num_shards=num_shards)
            del train_data
            gc.collect()
            LOGGER.info(f"Cached train_data to {temp_data_path} with {num_shards}")

            temp_params = {
                "train_params": cur_train_params.to_dict(),
                "pred_params": cur_pred_params.to_dict(),
                "dist_params": dist_params.to_dict(),
            }
            with open(temp_params_path, "w") as fout:
                fout.write(json.dumps(temp_params))
            LOGGER.info(f"Cached params: {json.dumps(temp_params)}")

            parent_model.save(temp_encoder_path)

            # start distributed training
            ds_utils.cli_launcher(
                dist_trainer.__name__,
                hostfile=dist_params.hostfile,
                module_args={
                    "data_path": temp_data_path,
                    "model_path": temp_encoder_path,
                    "output_path": temp_encoder_path,
                    "params_path": temp_params_path,
                    "fp16": 1 if dist_params.fp16 else 0,
                    "shard_scheme": dist_params.shard_scheme,
                },
            )

            LOGGER.info("Reload the best checkpoint from {}".format(temp_encoder_path))
            parent_model = TransformerMatcher.load(temp_encoder_path)
            parent_model.clear_cuda()
            prev_head = parent_model.text_model

        return cls(parent_model, None)
