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
import copy
import json
import logging
import os
import tempfile
import time

import dataclasses as dc
import numpy as np
import pecos
import scipy.sparse as smat
import torch
import transformers
from pecos.core import clib
from pecos.utils import smat_util, torch_util
from pecos.xmc import MLModel, MLProblem, PostProcessor
from sklearn.preprocessing import normalize as sk_normalize
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, AutoConfig, get_scheduler

from .module import XMCDataset
from .network import ENCODER_CLASSES, HingeLoss, TransformerLinearXMCHead

logging.getLogger(transformers.__name__).setLevel(logging.WARNING)

LOGGER = logging.getLogger(__name__)


class TransformerMatcher(pecos.BaseClass):
    """Transformer Matcher Model

    Construct, fine-tune the transformers and predict on a fixed output label space.
    """

    LOSS_FUNCTION_TYPES = {
        "hinge": HingeLoss(margin=1.0, power=1),
        "squared-hinge": HingeLoss(margin=1.0, power=2),
        "weighted-hinge": HingeLoss(margin=1.0, power=1, cost_weighted=True),
        "weighted-squared-hinge": HingeLoss(margin=1.0, power=2, cost_weighted=True),
        "cross-entropy": torch.nn.BCEWithLogitsLoss(reduction="sum"),
    }

    @dc.dataclass
    class TrainParams(pecos.BaseParams):  # type: ignore
        """Training Parameters of MLModel

        model_shortcut (str): string of pre-trained model shortcut. Default 'bert-base-cased'
        negative_sampling (str): negative sampling types. Default tfn
        loss_function (str): type of loss function to use for transformer
            training. Default 'squared-hinge'
        bootstrap_method (str): algorithm to bootstrap text_model. If not None, initialize
            TransformerMatcher projection layer with one of:
                'linear' (default): linear model trained on final embeddings of parent layer
                'inherit': inherit weights from parent labels
        lr_schedule (str): learning rate schedule. See transformers.SchedulerType for details.
            Default 'linear'

        threshold (float): threshold to sparsify the model weights. Default 0.1
        hidden_dropout_prob (float): hidden dropout prob in deep transformer models. Default 0.1
        batch_size (int):  batch size for transformer training. Default 8
        batch_gen_workers (int): number of workers for batch generation. Default 4
        max_active_matching_labels (int): max number of active matching labels,
            will sub-sample from existing negative samples if necessary. Default None
            to ignore
        max_num_labels_in_gpu (int): Upper limit on labels to put output layer in GPU.
            Default 65536.
        max_steps (int): if > 0: set total number of training steps to perform.
            Override num-train-epochs. Default -1.
        max_no_improve_cnt (int): if > 0, training will stop when this number of
            validation steps result in no improvement. Default -1.
        num_train_epochs (int): total number of training epochs to perform. Default 5
        gradient_accumulation_steps (int): number of updates steps to accumulate
            before performing a backward/update pass. Default 1.
        weight_decay (float): weight decay rate for regularization. Default 0 to ignore
        max_grad_norm (float): max gradient norm used for gradient clipping. Default 1.0
        learning_rate (float): maximum learning rate for Adam. Default 5e-5
        adam_epsilon (float): epsilon for Adam optimizer.Default 1e-8
        warmup_steps (float): learning rate warmup over warmup-steps. Default 0
        logging_steps (int): log training information every NUM updates steps. Default 50
        save_steps (int): save checkpoint every NUM updates steps. Default 100

        cost_sensitive_ranker (bool, optional): if True, use clustering count aggregating for ranker's cost-sensitive learnin
            Default False
        use_gpu (bool, optional): whether to use GPU even if available. Default True

        checkpoint_dir (str): path to save training checkpoints. Default empty to use a temp dir.
        cache_dir (str): dir to store the pre-trained models downloaded from
            s3. Default empty to use a temp dir.
        init_model_dir (str): path to load checkpoint of TransformerMatcher. If given,
            start from the given checkpoint rather than downloading a
            pre-trained model from S3. Default empty to ignore
        """

        model_shortcut: str = "bert-base-cased"
        negative_sampling: str = "tfn"
        loss_function: str = "squared-hinge"
        bootstrap_method: str = "linear"
        lr_schedule: str = "linear"

        threshold: float = 0.1
        hidden_dropout_prob: float = 0.1
        batch_size: int = 8
        batch_gen_workers: int = 4
        max_active_matching_labels: int = None  # type: ignore
        max_num_labels_in_gpu: int = 65536
        max_steps: int = 0
        max_no_improve_cnt: int = -1
        num_train_epochs: int = 5
        gradient_accumulation_steps: int = 1
        weight_decay: float = 0
        max_grad_norm: float = 1.0
        learning_rate: float = 1e-4
        adam_epsilon: float = 1e-8
        warmup_steps: int = 0
        logging_steps: int = 50
        save_steps: int = 100

        cost_sensitive_ranker: bool = False
        use_gpu: bool = True

        checkpoint_dir: str = ""
        cache_dir: str = ""
        init_model_dir: str = ""

    @dc.dataclass
    class PredParams(pecos.BaseParams):  # type: ignore
        """Prediction Parameters of MLModel

        only_topk (int, optional): the only topk specified in the model.
            Default to 20
        post_processor (str, optional):  the post_processor specified in the model.
            Default to "noop"
        ensemble_method (str, optional): micro ensemble method to generate prediction.
            Default to "transformer-only". See TransformerMatcher.ensemble_prediction for details.
        truncate_length (int, optional): length to truncate input text, default 128.

        """

        only_topk: int = 20
        post_processor: str = "noop"
        ensemble_method: str = "transformer-only"
        truncate_length: int = 128

        def override_with_kwargs(self, pred_kwargs):
            """Override Class attributes from prediction key-word arguments.

            Args:
                pred_kwargs (dict): Args for prediction.

            Returns:
                self (PredParams): Overriden self instance.
            """
            if pred_kwargs is not None:
                if not isinstance(pred_kwargs, dict):
                    raise TypeError("type(pred_kwargs) must be dict")
                overridden_only_topk = pred_kwargs.get("only_topk", None)
                overridden_post_processor = pred_kwargs.get("post_processor", None)
                overridden_ensemble_method = pred_kwargs.get("ensemble_method", None)
                overridden_truncate_length = pred_kwargs.get("truncate_length", None)
                if overridden_only_topk:
                    self.only_topk = overridden_only_topk
                if overridden_post_processor:
                    self.post_processor = overridden_post_processor
                if overridden_ensemble_method:
                    self.ensemble_method = overridden_ensemble_method
                if overridden_truncate_length:
                    self.truncate_length = overridden_truncate_length
            return self

    def __init__(
        self,
        text_encoder,
        text_tokenizer,
        text_model,
        C=None,
        concat_model=None,
        train_params=None,
        pred_params=None,
    ):
        """
        Args:
            text_encoder (BertForXMC, RobertaForXMC, XLMRobertaForXMC or XLNetForXMC): text text_encoder
                using transformer model
            text_tokenizer (BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer or XLNetTokenizer): text
                text_tokenizer for to convert raw text to torch tensors
            text_model (TransformerLinearXMCHead): linear projection from transformer
                text_encoder to label space
            C (csr_matrix): clustering matrix, shape = (nr_labels, nr_code)
            concat_model (MLModel): linear model that takes concatenation of transformer
                embeddings and input numerical features to predict on label space
            train_params (TransformerMatcher.TrainParams, optional): instance of TransformerMatcher.TrainParams.
            pred_params (TransformerMatcher.PredParams, optional): instance of TransformerMatcher.PredParams.
            model_folder (tempfile.TemporaryDirectory): Temporary directory object. This is a workaround for certain
                tokenizer models that rely on Sentencepiece, which is not in memory.
        """
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.C = C

        self.text_model = text_model
        self.concat_model = concat_model

        self.train_params = self.TrainParams.from_dict(train_params)
        self.pred_params = self.PredParams.from_dict(pred_params)

        # This is for tokenizers that rely on Sentencepiece (e.x. XLNetTokenizer)
        # re-pointing the text_tokenizer files to self.temp_folder.name
        self.temp_folder = tempfile.TemporaryDirectory()
        self.text_tokenizer.save_pretrained(self.temp_folder.name)
        self.text_tokenizer = self.text_tokenizer.from_pretrained(self.temp_folder.name)

    def get_pred_params(self):
        return copy.deepcopy(self.pred_params)

    def to_device(self, device, n_gpu=0):
        """Move the text_encoder to desired device

        Args:
            device (torch.device): the destination device
            n_gpu (int, optional): if > 1, text_encoder will be converted to torch.nn.DataParallel to use multi-GPU
        """
        self.text_encoder.to(device)
        # multi-gpu eval
        if n_gpu > 1 and not isinstance(self.text_encoder, torch.nn.DataParallel):
            self.text_encoder = torch.nn.DataParallel(self.text_encoder)
        return self

    def clear_cuda(self):
        """Clear CUDA memory"""
        if hasattr(self.text_encoder, "module"):
            self.text_encoder = self.text_encoder.module
        self.text_encoder.to(torch.device("cpu"))
        self.text_model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        return self

    @classmethod
    def get_loss_function(cls, loss_function):
        """Get the loss function for training

        Args:
            loss_function (str): type of loss function, in TransformerMatcher.LOSS_FUNCTION_TYPES

        Returns:
            loss_function (torch.nn.module)
        """
        return cls.LOSS_FUNCTION_TYPES[loss_function]

    @property
    def device(self):
        """Get the current device of the text_encoder

        Returns:
            torch.device
        """
        if hasattr(self.text_encoder, "module"):
            return self.text_encoder.module.device
        else:
            return self.text_encoder.device

    @property
    def nr_codes(self):
        """Get the number of codes"""
        return self.C.shape[1]

    @property
    def nr_features(self):
        """Get the feature dimension of concat_model"""
        return self.concat_model.nr_features if self.concat_model else None

    @property
    def nr_labels(self):
        """Get the number of labels"""
        return self.text_model.num_labels

    @property
    def model_type(self):
        """Get the encoder model type"""
        if hasattr(self.text_encoder, "module"):
            return self.text_encoder.module.config.model_type
        else:
            return self.text_encoder.config.model_type

    def save(self, save_dir):
        """Save the models, text_tokenizer and training arguments to file

        Args:
            save_dir (str): dir to save the model, will be created if it doesn't exist
        """
        os.makedirs(save_dir, exist_ok=True)
        # use .module when do parallel training
        encoder_to_save = (
            self.text_encoder.module if hasattr(self.text_encoder, "module") else self.text_encoder
        )

        param = {
            "model": self.__class__.__name__,
            "text_encoder": encoder_to_save.__class__.__name__,
            "nr_labels": self.nr_labels,
            "nr_features": self.nr_features,
            "nr_codes": self.nr_codes,
            "train_params": self.train_params.to_dict(),
            "pred_params": self.pred_params.to_dict(),
        }
        param = self.append_meta(param)
        with open(os.path.join(save_dir, "param.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(param, indent=True))

        smat_util.save_matrix(os.path.join(save_dir, "C.npz"), self.C)

        encoder_dir = os.path.join(save_dir, "text_encoder")
        os.makedirs(encoder_dir, exist_ok=True)
        # this creates config.json, pytorch_model.bin
        encoder_to_save.save_pretrained(encoder_dir)
        # this creates text_tokenizer files
        tokenizer_dir = os.path.join(save_dir, "text_tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        self.text_tokenizer.save_pretrained(tokenizer_dir)

        # this creates text_model
        text_model_dir = os.path.join(save_dir, "text_model")
        torch.save(self.text_model, text_model_dir)
        # save the concat_model
        concat_model_dir = os.path.join(save_dir, "concat_model")
        if self.concat_model:
            self.concat_model.save(concat_model_dir)

    @classmethod
    def load(cls, load_dir):
        """Load models, text_tokenizer and training arguments from file

        Args:
            load_dir (str): dir to load the models, text_tokenizer and training arguments

        Returns:
            TransformerMatcher
        """
        # load TrainParams and PredParams
        param_dir = os.path.join(load_dir, "param.json")
        param = dict()
        if os.path.exists(param_dir):
            param = json.loads(open(param_dir, "r").read())
        train_params = cls.TrainParams.from_dict(param.get("train_params", None))
        pred_params = cls.PredParams.from_dict(param.get("pred_params", None))

        # load text_encoder
        encoder_dir = os.path.join(load_dir, "text_encoder")
        if not os.path.isdir(encoder_dir):
            raise ValueError(f"text_encoder does not exist at {encoder_dir}")

        with open(os.path.join(encoder_dir, "config.json"), "r", encoding="utf-8") as fin:
            transformer_type = json.loads(fin.read())["model_type"]
        dnn_type = ENCODER_CLASSES[transformer_type]
        encoder_config = dnn_type.config_class.from_pretrained(encoder_dir)
        text_encoder, loading_info = dnn_type.model_class.from_pretrained(
            encoder_dir, config=encoder_config, output_loading_info=True
        )
        if len(loading_info["missing_keys"]) > 0:
            LOGGER.warning(
                "Weights of {} not initialized from pre-trained text_encoder: {}".format(
                    text_encoder.__class__.__name__, loading_info["missing_keys"]
                )
            )

        # load text_tokenizer
        tokenizer_dir = os.path.join(load_dir, "text_tokenizer")
        if not os.path.isdir(tokenizer_dir):
            raise ValueError(f"text_tokenizer does not exist at {tokenizer_dir}")
        text_tokenizer = dnn_type.tokenizer_class.from_pretrained(tokenizer_dir)

        # load text_model
        text_model_dir = os.path.join(load_dir, "text_model")
        if os.path.exists(text_model_dir):
            text_model = torch.load(text_model_dir)
        else:
            text_model = TransformerLinearXMCHead(
                encoder_config.hidden_size, encoder_config.num_labels
            )
            LOGGER.warning(
                f"XMC text_model of {text_encoder.__class__.__name__} not initialized from pre-trained model."
            )

        # load C
        C_path = os.path.join(load_dir, "C.npz")
        if not os.path.exists(C_path):
            raise ValueError(f"Cluster code does not exist at {C_path}")
        C = smat_util.load_matrix(C_path)

        # load concat_model
        concat_model_dir = os.path.join(load_dir, "concat_model")
        concat_model = None
        if os.path.exists(concat_model_dir):
            concat_model = MLModel.load(concat_model_dir)

        return cls(
            text_encoder,
            text_tokenizer,
            text_model,
            C=C,
            concat_model=concat_model,
            train_params=train_params,
            pred_params=pred_params,
        )

    @classmethod
    def download_model(cls, model_shortcut, num_labels=2, hidden_dropout_prob=0.1, cache_dir=""):
        """Initialize a matcher by downloading a pre-trained model from s3

        Args:
            model_shortcut (str): model name shortcut, e.g. 'bert-base-cased'
            num_labels (int): model output size
            hidden_dropout_prob (float, optional): hidden states dropout probability. Default 0.1
            cache_dir (str, optional): path to store downloaded model, if the model already exists
                            at cache_dir, downloading will be ignored

        Returns:
            TransformerMatcher
        """
        use_cache = cache_dir if cache_dir else None
        # AutoConfig will infer transformer type from shortcut
        config = AutoConfig.from_pretrained(
            model_shortcut,
            hidden_dropout_prob=hidden_dropout_prob,
            output_hidden_states=False,
            summary_use_proj=False,
            num_labels=num_labels,
            finetuning_task=None,
            cache_dir=use_cache,  # if None, create temp folder
        )
        if config.model_type not in ENCODER_CLASSES:
            raise ValueError(f"Model type {config.model_type} not supported.")

        dnn_type = ENCODER_CLASSES[config.model_type]
        text_tokenizer = dnn_type.tokenizer_class.from_pretrained(
            model_shortcut,
            cache_dir=use_cache,
        )
        text_encoder = dnn_type.model_class.from_pretrained(
            model_shortcut,
            config=config,
            cache_dir=use_cache,
        )
        text_model = TransformerLinearXMCHead(config.hidden_size, num_labels)
        return cls(text_encoder, text_tokenizer, text_model)

    def text_to_tensor(self, corpus, max_length=None, **kwargs):
        """Convert input text corpus into padded tensors

        Args:
            corpus (iterable over str): input text strings
            max_length(int, optional): max length to which input text will be padded/truncated.
                                    Default None to use the max length in the corpus

        Returns:
            feature_tensors (dict): {
                                    "input_ids": tensor of input token ids,
                                    "attention_mask": tensor of attention masks,
                                    "token_type_ids": tensor of token type ids,
                                    }
        """
        convert_kwargs = {
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
            "max_length": max_length,
            "return_tensors": "pt",  # return pytorch tensors
            "return_token_type_ids": True,
            "return_attention_mask": True,
        }
        # this it to disable the warning message for tokenizer
        # REF: https://github.com/huggingface/transformers/issues/5486
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        LOGGER.info("***** Encoding data len={} truncation={}*****".format(len(corpus), max_length))
        t_start = time.time()
        feature_tensors = self.text_tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=corpus,
            **convert_kwargs,
        )

        LOGGER.info("***** Finished with time cost={} *****".format(time.time() - t_start))
        return feature_tensors

    @staticmethod
    def _get_label_tensors(M, Y, idx_padding=-1, val_padding=0, max_labels=None):
        """
        Given matching matrix M and label matrix Y, construct label tensors for XMC training
        The non-zero indices of Y are seen as positive labels and therefore all
        included in the result.

        Example:
            M = smat.csr_matrix([[1, 1, 0, 0],
                                 [0, 0, 1, 1]])
            Y = smat.csr_matrix([[0, 1, 0, 2],
                                 [0, 0, 0, 3]])
            then the returned values will be:
            label_indices = torch.IntTensor([[1, 3, 0], [3, 2, -1]])
            label_values = torch.FloatTensor([[1., 2., 0.], [3., 0., 0.]])

        Args:
            M (csr_matrix or None): matching matrix, shape = (nr_inst, nr_labels)
                It's indices are the candidate label indices to consider
                It's values will not be used
            Y (csr_matrix or None): label matrix, shape = (nr_inst, nr_labels)
                It's non-zero indices are positive labels and will always be
                included.
            idx_padding (int, optional): the index used to pad all label_indices
                to the same length. Default -1
            val_padding (float, optional): the value used to fill in
                label_values corresponding to the zero entrees in Y. Default 0
            max_labels (int, optional): max number of labels considered for each
                instance, will subsample from existing label indices if need to.
                Default None to use max row nnz of M.

        Returns:
            label_indices (torch.IntTensor or None): containing label indices with
                shape = (nr_inst, max_labels). Return None if M is None
            label_values (torch.FloatTensor or None): containing label values
                with shape = (nr_inst, max_labels). If Y is None, return None
        """
        if M is None and Y is None:
            return None, None
        elif M is None and Y is not None:
            # if M is None, taking all labels into account
            return None, torch.FloatTensor(Y.toarray())

        if Y is not None:
            if Y.shape != M.shape:
                raise ValueError("Y and M shape mismatch: {} and {}".format(Y.shape, M.shape))
            label_lower_bound = max(Y.indptr[1:] - Y.indptr[:-1])
            # make sure all positive labels are included
            M1 = smat_util.binarized(M) + smat_util.binarized(Y)
        else:
            M1 = M
            label_lower_bound = 0

        label_upper_bound = max(M1.indptr[1:] - M1.indptr[:-1])
        if max_labels is None:
            max_labels = label_upper_bound
        else:
            max_labels = min(max_labels, label_upper_bound)
            if max_labels < label_lower_bound:
                max_labels = label_lower_bound
                LOGGER.warning(
                    f"Increasing max_labels to {label_lower_bound} to accommodate all positive labels."
                )

        nr_inst = M1.shape[0]
        label_indices = np.zeros((nr_inst, max_labels), dtype=np.int64) + idx_padding
        if Y is not None:
            label_values = np.zeros((nr_inst, max_labels), dtype=np.float32) + val_padding

        for i in range(nr_inst):
            offset = 0
            neg_samples = M1.indices[M1.indptr[i] : M1.indptr[i + 1]]
            # fill with positive samples first
            if Y is not None:
                y_nnz = Y.indptr[i + 1] - Y.indptr[i]
                rng = slice(Y.indptr[i], Y.indptr[i + 1])
                label_indices[i, :y_nnz] = Y.indices[rng]
                label_values[i, :y_nnz] = Y.data[rng]
                offset += y_nnz
                neg_samples = neg_samples[np.invert(np.isin(neg_samples, Y.indices[rng]))]
            # fill the rest slots with negative samples
            if neg_samples.size > max_labels - offset:
                # random sample negative labels
                neg_samples = np.random.choice(neg_samples, max_labels - offset)

            label_indices[i, offset : offset + neg_samples.size] = neg_samples

        label_indices = torch.IntTensor(label_indices)

        return label_indices, None if Y is None else torch.FloatTensor(label_values)

    @staticmethod
    def ensemble_prediction(transformer_pred_csr, concat_pred_csr, only_topk, ens_method):
        """Generate micro ensemble of concat predictions and transformer predictions

        Args:
            transformer_pred_csr (csr_matrix): transformer prediction matrix
            concat_pred_csr (csr_matrix): concat_model prediction matrix
            only_topk (int): number of top predictions to gather
            ens_method (str): the method used for micro ensemble. Choices:
                    concat-only: only use concat model predictions
                    transformer-only: only use transformer predictions
                    rank_average: rank-average concat/transformer predictions
                    round_robin: round robin ensemble liner/transformer predictions
                    average: average concat/transformer predictions

        Returns:
            ensemble_pred_csr (csr_matrix)
        """

        if transformer_pred_csr.shape != concat_pred_csr.shape:
            raise ValueError(
                f"Transformer/concat prediction mismatch: {transformer_pred_csr.shape} and {concat_pred_csr.shape}"
            )
        LOGGER.info(f"Using {ens_method} for transformer/concat ensemble of pred_csr")
        if ens_method == "concat-only":
            pred_csr_codes = concat_pred_csr
        elif ens_method == "transformer-only":
            pred_csr_codes = transformer_pred_csr
        elif ens_method == "average":
            pred_csr_codes = smat_util.CsrEnsembler.average(
                smat_util.sorted_csr(transformer_pred_csr),
                smat_util.sorted_csr(concat_pred_csr),
            )
        elif ens_method == "rank_average":
            pred_csr_codes = smat_util.CsrEnsembler.rank_average(
                smat_util.sorted_csr(transformer_pred_csr),
                smat_util.sorted_csr(concat_pred_csr),
            )
        elif ens_method == "round_robin":
            pred_csr_codes = smat_util.CsrEnsembler.round_robin(
                smat_util.sorted_csr(transformer_pred_csr),
                smat_util.sorted_csr(concat_pred_csr),
            )
        else:
            raise ValueError(f"Unknown ensemble method {ens_method}")
        return smat_util.sorted_csr(pred_csr_codes.astype(np.float32), only_topk=only_topk)

    def predict(
        self,
        X_text,
        X_feat=None,
        csr_codes=None,
        pred_params=None,
        **kwargs,
    ):
        """Predict with the transformer matcher, allow batch prediction to reduce memory cost

        Args:
            X_text (list of str or dict): prediction input text or
                dictionary of encoded tensors:
                    {
                    "input_ids": tensor of input token ids,
                    "attention_mask": tensor of attention masks,
                    "token_type_ids": tensor of token type ids,
                    }
            X_feat (csr_matrix or ndarray, optional): prediction instance
                    feature matrix, shape = (nr_insts, nr_features)
            csr_codes (csr_matrix, optional): matching matrix, shape = (nr_inst, nr_codes),
                    only its non-zero entrees will be evaluated.
                    Its values will be combined with current prediction through post_processor.
                    Default None to consider all labels.
            pred_params (TransformerMatcher.PredParams, optional): instance of TransformerMatcher.PredParams
                    or dictionary to override stored pred_params. Default None to ignore overriding
            kwargs:
                batch_size (int, optional): total batch_size for (multi-GPU) forward propagation. Default 8
                batch_gen_workers (int, optional): number of CPU workers for batch generation. Default 4
                max_pred_chunk (int, optional): maximum number of instances to
                        predict on for each round. Default None to predict on all
                        instances at once. Default 10^7
                only_embeddings (bool, optional): if True, skip logit prediction and only produce embeddings

        Returns:
            label_pred (csr_matrix): label prediction logits, shape = (nr_inst, nr_labels)
            embeddings (ndarray): array of instance embeddings shape = (nr_inst, hidden_dim)
        """
        if pred_params is None:
            pred_params = self.get_pred_params()
        elif isinstance(pred_params, dict):
            pred_params = self.get_pred_params().override_with_kwargs(pred_params)
        elif not isinstance(pred_params, TransformerMatcher.PredParams):
            raise TypeError(f"Unsupported type for pred_params: {type(pred_params)}")

        if isinstance(X_text, list):
            X_text = self.text_to_tensor(
                X_text,
                num_workers=kwargs.get("batch_gen_workers", 4),
                max_length=pred_params.truncate_length,
            )

        nr_inst = X_text["input_ids"].shape[0]
        max_pred_chunk = kwargs.pop("max_pred_chunk", 10 ** 7)

        if max_pred_chunk is None or max_pred_chunk >= nr_inst:
            label_pred, embeddings = self._predict(
                X_text,
                X_feat=X_feat,
                csr_codes=csr_codes,
                pred_params=pred_params,
                **kwargs,
            )
        else:
            # batch prediction to avoid OOM
            embedding_chunks = []
            P_chunks = []
            for i in range(0, nr_inst, max_pred_chunk):
                cur_P, cur_embedding = self._predict(
                    {k: v[i : i + max_pred_chunk] for k, v in X_text.items()},
                    X_feat=None if X_feat is None else X_feat[i : i + max_pred_chunk, :],
                    csr_codes=None if csr_codes is None else csr_codes[i : i + max_pred_chunk, :],
                    pred_params=pred_params,
                    **kwargs,
                )
                embedding_chunks.append(cur_embedding)
                P_chunks.append(cur_P)
            if not all(pp is None for pp in P_chunks):
                label_pred = smat_util.vstack_csr(P_chunks)
            else:
                # only_embeddings case
                label_pred = None
            embeddings = np.vstack(embedding_chunks)
        return label_pred, embeddings

    def _predict(
        self,
        X_text,
        X_feat=None,
        csr_codes=None,
        pred_params=None,
        **kwargs,
    ):
        """Predict with the transformer matcher

        Args:
            X_text (dict): prediction inputs, dictionary of tensors
                    {
                    "input_ids": tensor of input token ids,
                    "attention_mask": tensor of attention masks,
                    "token_type_ids": tensor of token type ids,
                    }
            X_feat (csr_matrix or ndarray, optional): prediction instance feature matrix,
                    shape = (nr_insts, nr_features)
            csr_codes (csr_matrix, optional): matching matrix, shape = (nr_inst, nr_codes),
                    only its non-zero entrees will be evaluated.
                    Its values will be combined with current prediction through post_processor.
                    Default None to consider all labels.
            pred_params (TransformerMatcher.PredParams, optional): instance of TransformerMatcher.PredParams
                    or dictionary to override stored pred_params. Default None to ignore overriding
            kwargs:
                batch_size (int, optional): total batch_size for (multi-GPU) forward propagation. Default 8
                batch_gen_workers (int, optional): number of CPU workers for batch generation. Default 4
                only_embeddings (bool, optional): if True, skip logit prediction and only produce embeddings

        Returns:
            label_pred (csr_matrix): label prediction logits, shape = (nr_inst, nr_labels)
            embeddings (ndarray): array of instance embeddings shape = (nr_inst, hidden_dim)
        """
        batch_gen_workers = kwargs.get("batch_gen_workers", 4)
        only_embeddings = kwargs.get("only_embeddings", False)

        if csr_codes is not None:
            # need to keep explicit zeros in csr_codes_next
            # therefore do not pass it through constructor
            if not isinstance(csr_codes, smat.csr_matrix):
                raise TypeError(f"Got type={type(csr_codes)} for csr_codes!")
            # getting the result in csr by computing csr * csr
            csr_codes_next = clib.sparse_matmul(
                csr_codes,
                self.C.T,
                eliminate_zeros=False,
                threads=batch_gen_workers,
            )

            LOGGER.info(
                "Predict with csr_codes_next({}) with avr_nnz={}".format(
                    csr_codes_next.shape, csr_codes_next.nnz / csr_codes_next.shape[0]
                )
            )
        else:
            csr_codes_next = None
            LOGGER.info("Predict on input text tensors({})".format(X_text["input_ids"].shape))

        label_indices_pt, label_values_pt = TransformerMatcher._get_label_tensors(
            csr_codes_next, None, idx_padding=self.text_model.label_pad
        )
        data = XMCDataset(
            X_text["input_ids"],
            X_text["attention_mask"],
            X_text["token_type_ids"],
            torch.arange(X_text["input_ids"].shape[0]),
            label_values=label_values_pt,
            label_indices=label_indices_pt,
        )

        # since number of active labels may vary
        # using pinned memory will slow down data loading
        dataloader = DataLoader(
            data,
            sampler=SequentialSampler(data),
            pin_memory=False,
            batch_size=kwargs.get("batch_size", 8),
            num_workers=batch_gen_workers,
        )

        local_topk = min(pred_params.only_topk, self.nr_labels)

        embeddings = []
        batch_cpred = []
        for batch in dataloader:
            self.text_encoder.eval()
            self.text_model.eval()
            cur_batch_size = batch[0].shape[0]
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "instance_number": batch[3],
                    "label_values": None,
                    "label_indices": None if csr_codes_next is None else batch[-1],
                }

                if not only_embeddings:
                    text_model_W_seq, text_model_b_seq = self.text_model(
                        output_indices=inputs["label_indices"],
                        num_device=len(self.text_encoder.device_ids)
                        if hasattr(self.text_encoder, "device_ids")
                        else 1,
                    )

                outputs = self.text_encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                    label_embedding=None
                    if only_embeddings
                    else (text_model_W_seq, text_model_b_seq),
                )

                if not only_embeddings:
                    c_pred = outputs["logits"]
                    # get topk prediction
                    if csr_codes_next is None:  # take all labels into consideration
                        cpred_csr = smat.csr_matrix(c_pred.cpu().numpy())
                        cpred_csr.data = PostProcessor.get(pred_params.post_processor).transform(
                            cpred_csr.data, inplace=True
                        )
                        cpred_csr = smat_util.sorted_csr(cpred_csr, only_topk=local_topk)
                        batch_cpred.append(cpred_csr)
                    else:
                        cur_act_labels = csr_codes_next[inputs["instance_number"].cpu()]
                        nnz_of_insts = cur_act_labels.indptr[1:] - cur_act_labels.indptr[:-1]
                        inst_idx = np.repeat(
                            np.arange(cur_batch_size, dtype=np.uint32), nnz_of_insts
                        )
                        label_idx = cur_act_labels.indices.astype(np.uint32)
                        val = c_pred.cpu().numpy().flatten()
                        val = val[
                            np.argwhere(
                                inputs["label_indices"].cpu().flatten() != self.text_model.label_pad
                            )
                        ].flatten()
                        val = PostProcessor.get(pred_params.post_processor).transform(
                            val, inplace=True
                        )
                        val = PostProcessor.get(pred_params.post_processor).combiner(
                            val, cur_act_labels.data
                        )
                        cpred_csr = smat_util.sorted_csr_from_coo(
                            cur_act_labels.shape, inst_idx, label_idx, val, only_topk=local_topk
                        )

                        batch_cpred.append(cpred_csr)

                embeddings.append(outputs["pooled_output"].cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        pred_csr_codes = None
        if not only_embeddings:
            pred_csr_codes = smat_util.vstack_csr(batch_cpred)
            ens_method = pred_params.ensemble_method
            # concat_model prediction requires concat_model and X_feat
            if all(v is not None for v in [self.concat_model, X_feat]):
                concat_pred_csr_codes = self.concat_model.predict(
                    TransformerMatcher.concat_features(X_feat, embeddings, normalize_emb=True),
                    csr_codes=csr_codes,  # use original csr_codes rather than csr_codes_next
                    only_topk=local_topk,
                    post_processor=pred_params.post_processor,
                )
                pred_csr_codes = TransformerMatcher.ensemble_prediction(
                    pred_csr_codes, concat_pred_csr_codes, local_topk, ens_method
                )
            elif self.concat_model is not None and ens_method != "transformer-only":
                LOGGER.warning(
                    f"X_feat is missing for {ens_method} prediction, fall back to transformer-only"
                )

        return pred_csr_codes, embeddings

    @staticmethod
    def concat_features(X_feat, X_emb, normalize_emb=True):
        """Concatenate instance numerical features with transformer embeddings

        Args:
            X_feat (csr_matrix or ndarray): instance numerical features of shape (nr_inst, nr_features)
            X_emb (ndarray): instance embeddings of shape (nr_inst, hidden_dim)
            normalize_emb (bool, optional): if True, rowwise normalize X_emb before concatenate.
                Default False

        Returns:
            X_cat (csr_matrix or ndarray): the concatenated features
        """
        if normalize_emb:
            X_cat = sk_normalize(X_emb)
        else:
            X_cat = X_emb

        if isinstance(X_feat, smat.csr_matrix):
            X_cat = smat_util.dense_to_csr(X_cat)
            X_cat = smat_util.hstack_csr([X_feat, X_cat], dtype=np.float32)
        elif isinstance(X_feat, np.ndarray):
            X_cat = np.hstack([X_feat, X_cat])
        elif X_feat is None:
            pass
        else:
            raise TypeError(f"Expected CSR or ndarray, got {type(X_feat)}")
        return X_cat

    def fine_tune_encoder(self, prob, val_prob=None, val_csr_codes=None):
        """Fine tune the transformer text_encoder

        Args:
            prob (MLProblemWithText): training problem
            val_prob (MLProblemWithText, optional): validation problem
            val_csr_codes (csr_matrix, optional): prediction matrix for
                    validation data, shape = (nr_val_inst, nr_codes)
                    its values and indices will be used in combiner for next
                    prediction

        Returns:
            TransformerMatcher
        """
        train_params = self.train_params
        pred_params = self.pred_params

        loss_function = TransformerMatcher.get_loss_function(train_params.loss_function).to(
            self.device
        )

        max_act_labels = train_params.max_active_matching_labels
        logging_steps = train_params.logging_steps
        max_steps = train_params.max_steps
        max_no_improve_cnt = train_params.max_no_improve_cnt
        if prob.M is not None:
            # need to keep explicit zeros in csr_codes_next
            # therefore do not pass it through constructor
            if not isinstance(prob.M, smat.csr_matrix):
                raise TypeError(f"Got type={type(prob.M)} for M!")
            # getting the result in csr by computing csr * csr
            M_next = clib.sparse_matmul(
                prob.M,
                self.C.T,
                eliminate_zeros=False,
                threads=train_params.batch_gen_workers,
            )

            do_resample = max_act_labels is not None and max_act_labels < max(
                M_next.indptr[1:] - M_next.indptr[:-1]
            )
        else:
            M_next = None
            do_resample = False

        if prob.M is None or train_params.max_num_labels_in_gpu >= self.nr_labels:
            # put text_model to GPU
            self.text_model.to(self.device)

        label_indices_pt, label_values_pt = TransformerMatcher._get_label_tensors(
            M_next,
            prob.Y,
            idx_padding=self.text_model.label_pad,
            max_labels=max_act_labels,
        )
        train_data = XMCDataset(
            prob.X_text["input_ids"],
            prob.X_text["attention_mask"],
            prob.X_text["token_type_ids"],
            torch.arange(prob.X_text["input_ids"].shape[0]),  # instance number
            label_values=label_values_pt,
            label_indices=label_indices_pt,
        )

        # since number of active labels may vary
        # using pinned memory will slow down data loading
        train_dataloader = DataLoader(
            train_data,
            sampler=RandomSampler(train_data),
            pin_memory=False,
            batch_size=train_params.batch_size,
            num_workers=train_params.batch_gen_workers,
        )

        # compute stopping criteria
        if train_params.max_steps > 0:
            t_total = train_params.max_steps
            steps_per_epoch = len(train_dataloader) // train_params.gradient_accumulation_steps
            train_params.num_train_epochs = train_params.max_steps // steps_per_epoch + 1
        else:
            steps_per_epoch = len(train_dataloader) // train_params.gradient_accumulation_steps
            t_total = steps_per_epoch * train_params.num_train_epochs

        train_params.save_steps = min(train_params.save_steps, t_total)
        train_params.logging_steps = min(train_params.logging_steps, t_total)

        # Prepare optimizer, disable weight decay for bias and layernorm weights
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.text_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": train_params.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.text_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=train_params.learning_rate,
            eps=train_params.adam_epsilon,
        )
        scheduler = get_scheduler(
            train_params.lr_schedule,
            optimizer,
            num_warmup_steps=train_params.warmup_steps,
            num_training_steps=t_total,
        )

        sparse_parameters = list(self.text_model.parameters())
        if prob.M is not None:
            emb_optimizer = torch.optim.SparseAdam(
                sparse_parameters,
                lr=train_params.learning_rate,
                eps=train_params.adam_epsilon,
            )
        else:
            # For the top matcher treat text_model as dense layer
            emb_optimizer = AdamW(
                sparse_parameters,
                lr=train_params.learning_rate,
                eps=train_params.adam_epsilon,
            )
        emb_scheduler = get_scheduler(
            train_params.lr_schedule,
            emb_optimizer,
            num_warmup_steps=train_params.warmup_steps,
            num_training_steps=t_total,
        )

        # Start Batch Training
        LOGGER.info("***** Running training *****")
        LOGGER.info("  Num examples = %d", prob.X_text["input_ids"].shape[0])
        LOGGER.info("  Num labels = %d", self.nr_labels)
        if prob.M is not None:
            LOGGER.info("  Num active labels per instance = %d", label_indices_pt.shape[1])
        LOGGER.info("  Num Epochs = %d", train_params.num_train_epochs)
        LOGGER.info("  Learning Rate Schedule = %s", train_params.lr_schedule)
        LOGGER.info("  Batch size = %d", train_params.batch_size)
        LOGGER.info("  Gradient Accumulation steps = %d", train_params.gradient_accumulation_steps)
        LOGGER.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        total_train_time, logging_elapsed = 0.0, 0.0
        best_matcher_prec = -1
        avg_matcher_prec = 0
        save_cur_model = False
        no_improve_cnt = 0

        self.text_encoder.zero_grad()
        self.text_model.zero_grad()
        for epoch in range(1, int(train_params.num_train_epochs) + 1):
            if do_resample and epoch > 1:  # redo subsample negative labels
                label_indices_pt, label_values_pt = TransformerMatcher._get_label_tensors(
                    M_next,
                    prob.Y,
                    idx_padding=self.text_model.label_pad,
                    max_labels=train_params.max_active_matching_labels,
                )
                train_data.refresh_labels(
                    label_values=label_values_pt,
                    label_indices=label_indices_pt,
                )
            for batch_cnt, batch in enumerate(train_dataloader):
                self.text_encoder.train()
                self.text_model.train()
                start_time = time.time()
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "instance_number": batch[3],
                    "label_values": batch[4],
                    "label_indices": batch[-1] if prob.M is not None else None,
                }
                text_model_W_seq, text_model_b_seq = self.text_model(
                    output_indices=inputs["label_indices"],
                    num_device=len(self.text_encoder.device_ids)
                    if hasattr(self.text_encoder, "device_ids")
                    else 1,
                )
                outputs = self.text_encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                    label_embedding=(text_model_W_seq, text_model_b_seq),
                )
                loss = loss_function(outputs["logits"], inputs["label_values"].to(self.device))
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if train_params.gradient_accumulation_steps > 1:
                    loss = loss / train_params.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()

                logging_elapsed += time.time() - start_time
                total_train_time += time.time() - start_time
                if (batch_cnt + 1) % train_params.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.text_encoder.parameters(), train_params.max_grad_norm
                    )

                    optimizer.step()  # perform gradient update
                    scheduler.step()  # update learning rate schedule
                    optimizer.zero_grad()  # clear gradient accumulation

                    torch.nn.utils.clip_grad_norm_(
                        self.text_model.parameters(), train_params.max_grad_norm
                    )
                    emb_optimizer.step()  # perform gradient update
                    emb_scheduler.step()  # update learning rate schedule
                    emb_optimizer.zero_grad()  # clear gradient accumulation
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        cur_loss = (tr_loss - logging_loss) / logging_steps
                        LOGGER.info(
                            "| [{:4d}/{:4d}][{:6d}/{:6d}] | {:4d}/{:4d} batches | ms/batch {:5.4f} | train_loss {:6e} | lr {:.6e}".format(
                                int(epoch),
                                int(train_params.num_train_epochs),
                                int(global_step),
                                int(t_total),
                                int(batch_cnt),
                                len(train_dataloader),
                                logging_elapsed * 1000.0 / logging_steps,
                                cur_loss,
                                scheduler.get_last_lr()[0],
                            )
                        )
                        logging_loss = tr_loss
                        logging_elapsed = 0

                    if train_params.save_steps > 0 and global_step % train_params.save_steps == 0:
                        if val_prob is not None:
                            if val_prob.M is None:
                                test_combos = zip(["all"], [None])
                            else:
                                test_combos = zip(
                                    ["trn_ns", "pred_ns"], [val_prob.M, val_csr_codes]
                                )
                            for val_type, valid_M in test_combos:
                                avr_beam = 1 if valid_M is None else valid_M.nnz / valid_M.shape[0]
                                # compute loss and prediction on test set
                                val_pred, _ = self.predict(
                                    val_prob.X_text,
                                    csr_codes=valid_M,
                                    batch_size=train_params.batch_size,
                                    batch_gen_workers=train_params.batch_gen_workers,
                                    pred_params={"ensemble_method": "transformer-only"},
                                )
                                LOGGER.info("-" * 89)
                                LOGGER.info(
                                    "| epoch {:3d} step {:6d} evaluation | training-time: {:5.4f}s average-beam: {:5.1f}".format(
                                        epoch,
                                        global_step,
                                        total_train_time,
                                        avr_beam,
                                    )
                                )
                                # compute precision on test set
                                val_metrics = smat_util.Metrics.generate(
                                    val_prob.Y,
                                    val_pred,
                                    topk=pred_params.only_topk,
                                )
                                LOGGER.info(
                                    "| {} test-prec {}".format(
                                        val_type,
                                        " ".join(
                                            "{:4.2f}".format(100 * v) for v in val_metrics.prec
                                        ),
                                    )
                                )
                                LOGGER.info(
                                    "| {} test-recl {}".format(
                                        val_type,
                                        " ".join(
                                            "{:4.2f}".format(100 * v) for v in val_metrics.recall
                                        ),
                                    )
                                )

                            avg_matcher_prec = np.mean(val_metrics.prec)
                            # save the model with highest val precision
                            save_cur_model = avg_matcher_prec > best_matcher_prec
                        else:
                            # if val set not given, always save
                            save_cur_model = True

                        if save_cur_model:
                            no_improve_cnt = 0
                            LOGGER.info(
                                "| **** saving model (avg_prec={}) to {} at global_step {} ****".format(
                                    100 * avg_matcher_prec,
                                    train_params.checkpoint_dir,
                                    global_step,
                                )
                            )
                            best_matcher_prec = avg_matcher_prec
                            self.save(train_params.checkpoint_dir)
                        else:
                            no_improve_cnt += 1
                        LOGGER.info("-" * 89)

                if (max_steps > 0 and global_step > max_steps) or (
                    max_no_improve_cnt > 0 and no_improve_cnt >= max_no_improve_cnt
                ):
                    break
            if (max_steps > 0 and global_step > max_steps) or (
                max_no_improve_cnt > 0 and no_improve_cnt >= max_no_improve_cnt
            ):
                break

        return self

    @classmethod
    def train(
        cls,
        prob,
        csr_codes=None,
        val_prob=None,
        val_csr_codes=None,
        train_params=None,
        pred_params=None,
        **kwargs,
    ):
        """Train the transformer matcher

        Args:
            prob (MLProblemWithText): training problem
            csr_codes (csr_matrix, optional): prediction matrix, shape = (nr_inst, nr_codes)
                    its values and indices will be used in combiner for next
                    prediction
            val_prob (MLProblemWithText, optional): validation problem
            val_csr_codes (csr_matrix, optional): prediction matrix for
                    validation data, shape = (nr_val_inst, nr_codes)
                    its values and indices will be used in combiner for next
                    prediction
            train_params (TransformerMatcher.TrainParams, optional): instance of TransformerMatcher.TrainParams.
            pred_params (TransformerMatcher.PredParams, optional): instance of TransformerMatcher.PredParams.
            kwargs:
                saved_trn_pt (str): path to the tokenized trn text. If given, will skip tokenization
                saved_val_pt (str): path to the tokenized val text. If given, will skip tokenization
                bootstrapping (tuple): (init_encoder, init_embeddings) the
                    text_encoder and corresponding instance embeddings generated by it.
                    Used for bootstrap current text_encoder and text_model. Default None to
                    ignore
                return_dict (bool): if True, return a dictionary with model
                     and its prediction/embeddings on train/validation dataset.
                     Default False.
                return_train_pred (bool): if True and return_dict, return prediction matrix on training data
                return_train_embeddings (bool): if True and return_dict, return training instance embeddings
        Returns:
            results (TransformerMatcher or dict):
            if return_dict=True, return a dictionary:
                {
                    matcher: TransformerMatcher instance
                    trn_pred (csr_matrix): topk prediction on training data, shape = (nr_inst, nr_labels)
                    val_pred (csr_matrix or None): topk prediction on validation data, shape = (nr_val_inst, nr_labels)
                    trn_embeddings (ndarray): instance embedding on training data, shape = (nr_inst, hidden_dim).
                    val_embeddings (ndarray or None): instance embedding on validation data, shape = (nr_val_inst, hidden_dim).
                }
            otherwise return the trained TransformerMatcher instance
        """
        train_params = cls.TrainParams.from_dict(train_params)
        pred_params = cls.PredParams.from_dict(pred_params)
        LOGGER.debug(f"TransformerMatcher train_params: {train_params.to_dict()}")
        LOGGER.debug(f"TransformerMatcher pred_params: {pred_params.to_dict()}")
        if prob.X_feat is None:
            pred_params.ensemble_method = "transformer-only"

        # save to a temp dir if not given
        if not train_params.checkpoint_dir:
            temp_dir = tempfile.TemporaryDirectory()
            train_params.checkpoint_dir = temp_dir.name

        if train_params.init_model_dir:
            matcher = cls.load(train_params.init_model_dir)
            LOGGER.info("Loaded model from {}.".format(train_params.init_model_dir))
            if prob.Y.shape[1] != matcher.nr_labels:
                LOGGER.warning(
                    f"Got mismatch nr_labels (expected {prob.Y.shape[1]} but got {matcher.nr_labels}), text_model reinitialized!"
                )
                matcher.text_model = TransformerLinearXMCHead(
                    matcher.text_encoder.config.hidden_size, prob.Y.shape[1]
                )
                matcher.text_encoder.config.num_labels = prob.Y.shape[1]
        else:
            matcher = cls.download_model(
                train_params.model_shortcut,
                prob.Y.shape[1],
                hidden_dropout_prob=train_params.hidden_dropout_prob,
                cache_dir=train_params.cache_dir,
            )
            LOGGER.info("Downloaded {} model from s3.".format(train_params.model_shortcut))

        # assign clusters and train/pred params
        matcher.C = prob.C
        matcher.train_params = train_params
        matcher.pred_params = pred_params

        # tokenize X_text if X_text is given as raw text
        saved_trn_pt = kwargs.get("saved_trn_pt", "")
        if not prob.is_tokenized:
            if saved_trn_pt and os.path.isfile(saved_trn_pt):
                trn_tensors = torch.load(saved_trn_pt)
                LOGGER.info("trn tensors loaded_from {}".format(saved_trn_pt))
            else:
                trn_tensors = matcher.text_to_tensor(
                    prob.X_text,
                    num_workers=train_params.batch_gen_workers,
                    max_length=pred_params.truncate_length,
                )
                if saved_trn_pt:
                    torch.save(trn_tensors, saved_trn_pt)
                    LOGGER.info("trn tensors saved to {}".format(saved_trn_pt))
            prob.X_text = trn_tensors

        if val_prob is not None and not val_prob.is_tokenized:
            saved_val_pt = kwargs.get("saved_val_pt", "")
            if saved_val_pt and os.path.isfile(saved_val_pt):
                val_tensors = torch.load(saved_val_pt)
                LOGGER.info("val tensors loaded from {}".format(saved_val_pt))
            else:
                val_tensors = matcher.text_to_tensor(
                    val_prob.X_text,
                    num_workers=train_params.batch_gen_workers,
                    max_length=pred_params.truncate_length,
                )
                if saved_val_pt:
                    torch.save(val_tensors, saved_val_pt)
                    LOGGER.info("val tensors saved to {}".format(saved_val_pt))
            val_prob.X_text = val_tensors

        bootstrapping = kwargs.get("bootstrapping", None)
        if bootstrapping is not None:
            init_encoder, init_embeddings, prev_head = bootstrapping
            matcher.text_encoder.init_from(init_encoder)
            LOGGER.info("Continue training form given text_encoder!")
            if "linear" in train_params.bootstrap_method:
                bootstrap_prob = MLProblem(
                    init_embeddings,
                    prob.Y,
                    C=prob.C if prob.M is not None else None,
                    M=prob.M,
                    R=prob.Y if "weighted" in train_params.bootstrap_method else None,
                )
                matcher.text_model.bootstrap(bootstrap_prob)
                LOGGER.info("Initialized transformer text_model with xlinear!")
            elif train_params.bootstrap_method == "inherit":
                matcher.text_model.inherit(prev_head, prob.C)
                LOGGER.info("Initialized transformer text_model form parent layer!")
            elif train_params.bootstrap_method == "no-bootstrap":
                matcher.text_model.random_init()
                LOGGER.info("Randomly initialized transformer text_model!")
            else:
                raise ValueError(f"Unknown bootstrap_method: {train_params.bootstrap_method}")

        # move matcher to desired hardware
        device, n_gpu = torch_util.setup_device(train_params.use_gpu)
        matcher.to_device(device, n_gpu)
        train_params.batch_size *= max(1, n_gpu)

        # train the matcher
        if train_params.max_steps > 0 or train_params.num_train_epochs > 0:
            LOGGER.info("Start fine-tuning transformer matcher...")
            matcher.fine_tune_encoder(prob, val_prob=val_prob, val_csr_codes=val_csr_codes)
            if os.path.exists(train_params.checkpoint_dir):
                LOGGER.info(
                    "Reload the best checkpoint from {}".format(train_params.checkpoint_dir)
                )
                matcher = TransformerMatcher.load(train_params.checkpoint_dir)
                matcher.to_device(device, n_gpu)

        # ignore concat_model even if there exist one
        matcher.concat_model = None

        return_dict = kwargs.get("return_dict", False)
        return_train_pred = kwargs.get("return_train_pred", False) and return_dict
        return_train_embeddings = kwargs.get("return_train_embeddings", False) and return_dict

        P_trn, inst_embeddings = None, None
        train_concat = pred_params.ensemble_method not in ["transformer-only"]
        if train_concat or return_train_pred or return_train_embeddings:
            # getting the instance embeddings of training data
            # since X_feat is not passed, transformer-only result is produced
            P_trn, inst_embeddings = matcher.predict(
                prob.X_text,
                csr_codes=csr_codes,
                pred_params=pred_params,
                batch_size=train_params.batch_size,
                batch_gen_workers=train_params.batch_gen_workers,
            )

        if train_concat:
            # train the same layer concat_model with current embedding
            LOGGER.info("Concatenating instance embeddings with features...")
            cat_embeddings = TransformerMatcher.concat_features(
                prob.X_feat,
                inst_embeddings,
                normalize_emb=True,
            )

            LOGGER.info("Start training concat_model of transformer matcher...")
            lprob = MLProblem(
                cat_embeddings,
                prob.Y,
                C=prob.C if prob.M is not None else None,
                M=prob.M,
                R=sk_normalize(prob.Y, norm="l1") if train_params.cost_sensitive_ranker else None,
            )
            matcher.concat_model = MLModel.train(lprob, threshold=train_params.threshold)
            matcher.save(train_params.checkpoint_dir)

            # P_trn with concat_model
            concat_P_trn = matcher.concat_model.predict(
                lprob.X,
                csr_codes=csr_codes,
                only_topk=pred_params.only_topk,
                post_processor=pred_params.post_processor,
            )
            P_trn = TransformerMatcher.ensemble_prediction(
                P_trn,
                concat_P_trn,
                pred_params.only_topk,
                pred_params.ensemble_method,
            )

        P_val, val_inst_embeddings = None, None
        if val_prob is not None:
            P_val, val_inst_embeddings = matcher.predict(
                val_prob.X_text,
                X_feat=val_prob.X_feat,
                csr_codes=val_csr_codes,
                batch_size=train_params.batch_size,
                batch_gen_workers=train_params.batch_gen_workers,
            )
            LOGGER.info("*************** Final Evaluation ***************")
            # compute precision on test set
            val_type = "man" if val_csr_codes is not None else "all"
            val_metrics = smat_util.Metrics.generate(val_prob.Y, P_val, topk=pred_params.only_topk)
            avr_val_beam = (
                1 if val_csr_codes is None else val_csr_codes.nnz / val_csr_codes.shape[0]
            )
            LOGGER.debug("avr_beam={}".format(avr_val_beam))
            LOGGER.info(
                "| {} test-prec {}".format(
                    val_type,
                    " ".join("{:4.2f}".format(100 * v) for v in val_metrics.prec),
                )
            )
            LOGGER.info(
                "| {} test-recl {}".format(
                    val_type,
                    " ".join("{:4.2f}".format(100 * v) for v in val_metrics.recall),
                )
            )
            LOGGER.info("*" * 72)

        matcher.clear_cuda()

        if return_dict:
            return {
                "matcher": matcher,
                "trn_pred": P_trn if return_train_pred else None,
                "val_pred": P_val,
                "trn_embeddings": inst_embeddings if return_train_embeddings else None,
                "val_embeddings": val_inst_embeddings,
            }
        else:
            return matcher
