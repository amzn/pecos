import dataclasses as dc
import json
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Tuple, Any, Optional, Union

import peft
import torch
from datasets import IterableDataset, Dataset
from torch.utils.data import DataLoader
from peft import AutoPeftModelForSequenceClassification, get_peft_model
from peft.config import PeftConfig
from peft.mixed_model import PeftMixedModel
from peft.peft_model import PeftModel
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from transformers import AutoTokenizer, PreTrainedTokenizer, PretrainedConfig

import pecos
from pecos.xmr.reranker.trainer import RankLlamaTrainer, PARAM_FILENAME
from .data_utils import RankingDataUtils

logger = logging.getLogger(__name__)


class CrossEncoderConfig(PretrainedConfig):
    """
    The configuration class for the cross encoder model. This class contains the model shortcut, model modifier and
    model initialization arguments for the model. The model shortcut is the name of the huggingface model. The
    `model_modifier` is the configuration of the modifier (e.g. PEFT) and the `model_init_kwargs` are the arguments
    for the model.
    """

    model_type = "reranker_crossencoder"

    def __init__(
        self,
        model_shortcut: str = "",
        model_modifier: Dict = {},
        model_init_kwargs: dict = {},
        **kwargs,
    ):
        """
        Initialize the cross encoder configuration
        Args:
            model_shortcut: The model shortcut for the huggingface model
            model_modifier: The model modifier configuration (e.g. PEFT)
            model_init_kwargs: The model initialization arguments. These are the arguments for the huggingface model
        """
        super().__init__(**kwargs)

        self.model_shortcut = model_shortcut
        self.model_modifier = model_modifier
        self.model_init_kwargs = model_init_kwargs


class CrossEncoder(PreTrainedModel):
    """
    The cross encoder model for ranking tasks (retrieval-based). This model is used for training and evaluation.
    It is a wrapper around the huggingface transformer model.
    """

    TRANSFORMER_CLS = AutoModelForSequenceClassification
    TRANSFORMER_PEFT_CLS = AutoPeftModelForSequenceClassification

    @dataclass
    class Config(pecos.BaseParams):
        """Encoder configuration
        model_shortcut (str): the model shortcut of the HuggingFace model
        model_init_kwargs (dict): model initialization kwargs
        model_modifier (dict): model modifier configuration
        """

        model_shortcut: str = ""
        model_init_kwargs: dict = dc.field(default_factory=lambda: dict())
        model_modifier: dict = dc.field(default_factory=lambda: dict())

    config_class = CrossEncoderConfig

    def __init__(self, config: CrossEncoderConfig):
        """
        Initialize the cross encoder model
        Args:
            config: The configuration for the cross encoder
        """
        super().__init__(config)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_shortcut, num_labels=1, **config.model_init_kwargs
        )
        base_model.config.pad_token_id = (
            0 if base_model.config.pad_token_id is None else base_model.config.pad_token_id
        )
        self.hf_model = base_model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        **kwargs,
    ):
        """
        Load the model from the pretrained model name or path. Override the `from_pretrained` method of the
        `PreTrainedModel` class.
        """
        is_local = os.path.isdir(pretrained_model_name_or_path)
        param_folder = pretrained_model_name_or_path

        def super_return():
            return PreTrainedModel.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config,
                cache_dir,
                ignore_mismatched_sizes,
                force_download,
                local_files_only,
                token,
                revision,
                use_safetensors,
                **kwargs,
            )

        if not is_local:
            raise NotImplementedError(f"{cls} can only load local models")

        with open(os.path.join(param_folder, PARAM_FILENAME), "r") as param_file:
            params = json.load(param_file)

        xe_config = CrossEncoder.Config.from_dict(params["model_params"]["encoder_args"])
        xe_config = CrossEncoderConfig(**xe_config.to_dict())
        for k, v in kwargs.items():
            xe_config.model_init_kwargs[k] = v
        model = CrossEncoder(xe_config)

        try:
            if xe_config.model_modifier["modifier_type"] == "peft":
                model = PeftModel.from_pretrained(model, param_folder)
            else:
                super_return()
        except KeyError:
            logger.info("No peft configuration found")

        return model

    def forward(self, *args, **kwargs):
        """
        Returns the forward output of the huggingface model
        """
        return self.hf_model(*args, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the model
        """

        try:
            if self.config.model_modifier["modifier_type"] == "peft":
                self.hf_model.enable_input_require_grads()
        except KeyError:
            pass
        self.hf_model.gradient_checkpointing_enable(**kwargs)


class RankingModel(pecos.BaseClass):
    """
    The ranking model class for training and evaluation of the cross encoder model. This class is used for training
    and evaluation of the cross encoder model. It is a wrapper around the cross encoder model. It also contains the
    parameters for the model. The model can be used for training and evaluation.
    """

    @dataclass
    class TrainParams(pecos.BaseParams):
        """
        The training parameters for the ranking model.
        Args:
            training_args (RankLlamaTrainer.TrainingArgs): The training arguments for the model
            target_data_folder (str): The path to the target data folder
            input_data_folder (str): The path to the input data folder
            label_data_folder (str): The path to the label data folder
        """

        training_args: RankLlamaTrainer.TrainingArgs
        target_data_folder: str = field(
            metadata={
                "help": "Path to folder containing target parquet files (inp_id, [lbl_id], [rel_val])"
            }
        )
        input_data_folder: str = field(
            metadata={"help": "Path to folder containing input parquet files (inp_id, keywords)"}
        )
        label_data_folder: str = field(
            metadata={
                "help": "Path to folder containing label parquet files (lbl_id, title, contents)"
            }
        )

    @dataclass
    class ModelParams(pecos.BaseParams):
        """
        The parameters for the ranking model. This class contains the data, encoder and training arguments for the model.
        """

        encoder_args: CrossEncoder.Config

        positive_passage_no_shuffle: bool = False
        negative_passage_no_shuffle: bool = False
        rerank_max_len: int = 20000
        query_prefix: str = "query: "
        passage_prefix: str = "document: "
        inp_id_col: str = "inp_id"
        lbl_idxs_col: str = "ret_idxs"
        score_col: str = "rel"
        keyword_col_name: str = "keywords"
        content_col_names: List[str] = field(default_factory=lambda: ["title", "contents"])
        content_sep: str = " "
        append_eos_token: bool = False
        pad_to_multiple_of: Optional[int] = 8

    @dataclass
    class EvalParams(pecos.BaseParams):
        """
        Evaluation parameters
        """

        model_name_or_path: str
        target_data_folder: str
        input_data_folder: str
        label_data_folder: str
        output_dir: str
        output_file_prefix: str = "output_"
        output_file_suffix: str = ""
        per_device_eval_batch_size: int = 128
        dataloader_num_workers: int = 2
        dataloader_prefetch_factor: int = 10
        rerank_max_len: int = 196
        query_prefix: str = "query: "
        passage_prefix: str = "document: "
        inp_id_col: str = "inp_id"
        lbl_id_col: str = "lbl_id"
        keyword_col_name: str = "keywords"
        content_col_names: List[str] = field(default_factory=lambda: ["title", "contents"])
        content_sep: str = " "
        append_eos_token: bool = False
        pad_to_multiple_of: int = 16
        bf16: bool = True
        device: str = "cuda"
        model_init_kwargs: dict = dc.field(default_factory=lambda: dict())

    def __init__(
        self,
        encoder: Union[CrossEncoder, PeftModel, PeftMixedModel],
        tokenizer: AutoTokenizer,
        model_params: ModelParams,
        train_params: Optional[TrainParams] = None,
    ):
        """
        Initialize the ranking model. The model contains the encoder, tokenizer, model parameters and training parameters.
        Args:
            encoder (Union[CrossEncoder, PeftModel, PeftMixedModel]): The encoder model
            tokenizer (AutoTokenizer): The tokenizer for the model
            model_params (RankingModel.ModelParams): The model parameters
            train_params (Optional[RankingModel.TrainParams]): The training parameters
        """
        self.tokenizer = tokenizer
        self.cross_encoder = encoder

        self.model_params = self.ModelParams.from_dict(model_params)
        self.train_params = self.TrainParams.from_dict(train_params) if train_params else None

    @classmethod
    def get_modified_model(cls, model: CrossEncoder, mod_config: Dict):
        """
        Takes a pretrained Huggingface model and modifies it to include new features. Currently, the `modifier_type`
        supported by this method is limited to the `peft` package.

        Args:
            model (CrossEncoder): A PreTrainedModel from the transformers package.
            mod_config (Dict): A dictionary containing the configuration for the model modifier.
        Returns: The modified model
        """
        if mod_config["modifier_type"] == "peft":
            config_type = getattr(peft, mod_config["config_type"])
            peft_config: PeftConfig = config_type(**mod_config["config"])

            model = get_peft_model(model, peft_config)

            return model
        else:
            logger.warn("Using model without modifiers (e.g. LoRA)")
            return model

    @classmethod
    def init_model(cls, model_params: ModelParams, train_params: TrainParams):
        """Initiate a model with training parameters

        Args:
            model_params (RankingModel.ModelParams): the model parameters
            train_params (RankingModel.TrainParams): the training parameters
        Returns:
            An instance of RankingModel
        """
        hf_trainer_args = train_params.training_args
        if hf_trainer_args.local_rank > 0:
            torch.distributed.barrier()

        config = model_params.encoder_args.to_dict()
        config = CrossEncoderConfig(**config)
        encoder = CrossEncoder(
            config=config,
        )

        if hf_trainer_args.bf16:
            encoder = encoder.bfloat16()

        if config.model_modifier:
            encoder = cls.get_modified_model(model=encoder, mod_config=config.model_modifier)

        tokenizer = AutoTokenizer.from_pretrained(
            model_params.encoder_args.model_shortcut,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.padding_side = "right"

        if torch.distributed.is_initialized():
            if hf_trainer_args.local_rank == 0:
                torch.distributed.barrier()

        return cls(encoder, tokenizer, model_params, train_params=train_params)

    @classmethod
    def _collate_sharded(
        cls,
        tokenizer: Union[PreTrainedTokenizer, AutoTokenizer],
        model_params: ModelParams,
        train_params: TrainParams,
        table_stores: Dict[str, Dataset],
        data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Collate function for training. Tokenizes the input and return features and returns the collated batch.
        Args:
            tokenizer: The huggingface tokenizer
            params: The model parameters
            table_stores: The table stores for the input and label data
            data: The data to be collated
        Returns: The collated batch in the form of a dictionary with input and scores
        """
        fts_w_scores = []
        for s in data:
            inp_id = s[model_params.inp_id_col]
            retr_idxs = s[model_params.lbl_idxs_col]
            scores = s[model_params.score_col]

            fts_w_scores.append(
                RankingDataUtils._create_sample(
                    inp_id,
                    retr_idxs,
                    scores,
                    table_stores,
                    train_params.training_args.train_group_size,
                    model_params.query_prefix,
                    model_params.passage_prefix,
                    model_params.keyword_col_name,
                    model_params.content_col_names,
                    model_params.content_sep,
                )
            )

        return cls._collate(tokenizer, model_params, fts_w_scores)

    @classmethod
    def _collate(
        cls,
        tokenizer: Union[PreTrainedTokenizer, AutoTokenizer],
        model_params: ModelParams,
        features_w_scores: List[Tuple[Any, Any]],
    ):
        """
        Collate function for training. Tokenizes the input and return features and returns the collated batch.
        Args:
            tokenizer: The huggerface tokenizer
            params: The model parameters
            features_w_scores: Tuple of features list and scores list
        Returns: The collated batch in the form of a dictionary with input and scores
        """
        features = [f for f, _ in features_w_scores]
        scores = [s for _, s in features_w_scores]

        all_pairs = []
        for pairs in features:
            all_pairs.extend(pairs)

        tokenized_pairs = tokenizer(
            all_pairs,
            padding=False,
            truncation=True,
            max_length=(
                model_params.rerank_max_len - 1
                if model_params.append_eos_token
                else model_params.rerank_max_len
            ),
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        if model_params.append_eos_token:
            tokenized_pairs["input_ids"] = [
                p + [tokenizer.eos_token_id] for p in tokenized_pairs["input_ids"]
            ]

        pairs_collated = tokenizer.pad(
            tokenized_pairs,
            padding=True,
            pad_to_multiple_of=model_params.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )
        # NOTE: Here scores has to be flattened, otherwise the huggingface trainer will distribute it
        # incorrectly across devices in distributed training.
        m_scores = torch.tensor(scores, dtype=torch.float).flatten()

        return {"input": pairs_collated, "scores": m_scores}

    @classmethod
    def _collate_sharded_eval(
        cls,
        tokenizer: Union[PreTrainedTokenizer, AutoTokenizer],
        eval_params: EvalParams,
        table_stores: Dict[str, Dataset],
        data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Collate function for evaluation. Tokenizes the input and return features and returns the collated batch.
        Args:
            tokenizer: The huggingface tokenizer
            eval_params: The evaluation parameters
            table_stores: The table stores for the input and label datasets
            data: The data to be collated
        Returns: The collated batch in the form of a dictionary with the tokenized texts together with the input and label indices
        """
        fts = []
        inp_idxs = []
        lbl_idxs = []
        for s in data:
            inp_id = s[eval_params.inp_id_col]
            retr_id = s[eval_params.lbl_id_col]
            inp_idxs.append(inp_id)
            lbl_idxs.append(retr_id)

            fts.append(
                RankingDataUtils._format_sample(
                    table_stores["input"][inp_id][eval_params.keyword_col_name],
                    [table_stores["label"][retr_id][col] for col in eval_params.content_col_names],
                    eval_params.query_prefix,
                    eval_params.passage_prefix,
                    eval_params.content_sep,
                )
            )

        return cls._collate_eval(tokenizer, eval_params, fts, inp_idxs, lbl_idxs)

    @classmethod
    def _collate_eval(
        cls,
        tokenizer: Union[PreTrainedTokenizer, AutoTokenizer],
        eval_params: EvalParams,
        features: List[str],
        inp_idxs: List[int],
        lbl_idxs: List[int],
    ):
        """
        Collate function for training. Tokenizes the input and return features and returns the collated batch.
        Args:
            tokenizer: The huggerface tokenizer
            eval_params: The evaluation parameters
            features: The list of features
            inp_idxs: The list of input indices
            lbl_idxs: The list of label indices
        Returns: The collated batch in the form of a dictionary with tokenized input, input indices and label indices
        """

        all_pairs = []
        for pairs in features:
            all_pairs.append(pairs)

        tokenized_pairs = tokenizer(
            all_pairs,
            padding=False,
            truncation=True,
            max_length=(
                eval_params.rerank_max_len - 1
                if eval_params.append_eos_token
                else eval_params.rerank_max_len
            ),
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        if eval_params.append_eos_token:
            tokenized_pairs["input_ids"] = [
                p + [tokenizer.eos_token_id] for p in tokenized_pairs["input_ids"]
            ]

        pairs_collated = tokenizer.pad(
            tokenized_pairs,
            padding=True,
            pad_to_multiple_of=eval_params.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "inp_idxs": inp_idxs,
            "lbl_idxs": lbl_idxs,
            "inputs": pairs_collated,
        }

    @classmethod
    def predict(
        cls,
        eval_dataset: IterableDataset,
        table_stores: Dict[str, Dataset],
        eval_params: EvalParams,
        tokenizer: AutoTokenizer,
        model: CrossEncoder,
    ):
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0
        tokenizer.padding_side = "right"

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_params.per_device_eval_batch_size,
            # Ensure that at least one worker is creating batches
            # parallel to the model compute
            num_workers=max(eval_params.dataloader_num_workers, 1),
            # To ensure efficiency we prefetch samples in parallel
            prefetch_factor=eval_params.dataloader_prefetch_factor,
            collate_fn=partial(cls._collate_sharded_eval, tokenizer, eval_params, table_stores),
        )

        all_results = []
        for batch in eval_dataloader:
            with torch.inference_mode():
                inp_ids = batch["inp_idxs"]
                lbl_ids = batch["lbl_idxs"]
                inputs = batch["inputs"].to(eval_params.device)
                model_output = model(**inputs).logits
                scores = model_output.cpu().detach().float().numpy()
                for i in range(len(scores)):
                    inp_id = inp_ids[i]
                    ret_id = lbl_ids[i]
                    score = scores[i][0]
                    all_results.append((inp_id, ret_id, score))

        return all_results

    @classmethod
    def train(
        cls,
        train_dataset: IterableDataset,
        table_stores: Dict[str, Dataset],
        model_params: ModelParams,
        train_params: TrainParams,
    ):
        """
        Train the ranking model
        Args:
            train_dataset: The training dataset
            table_stores: The table stores for the input and label data
            model_params: The model parameters
            train_params: The training parameters
        """
        training_args = train_params.training_args
        # we need to have 'unused' columns to maintain information about
        # group and scores coming from the collator
        training_args.remove_unused_columns = False
        outer_model = cls.init_model(model_params, train_params)
        inner_model = outer_model.cross_encoder

        logger.info("Model loading...")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        else:
            # NOTE This is needed for the case where the program is run in a single process mode
            if training_args.bf16 and not torch.distributed.is_initialized():
                inner_model = inner_model.bfloat16()

        logger.info("=" * 50)
        logger.info(
            f"Memory used by model: {round(inner_model.get_memory_footprint() / 1024 / 1024 / 1024, 2)} GB"
        )

        trainer = RankLlamaTrainer(
            model=inner_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=partial(
                cls._collate_sharded,
                outer_model.tokenizer,
                model_params,
                train_params,
                table_stores,
            ),
            outer_model=outer_model,
        )

        # NOTE: in the huggingface trainers `_prepare_input` method, the inputs are converted from
        # mps device to cpu. To run on Apple Silicon, the method should be overridden. It is not
        # clear if training is supported for Apple Silicon devices.
        trainer.train()
        trainer.save_model()
