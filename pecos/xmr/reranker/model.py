import copy
import dataclasses as dc
import json
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Tuple, Any, Optional, Union

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

import peft
from datasets import IterableDataset, Dataset
from peft import get_peft_model
from peft.config import PeftConfig
from peft.mixed_model import PeftMixedModel
from peft.peft_model import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModel,
    CONFIG_MAPPING,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.utils import ModelOutput

import pecos
from pecos.xmr.reranker.trainer import (
    RankingTrainer,
    PARAM_FILENAME,
)
from pecos.xmr.reranker.data_utils import RankingDataUtils


logger = logging.getLogger(__name__)


ACT_FCT_DICT = {
    "identity": nn.Identity,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}


@dc.dataclass
class RerankerOutput(ModelOutput):
    text_emb: Optional[Tensor] = None
    numr_emb: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class NumrMLPEncoderConfig(PretrainedConfig):

    model_type = "numr_mlp_encoder"

    def __init__(
        self,
        inp_feat_dim: int = 1,
        inp_dropout_prob: float = 0.1,
        hid_dropout_prob: float = 0.1,
        hid_actv_type: str = "relu6",
        hid_size_list: list = [64, 128, 256],
        **kwargs,
    ):
        self.inp_feat_dim = inp_feat_dim
        self.hid_size_list = hid_size_list
        self.hid_actv_type = hid_actv_type
        self.inp_dropout_prob = inp_dropout_prob
        self.hid_dropout_prob = hid_dropout_prob
        super().__init__(**kwargs)


class TextNumrEncoderConfig(PretrainedConfig):
    """
    The configuration class for the cross encoder model. This class contains the model shortcut, model modifier and
    model initialization arguments for the model. The model shortcut is the name of the huggingface model. The
    `model_modifier` is the configuration of the modifier (e.g. PEFT) and the `model_init_kwargs` are the arguments
    for the model.
    """

    default_text_model_type = "xlm-roberta"
    model_type = "text_numr_crossencoder"

    def __init__(
        self,
        text_config=None,
        numr_config=None,
        text_pooling_type="cls",
        head_actv_type="relu6",
        head_dropout_prob=0.1,
        head_size_list=[128, 64],
        **kwargs,
    ):
        """
        Initialize the cross encoder configuration
        """

        if text_config is None:
            pass
        elif isinstance(text_config, PretrainedConfig):
            text_config = copy.deepcopy(text_config)
        elif isinstance(text_config, dict):
            text_model_type = text_config.get("model_type", self.default_text_model_type)
            text_config = CONFIG_MAPPING[text_model_type](**text_config)
        else:
            raise TypeError(f"Type(text_config) is not valid, got {type(text_config)}!")
        self.text_config = text_config
        self.text_pooling_type = text_pooling_type

        if numr_config is None:
            pass
        elif isinstance(numr_config, PretrainedConfig):
            numr_config = copy.deepcopy(numr_config)
        elif isinstance(numr_config, dict):
            numr_config = NumrMLPEncoderConfig(**numr_config)
        else:
            raise TypeError(f"Type(numr_config) is not valid, got {type(numr_config)}!")
        self.numr_config = numr_config

        self.head_size_list = head_size_list
        self.head_actv_type = head_actv_type
        self.head_dropout_prob = head_dropout_prob

        super().__init__(**kwargs)


class MLPBlock(nn.Module):
    def __init__(self, inp_size, dropout_prob, actv_type, hid_size_list):
        super(MLPBlock, self).__init__()

        cur_inp_size = inp_size
        self.mlp_layers = nn.ModuleList()
        for cur_hid_size in hid_size_list:
            self.mlp_layers.append(nn.Linear(cur_inp_size, cur_hid_size, bias=True))
            self.mlp_layers.append(ACT_FCT_DICT[actv_type]())
            self.mlp_layers.append(nn.Dropout(dropout_prob))
            cur_inp_size = cur_hid_size

    def forward(self, x):
        for cur_layer in self.mlp_layers:
            x = cur_layer(x)
        return x


class NumrMLPEncoder(PreTrainedModel):

    config_class = NumrMLPEncoderConfig

    def __init__(self, config: NumrMLPEncoderConfig):
        super().__init__(config)

        self.inp_dropout = nn.Dropout(config.inp_dropout_prob)
        self.mlp_block = MLPBlock(
            config.inp_feat_dim,
            config.hid_dropout_prob,
            config.hid_actv_type,
            config.hid_size_list,
        )
        self.layer_norm = nn.LayerNorm(config.hid_size_list[-1])

    def forward(self, numeric_inputs):
        numr_emb = self.inp_dropout(numeric_inputs)
        numr_emb = self.mlp_block(numr_emb)
        return self.layer_norm(numr_emb)


class TextNumrEncoder(PreTrainedModel):

    config_class = TextNumrEncoderConfig

    def __init__(self, config: TextNumrEncoderConfig):
        """
        Initialize the cross encoder model
        Args:
            config: The configuration for the cross encoder
        """

        # sanity check
        if config.text_pooling_type not in ["cls", "avg", "last"]:
            raise NotImplementedError(
                f"text_pooling_type={config.text_pooling_type} is not support!"
            )
        if config.text_config is None and config.numr_config is None:
            raise ValueError(f"text_config and numr_config can not be None at the same time!")
        super().__init__(config)

        # text encoder
        if config.text_config:
            text_encoder = AutoModel.from_pretrained(
                config.text_config._name_or_path,
                attn_implementation=config.text_config._attn_implementation,
                trust_remote_code=config.text_config.trust_remote_code,
                token=getattr(config.text_config, "token", None),
            )
            text_encoder.config.pad_token_id = (
                0 if text_encoder.config.pad_token_id is None else text_encoder.config.pad_token_id
            )
            self.text_encoder = text_encoder
            self.text_emb_dim = self.text_encoder.config.hidden_size
            self.text_pooling_type = config.text_pooling_type
        else:
            self.text_encoder = None  # type: ignore
            self.text_emb_dim = 0

        # numeric encoder
        if config.numr_config:
            self.numr_encoder = NumrMLPEncoder(config.numr_config)
            self.numr_emb_dim = self.numr_encoder.config.hid_size_list[-1]
        else:
            self.numr_encoder = None  # type: ignore
            self.numr_emb_dim = 0

        # head layer
        cur_feat_dim = self.text_emb_dim + self.numr_emb_dim
        self.head_layers = MLPBlock(
            cur_feat_dim,
            config.head_dropout_prob,
            config.head_actv_type,
            config.head_size_list,
        )
        self.scorer = nn.Linear(config.head_size_list[-1], 1, bias=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        numeric_inputs=None,
    ):
        """
        Returns the forward output of the huggingface model
        """

        # get text embedding from HF Pretrained Transformers encoder
        text_emb = None
        if self.text_encoder:
            text_input_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids:
                text_input_dict["token_type_ids"] = token_type_ids
            text_outputs = self.text_encoder(**text_input_dict, return_dict=True)
            if hasattr(text_outputs, "pooler_output"):
                text_emb = text_outputs.pooler_output
            else:
                text_emb = self.text_pooler(text_outputs.last_hidden_state, attention_mask)

        # get numr embedding from Numerical MLP Encoder
        numr_emb = None
        if self.numr_encoder:
            numr_emb = self.numr_encoder(numeric_inputs)

        # head layer + scorer
        if self.text_encoder and self.numr_encoder:
            head_emb = torch.cat((text_emb, numr_emb), 1)
        elif self.text_encoder is not None:
            head_emb = text_emb
        elif self.numr_encoder is not None:
            head_emb = numr_emb
        head_emb = self.head_layers(head_emb)
        scores = self.scorer(head_emb)

        return RerankerOutput(
            text_emb=text_emb,
            numr_emb=numr_emb,
            scores=scores,
        )

    def text_pooler(self, last_hidden_states, attention_mask):
        if self.text_pooling_type == "cls":
            text_emb = last_hidden_states[:, 0, :]
        elif self.text_pooling_type == "avg":
            # https://huggingface.co/intfloat/multilingual-e5-base
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            text_emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.text_pooling_type == "last":
            # https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                text_emb = last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                text_emb = last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
                ]
        return text_emb

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the model
        """

        self.text_encoder.gradient_checkpointing_enable(**kwargs)


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
            hf_trainer_args (RankingTrainer.TrainingArgs): The training arguments for the model
            target_data_folder (str): The path to the target data folder
            input_data_folder (str): The path to the input data folder
            label_data_folder (str): The path to the label data folder
        """

        hf_trainer_args: RankingTrainer.TrainingArgs
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

        encoder_config: TextNumrEncoderConfig = None  # type: ignore
        model_modifier: dict = dc.field(default_factory=lambda: dict())

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

        target_data_folder: str
        input_data_folder: str
        label_data_folder: str
        model_path: str
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
        inp_id_orig_col: Optional[str] = None
        lbl_id_orig_col: Optional[str] = None
        keyword_col_name: str = "keywords"
        content_col_names: List[str] = field(default_factory=lambda: ["title", "contents"])
        content_sep: str = " "
        append_eos_token: bool = False
        pad_to_multiple_of: int = 16
        bf16: bool = True

    def __init__(
        self,
        encoder: Union[PreTrainedModel, PeftModel, PeftMixedModel],
        tokenizer: AutoTokenizer,
        model_params: ModelParams,
        train_params: Optional[TrainParams] = None,
    ):
        """
        Initialize the ranking model. The model contains the encoder, tokenizer, model parameters and training parameters.
        Args:
            encoder (Union[PreTrainedModel, PeftModel, PeftMixedModel]): The encoder model
            tokenizer (AutoTokenizer): The tokenizer for the model
            model_params (RankingModel.ModelParams): The model parameters
            train_params (Optional[RankingModel.TrainParams]): The training parameters
        """
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.model_params = self.ModelParams.from_dict(model_params)
        self.train_params = self.TrainParams.from_dict(train_params) if train_params else None

    @classmethod
    def get_modified_model(cls, model: PreTrainedModel, config: Dict):
        """
        Takes a pretrained Huggingface model and modifies it to include new features. Currently, the `modifier_type`
        supported by this method is limited to the `peft` package.

        Args:
            model (PreTrainedModel): A PreTrainedModel from the transformers package.
            config (Dict): A dictionary containing the configuration for the model modifier.
        Returns: The modified model
        """
        if config["modifier_type"] == "peft":
            config_cls = getattr(peft, config["config_type"])
            peft_config: PeftConfig = config_cls(**config["config"])
            modified_model = get_peft_model(model, peft_config)
            return modified_model
        else:
            raise NotImplementedError(f"We only support modifier_type==peft for now!")

    @classmethod
    def init_model(cls, model_params: ModelParams, train_params: TrainParams):
        """Initiate a model with training parameters

        Args:
            model_params (RankingModel.ModelParams): the model parameters
            train_params (RankingModel.TrainParams): the training parameters
        Returns:
            An instance of RankingModel
        """
        hf_trainer_args = train_params.hf_trainer_args
        if hf_trainer_args.local_rank > 0:
            torch.distributed.barrier()

        encoder_config = TextNumrEncoderConfig(**model_params.encoder_config)
        encoder = TextNumrEncoder(encoder_config)

        if hf_trainer_args.bf16:
            encoder = encoder.bfloat16()

        if model_params.model_modifier:
            if hf_trainer_args.gradient_checkpointing:
                encoder.text_encoder.enable_input_require_grads()
            encoder = cls.get_modified_model(
                model=encoder,
                config=model_params.model_modifier,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            encoder_config.text_config._name_or_path,
            trust_remote_code=encoder_config.text_config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.padding_side = "right"

        if hf_trainer_args.local_rank == 0:
            torch.distributed.barrier()

        return cls(encoder, tokenizer, model_params, train_params=train_params)

    @classmethod
    def load(cls, load_dir):
        """Load the model from the folder load_dir

        Args:
            load_dir (str): path of the loading folder
        """

        param_file = os.path.join(load_dir, PARAM_FILENAME)
        if not os.path.exists(param_file):
            raise FileNotFoundError(f"The model {load_dir} does not exists.")

        param = json.loads(open(param_file, "r").read())
        model_params = cls.ModelParams.from_dict(param.get("model_params", None))

        if model_params.model_modifier:
            encoder_config = TextNumrEncoderConfig(**model_params.encoder_config)
            encoder = TextNumrEncoder(encoder_config)
            encoder = PeftModel.from_pretrained(encoder, load_dir)
            encoder = encoder.merge_and_unload()
        else:
            encoder = TextNumrEncoder.from_pretrained(load_dir)

        tokenizer = AutoTokenizer.from_pretrained(load_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.padding_side = "right"

        return cls(encoder, tokenizer, model_params)

    def get_param_to_save(self):
        param = {
            "model": self.__class__.__name__,
            "model_params": self.model_params.to_dict(),
            "train_params": self.train_params.to_dict(),
        }
        param = self.append_meta(param)
        return param

    def save(self, save_dir):
        """Save the model to the folder save_dir

        Args:
            save_dir (str): path of the saving folder
        """

        os.makedirs(save_dir, exist_ok=True)
        param_file = os.path.join(save_dir, PARAM_FILENAME)

        param_to_save = self.get_param_to_save()
        with open(param_file, "w", encoding="utf-8") as fout:
            fout.write(json.dumps(param_to_save, indent=True))

        self.encoder.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

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
                    train_params.hf_trainer_args.group_size,
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

        ret_dict = dict(target=m_scores)
        ret_dict.update(pairs_collated)
        return ret_dict

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
        inp_id_orig_col = (
            eval_params.inp_id_orig_col if eval_params.inp_id_orig_col else eval_params.inp_id_col
        )
        lbl_id_orig_col = (
            eval_params.lbl_id_orig_col if eval_params.lbl_id_orig_col else eval_params.lbl_id_col
        )
        for s in data:
            inp_id = s[eval_params.inp_id_col]
            retr_id = s[eval_params.lbl_id_col]
            inp_idxs.append(table_stores["input"][inp_id][inp_id_orig_col])
            lbl_idxs.append(table_stores["label"][retr_id][lbl_id_orig_col])

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

        ret_dict = dict(inp_idxs=inp_idxs, lbl_idxs=lbl_idxs)
        ret_dict.update(pairs_collated)
        return ret_dict

    @classmethod
    def predict(
        cls,
        eval_dataset: IterableDataset,
        table_stores: Dict[str, Dataset],
        eval_params: EvalParams,
        encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Use pytorch device: {}".format(device))
        device = torch.device(device)  # type: ignore
        if torch.cuda.device_count() > 1 and not isinstance(encoder, torch.nn.DataParallel):
            encoder = torch.nn.DataParallel(encoder)
        encoder = encoder.to(device)

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
            shuffle=False,
        )

        encoder.eval()
        all_results = []
        with torch.inference_mode():
            for batch in eval_dataloader:
                inp_ids = batch["inp_idxs"]
                lbl_ids = batch["lbl_idxs"]

                # place inputs to the device
                for k in batch.keys():
                    if torch.is_tensor(batch[k]):
                        batch[k] = batch[k].to(device)

                # forward
                output = encoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    numeric_inputs=batch.get("numeric_inputs", None),
                    token_type_ids=batch.get("token_type_ids", None),
                ).scores
                scores = output.cpu().detach().float().numpy()
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
        hf_trainer_args = train_params.hf_trainer_args
        # we need to have 'unused' columns to maintain information about
        # group and scores coming from the collator
        hf_trainer_args.remove_unused_columns = False
        model = cls.init_model(model_params, train_params)
        param_to_save = model.get_param_to_save()

        trainer = RankingTrainer(
            model=model.encoder,
            args=hf_trainer_args,
            tokenizer=model.tokenizer,
            train_dataset=train_dataset,
            data_collator=partial(
                cls._collate_sharded,
                model.tokenizer,
                model_params,
                train_params,
                table_stores,
            ),
            param_to_save=param_to_save,
        )

        trainer.train(
            resume_from_checkpoint=hf_trainer_args.resume_from_checkpoint,
        )
        return model
