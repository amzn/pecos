import copy
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Any, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
import pecos

PARAM_FILENAME: str = "param.json"

logger = logging.getLogger(__name__)


class PairwisePointwiseHybridLoss(nn.Module):
    def __init__(self, pairwise_loss, pointwise_loss):
        super(PairwisePointwiseHybridLoss, self).__init__()
        self.pairwise_loss = pairwise_loss
        self.pointwise_loss = pointwise_loss

    def forward(self, preds, target, alpha=0.5):
        """
        Args:
            preds (torch.Tensor): prediction of shape (B, 2)
            target (torch.Tensor): gt target of shape (B, 2)
                target[:, 0] corresponds to relevance scores of positive labels
                target[:, 1] correspodns to relevance scores of negative labels
        """
        pairwise_target = torch.ones(preds.shape[0], device=preds.device).long()
        loss1 = self.pairwise_loss(preds[:, 0], preds[:, 1], pairwise_target)

        if self.pointwise_loss is not None:
            loss2 = self.pointwise_loss(preds.flatten(), target.flatten())
            return alpha * loss1 + (1.0 - alpha) * loss2
        else:
            return loss1


class ListwisePointwiseHybridLoss(nn.Module):
    def __init__(self, listwise_loss, pointwise_loss):
        super(ListwisePointwiseHybridLoss, self).__init__()
        self.listwise_loss = listwise_loss
        self.pointwise_loss = pointwise_loss

    def forward(self, preds, target, alpha=0.5):
        """
        Args:
            preds (torch.Tensor): prediction of shape (B, M)
            target (torch.Tensor): gt target of shape (B, M)
                target[:, 0]  corresponds to the relevance scores of positive labels
                target[:, 1:] corresponds to the relevance scores of negative labels
        """
        listwise_target = torch.zeros(preds.shape[0], device=preds.device).long()
        loss1 = self.listwise_loss(preds, listwise_target)

        if self.pointwise_loss is not None:
            loss2 = self.pointwise_loss(preds.flatten(), target.flatten())
            return alpha * loss1 + (1.0 - alpha) * loss2
        else:
            return loss1


LOSS_FN_DICT = {
    "pairwise": PairwisePointwiseHybridLoss(
        nn.MarginRankingLoss(reduction="mean", margin=0.1),
        nn.MSELoss(reduction="mean"),
    ),
    "listwise": ListwisePointwiseHybridLoss(
        nn.CrossEntropyLoss(reduction="mean"),
        nn.BCEWithLogitsLoss(reduction="mean"),
    ),
}


class LoggerCallback(TrainerCallback):
    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        train_dataloader,
        **kwargs,
    ):
        if isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader._IterableDataset_len_called = None
        else:
            pass

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        # avoid modifying the logs object as it is shared between callbacks
        logs = copy.deepcopy(logs)
        _ = logs.pop("total_flos", None)
        # round numbers so that it looks better in console
        if "loss" in logs:
            logs["loss"] = round(logs["loss"], 6)
        if "grad_norm" in logs:
            logs["grad_norm"] = round(logs["grad_norm"], 6)
        if "epoch" in logs:
            logs["epoch"] = round(logs["epoch"], 2)
        if state.is_world_process_zero:
            logger.info(logs)


class RankingTrainer(Trainer, pecos.BaseClass):
    """
    Trainer class for the pecos.xmr.reranker.RankingModel.
    """

    @dataclass
    class TrainingArgs(TrainingArguments, pecos.BaseParams):
        loss_fn: str = "listwise"
        loss_alpha: float = 1.0
        group_size: int = 8

        @classmethod
        def from_dict(cls, param=None):
            if param is None:
                return cls()
            elif isinstance(param, cls):
                return copy.deepcopy(param)
            elif isinstance(param, dict):
                parser = HfArgumentParser(cls)
                return parser.parse_dict(param, allow_extra_keys=True)[0]
            raise ValueError(f"{param} is not a valid parameter dictionary for {cls.name}")

        def to_dict(self, with_meta=True):
            d = super().to_dict()
            return self.append_meta(d) if with_meta else d

    def __init__(self, *args, **kwargs):
        param_to_save = kwargs.pop("param_to_save")
        super(RankingTrainer, self).__init__(*args, **kwargs)

        self.loss_fn = LOSS_FN_DICT[self.args.loss_fn]
        self.loss_alpha = self.args.loss_alpha
        self.param_to_save = param_to_save

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader. This function is called by the Trainer class.
        """
        prefetch_factor = self.args.dataloader_prefetch_factor
        prefetch_factor = prefetch_factor if prefetch_factor else 10
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            # Ensure that at least one worker is creating batches
            # parallel to the model compute
            num_workers=max(self.args.dataloader_num_workers, 1),
            # To ensure efficiency we prefetch samples in parallel
            prefetch_factor=prefetch_factor,
            collate_fn=self.data_collator,
        )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Save the model and tokenizer to the output directory. Makes sure the huggingface model is saved correctly.
        Args:
            output_dir: The output directory to save the model and tokenizer.
            state_dict: The state dictionary to save
        """
        # If we are executing this function, we are the process zero, so we don't check for that.
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving to {output_dir}")
        super()._save(output_dir, state_dict)

        # save the config
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        with open(os.path.join(output_dir, PARAM_FILENAME), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.param_to_save, indent=True))

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss for the model. This function is called by the Trainer class.
        Args:
            model: The model to compute the loss for
            inputs: The inputs to the model
            return_outputs: Whether to return the outputs
        """
        self.args: RankingTrainer.TrainingArgs
        group_size = self.args.group_size

        # ground truth target
        target = inputs["target"]
        target = target.view(-1, group_size)  # [B, M]
        batch_size = target.shape[0]

        # model prediction scores
        preds_1d = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            numeric_inputs=inputs.get("numeric_inputs", None),
            token_type_ids=inputs.get("token_type_ids", None),
        ).scores
        preds_2d = preds_1d.view(batch_size, -1)  # [B, M]
        assert preds_2d.shape == target.shape

        loss = self.loss_fn(preds_2d, target, alpha=self.loss_alpha)
        return (loss, preds_1d) if return_outputs else loss
