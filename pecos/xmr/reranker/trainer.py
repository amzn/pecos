import copy
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Any, Tuple, Dict

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, HfArgumentParser

import pecos

PARAM_FILENAME: str = "param.json"

logger = logging.getLogger(__name__)


class RankLlamaTrainer(Trainer, pecos.BaseClass):
    """
    Trainer class for the RankLlama model. This class extends the Trainer class.
    """

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    outer_model = None

    def __init__(self, *args, **kwargs):
        self.outer_model = kwargs.pop("outer_model")
        super(RankLlamaTrainer, self).__init__(*args, **kwargs)

    @dataclass
    class TrainingArgs(TrainingArguments, pecos.BaseParams):
        train_group_size: int = 8

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

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader. This function is called by the Trainer class.
        """
        prefetch_factor = self.args.dataloader_prefetch_factor
        prefetch_factor = prefetch_factor if prefetch_factor else 10
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
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

        outer_model: Any = self.outer_model
        super()._save(output_dir, state_dict)

        # save the config
        param = {
            "model": outer_model.__class__.__name__,
            "model_params": outer_model.model_params.to_dict(),
            "train_params": outer_model.train_params.to_dict(),
        }

        output_dir = output_dir if output_dir is not None else self.args.output_dir

        param = outer_model.append_meta(param)
        with open(os.path.join(output_dir, PARAM_FILENAME), "w", encoding="utf-8") as f:
            f.write(json.dumps(param, indent=True))

    def _prepare_inputs(self, inputs):
        """
        Prepare the inputs for the model. This function is called by the Trainer class. Converts the inputs to mps
        tensors if available.
        """
        super_inputs = super(RankLlamaTrainer, self)._prepare_inputs(inputs)
        if torch.backends.mps.is_available():
            super_inputs = {k: v.to("mps") for k, v in super_inputs.items()}
        return super_inputs

    def compute_loss(
        self, model, inputs: Dict[str, Any], return_outputs: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss for the model. This function is called by the Trainer class.
        Args:
            model: The model to compute the loss for
            inputs: The inputs to the model
            return_outputs: Whether to return the outputs
        """
        self.args: RankLlamaTrainer.TrainingArgs
        train_group_size = self.args.train_group_size
        if not train_group_size:
            raise NotImplementedError("Cannot perform ranking without train group")
        gt_scores = inputs["scores"].reshape(-1, train_group_size)
        ranker_logits = model(**inputs["input"], return_dict=True).logits
        batch_size = gt_scores.shape[0]

        grouped_logits = ranker_logits.view(batch_size, -1)
        assert grouped_logits.shape == gt_scores.shape
        loss = self.loss_fn(grouped_logits, gt_scores)

        return (loss, ranker_logits) if return_outputs else loss
