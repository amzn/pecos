#!/usr/bin/env python3 -u
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

import os
import torch
import logging

from pecos.xmc.xtransformer.matcher import TransformerMatcher

LOGGER = logging.getLogger(__name__)


class AllInOneForXMCModel(torch.nn.Module):
    """Wrapper class to pack transformer encoder and label embeddings

    Args:
        encoder (BertForXMC, RobertaForXMC, XLMRobertaForXMC, XLNetForXMC, DistilBertForXMC)
        label_embedding (TransformerLinearXMCHead)

    """

    def __init__(self, encoder, label_embedding):
        super().__init__()
        self.encoder = encoder
        self.label_embedding = label_embedding

    @classmethod
    def load(cls, load_dir):
        matcher = TransformerMatcher.load(load_dir)
        return cls(matcher.text_encoder, matcher.text_model)

    def save(self, save_dir):
        encoder_dir = os.path.join(save_dir, "text_encoder")
        os.makedirs(encoder_dir, exist_ok=True)
        # this creates config.json, pytorch_model.bin
        self.encoder.save_pretrained(encoder_dir)
        text_model_dir = os.path.join(save_dir, "text_model")
        torch.save(self.label_embedding, text_model_dir)

    @property
    def nr_labels(self):
        return self.label_embedding.num_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        instance_number=None,
        label_values=None,
        label_indices=None,
    ):
        pooled_output = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )["pooled_output"]

        W_act, b_act = self.label_embedding(
            output_indices=label_indices,
            num_device=len(self.device_ids) if hasattr(self, "device_ids") else 1,
        )
        logits = (pooled_output.unsqueeze(1) * W_act).sum(dim=-1) + b_act.squeeze(2)
        return logits

    def prepare_params(self, weight_decay=0):
        no_decay = ["bias", "LayerNorm.weight"]
        all_params = [
            {
                "params": [
                    p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return all_params


class DeepSpeedUtils(object):
    """Utility functions to use DeepSpeed"""

    @staticmethod
    def cli_launcher(module_name, hostfile=None, module_args={}):
        """Deepspeed launcher method

        Args:
            module_name (str): the python module to launch
            hostfile (str, optional): hostfile to launch distributed job.
                Default None to use local multi-gpu distribution.
            module_args (dict, optional): arguments and corresponding values to pass to the module.

        Example:
            module_args = {'arg_name_1': v1, 'arg_name_2': v2}
            Actual command
                deepspeed --hostfile [hostfile] --module [module_name] \
                        --arg-name-1 v1 --arg-name-2 v2
        """

        worker_cmd = [f"deepspeed"]
        if hostfile:
            worker_cmd += [f"--hostfile {hostfile}"]
        else:
            worker_cmd += [f"--num_gpus {torch.cuda.device_count()}"]

        worker_cmd += [f"--module {module_name}"]

        for k, v in module_args.items():
            worker_cmd += [f"--{k.replace('_', '-')} {v}"]
        worker_cmd = [" ".join(worker_cmd)]
        LOGGER.info(f"Actual command: {worker_cmd}")

        import subprocess

        subprocess.check_call(
            worker_cmd,
            shell=True,
            encoding="utf-8",
            universal_newlines=True,
            bufsize=1,
        )

    @staticmethod
    def get_config_from_params(train_params=None):
        """Construct DeepSpeed config from TransformerMatcher.TrainParams"""

        if train_params is None:
            train_params = TransformerMatcher.TrainParams()

        ds_config = {
            "fp16": {"enabled": train_params.fp16 if hasattr(train_params, "fp16") else False},
            "bf16": {"enabled": False},
            "zero_optimization": {
                "stage": 1,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": train_params.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": train_params.adam_epsilon,
                    "weight_decay": train_params.weight_decay,
                },
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "total_num_steps": train_params.max_steps,
                    "warmup_min_lr": 0,
                    "warmup_max_lr": train_params.learning_rate,
                    "warmup_num_steps": train_params.warmup_steps,
                    "warmup_type": "linear",
                },
            },
            "steps_per_print": train_params.logging_steps,
            "sparse_gradients": True,
            "gradient_clipping": train_params.max_grad_norm,
            "train_micro_batch_size_per_gpu": train_params.batch_size,
            "gradient_accumulation_steps": train_params.gradient_accumulation_steps,
            "wall_clock_breakdown": False,
            "dump_state": False,
        }
        return ds_config
