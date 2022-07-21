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
import logging
import os
import json
import numpy as np
import torch
import scipy.sparse as smat
from pecos.utils import smat_util
from torch.utils.data import Dataset
from transformers import BatchEncoding

LOGGER = logging.getLogger(__name__)


class MLProblemWithText(object):
    """Object that defines a ML-Problem with text input.
    Containing the input text and X, Y, C, M, M_pred matrices.

        X_text (list of str or dict of tensors): instance text, len(text) = nr_inst
                or dictionary of tokenized text (dict of torch.tensor)
        Y (csr_matrix): training labels, shape = (nr_inst, nr_labels)
        X_feat (csr_matrix or ndarray): instance numerical features, shape = (nr_inst, nr_features)
        C (csc_matrix, optional): clustering matrix, shape = (nr_labels, nr_codes)
        M (csr_matrix, optional): matching matrix, shape = (nr_inst, nr_codes)
                model will be trained only on its non-zero indices
                its values will not be used.
    """

    def __init__(self, X_text, Y, X_feat=None, C=None, M=None):
        self.X_text = X_text

        self.Y = Y
        self.X_feat = X_feat
        self.C = C
        self.M = M

        self.type_check()

    @property
    def is_tokenized(self):
        return isinstance(self.X_text, (dict, BatchEncoding))

    def type_check(self):
        if self.X_feat is not None and not isinstance(self.X_feat, (smat.csr_matrix, np.ndarray)):
            raise TypeError(f"Expect X to be csr_matrix or ndarray, got {type(self.X)}")
        if not isinstance(self.Y, smat.csr_matrix):
            raise TypeError(f"Expect Y to be csr_matrix, got {type(self.Y)}")
        if self.C is not None and not isinstance(self.C, smat.csc_matrix):
            raise TypeError(f"Expect C to be csc_matrix, got {type(self.C)}")
        if self.M is not None and not isinstance(self.M, smat.csr_matrix):
            raise TypeError(f"Expect M to be csr_matrix, got {type(self.M)}")

    @property
    def nr_labels(self):
        return self.Y.shape[1]

    @property
    def nr_features(self):
        return None if self.X_feat is None else self.X_feat.shape[1]

    @property
    def nr_codes(self):
        return None if self.C is None else self.C.shape[1]

    @property
    def nr_inst(self):
        return self.Y.shape[0]


class XMCTextTensorizer(object):

    DEFAULT_FEATURE_KEYS = ["input_ids", "attention_mask", "token_type_ids", "instance_number"]

    def __init__(self, text, feature_keys=None, input_transform=None):
        self.text = text
        self.feature_keys = feature_keys
        self.input_transform = input_transform

        if self.feature_keys is None:
            self.feature_keys = self.DEFAULT_FEATURE_KEYS

        if self.is_tokenized:
            for k in self.feature_keys:
                if k not in self.text:
                    raise KeyError(f"Missing key ({k}) from tokenized inputs")
        else:
            if input_transform is None:
                raise ValueError(f"Expect tokenizer if raw text given to XMCTextTensorizer")

        LOGGER.info(
            f"Constructed XMCTextTensorizer, tokenized={self.is_tokenized}, len={len(self)}"
        )

    def get_shard(self, start, end):
        if end <= start:
            raise ValueError(f"end >= start: {end} <= {start}")
        if self.is_tokenized:
            text_shard = {k: self.text[k][start:end] for k in self.feature_keys if k in self.text}
        else:
            text_shard = self.text[start:end]
        return XMCTextTensorizer(
            text_shard,
            feature_keys=self.feature_keys,
            input_transform=self.input_transform,
        )

    def __len__(self):
        if self.is_tokenized:
            return self.text[self.feature_keys[0]].shape[0]
        else:
            return len(self.text)

    @property
    def is_tokenized(self):
        return isinstance(self.text, (dict, BatchEncoding))

    def __getitem__(self, i):
        if self.is_tokenized:
            ret = {k: self.text[k][i] for k in self.feature_keys if k in self.text}
            return tuple(self.text[k][i] for k in self.feature_keys)
        else:
            ret = self.input_transform(self.text[i])
        ret["instance_number"] = torch.IntTensor([i])
        return tuple(ret[kk].squeeze(dim=0) for kk in self.feature_keys)


class XMCLabelTensorizer(object):
    """
    Args:
        Y (csr_matrix, optional): training labels, shape = (nr_inst, nr_labels)
        M (csr_matrix, optional): matching matrix, shape = (nr_inst, nr_codes)
            model will be trained only on its non-zero indices
            its values will not be used.
        label_padding_idx (int, optional): the index used to pad all label_indices
            to the same length. Default -1
        max_labels (int, optional): max number of labels considered for each
            instance, will subsample from existing label indices if need to.
            Default None to ignore.
        pre_compute (bool, optional): whether to pre-generate label tensors for the dataset.
            Default False

    Return values depend on the Y and M:
        1. Both Y and M are not None (train on middle layer):
            data[i] = (label_values[i], label_indices[i])
        2. Both Y and M are None (inference on top layer):
            data[i] = (,)
        2. Y is not None, M is None (train on top layer):
            data[i] = (label_values[i],)
        3. Y is None, M is not None (inference on middle layer):
            data[i] = (label_indices[i],)
    """

    def __init__(
        self,
        Y=None,
        M=None,
        label_padding_idx=-1,
        max_labels=None,
        pre_compute=False,
    ):

        self.label_padding_idx = label_padding_idx
        self.has_label = Y is not None
        self.has_ns = M is not None
        self.pre_compute = pre_compute

        self.label_width = None
        self.offset = 0

        if pre_compute:
            # pre-computed will use these
            self.get_lbl_tensors(M, Y, max_labels=max_labels)
        else:
            # realtime compute will use these
            self.get_lbl_mat(M, Y, max_labels=max_labels)

        LOGGER.debug(
            f"Constructed XMCLabelTensorizer, pre_compute={self.pre_compute}, len={len(self)}, num_active_labels={self.num_active_labels}"
        )

    def get_shard(self, start, end):
        if end <= start:
            raise ValueError(f"end <= start: {end} <= {start}")

        ret = XMCLabelTensorizer(
            label_padding_idx=self.label_padding_idx,
            pre_compute=self.pre_compute,
        )
        ret.has_label = self.has_label
        ret.has_ns = self.has_ns
        ret.label_width = self.label_width
        ret.offset = self.offset

        if self.pre_compute:
            ret.label_indices = (
                None if self.label_indices is None else self.label_indices[start:end, :]
            )
            ret.label_values = (
                None if self.label_values is None else self.label_values[start:end, :]
            )
        elif self.lbl_mat is not None:
            ret.lbl_mat = self.lbl_mat[start:end, :]
        else:
            ret.lbl_mat = None

        return ret

    def __len__(self):
        if not self.has_ns and not self.has_label:
            return 0
        if self.pre_compute:
            if self.has_ns:
                return self.label_indices.shape[0]
            else:
                return self.label_values.shape[0]
        else:
            return self.lbl_mat.shape[0]

    @property
    def num_active_labels(self):
        return self.label_width

    def get_lbl_mat(self, M, Y, max_labels=None):

        if M is None and Y is None:
            # 1.inference at top layer
            self.label_width = 0
            self.lbl_mat = None
        elif M is not None and Y is None:
            # 2.inference at intermediate layer
            self.label_width = max(M.indptr[1:] - M.indptr[:-1])
            self.lbl_mat = smat_util.binarized(M)
        elif M is None and Y is not None:
            # 3.train at top layer
            self.label_width = Y.shape[1]
            self.lbl_mat = Y.astype(np.float32)
        elif M is not None and Y is not None:
            # 4.train at intermediate layer
            self.lbl_mat = smat_util.binarized(M) + smat_util.binarized(Y)
            self.label_width = max(self.lbl_mat.indptr[1:] - self.lbl_mat.indptr[:-1])
            # put values in M, positive labels equal to y + offset, negative to offset
            # offset is used to avoid elimination of zero entrees
            self.offset = Y.data.max() + 10.0
            self.lbl_mat.data[:] = self.offset
            self.lbl_mat += Y

        if self.label_width is not None and max_labels is not None:
            if self.label_width > max_labels:
                LOGGER.warning(f"will need to sub-sample from {self.label_width} to {max_labels}")
                self.label_width = max_labels

        if Y is not None:
            label_lower_bound = max(Y.indptr[1:] - Y.indptr[:-1])
            if label_lower_bound > self.label_width:
                LOGGER.warning(
                    f"label-width ({self.label_width}) is not able to cover all positive labels ({label_lower_bound})!"
                )

    def get_lbl_tensors(self, M, Y, max_labels=None):
        if M is None and Y is None:
            self.label_indices = None
            self.label_values = None
            self.label_width = 0
        elif M is None and Y is not None:
            # if M is None, taking all labels into account
            self.label_indices = None
            self.label_values = torch.FloatTensor(Y.toarray())
            self.label_width = Y.shape[1]
        else:
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
            label_indices = np.zeros((nr_inst, max_labels), dtype=np.int64) + self.label_padding_idx
            if Y is not None:
                label_values = np.zeros((nr_inst, max_labels), dtype=np.float32)

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

            self.label_indices = torch.IntTensor(label_indices)
            self.label_values = None if Y is None else torch.FloatTensor(label_values)
            self.label_width = max_labels

    def __getitem__(self, i):
        if self.pre_compute:
            if self.label_values is not None and self.label_indices is not None:
                return (self.label_values[i], self.label_indices[i])
            elif self.label_indices is not None:
                return (self.label_indices[i],)
            elif self.label_values is not None:
                return (self.label_values[i],)
            else:
                return tuple()

        else:
            if not self.has_ns:
                if not self.has_label:
                    return tuple()
                else:
                    return (torch.FloatTensor(self.lbl_mat[i].toarray()).squeeze(dim=0),)
            else:
                nr_active = self.lbl_mat.indptr[i + 1] - self.lbl_mat.indptr[i]
                rng = slice(self.lbl_mat.indptr[i], self.lbl_mat.indptr[i + 1])

                if nr_active > self.label_width:
                    # sub-sample to fit in self.label_width
                    nr_active = self.label_width
                    rng = np.random.choice(
                        np.arange(self.lbl_mat.indptr[i], self.lbl_mat.indptr[i + 1]),
                        nr_active,
                        replace=False,
                    )

                label_indices = (
                    torch.zeros((self.label_width,), dtype=torch.int) + self.label_padding_idx
                )
                label_indices[:nr_active] = torch.from_numpy(self.lbl_mat.indices[rng])

                if not self.has_label:
                    return (label_indices,)
                else:
                    label_values = torch.zeros((self.label_width,), dtype=torch.float32)
                    label_values[:nr_active] = torch.from_numpy(
                        self.lbl_mat.data[rng] - self.offset
                    )
                    return (label_values, label_indices)


class XMCTextDataset(Dataset):
    """Dataset to hold text and label/matching matrices for XMC training and prediction.
        Conduct real-time tokenization of input text and label tensor generation to save memory.

    Args:
        text (list of str): input text, length = nr_inst
        input_transform (function): the transform function to process/tokenize text
        feature_keys (list of str): the feature keys in order for batch generation.
        Y (csr_matrix, optional): training labels, shape = (nr_inst, nr_labels)
        M (csr_matrix, optional): matching matrix, shape = (nr_inst, nr_codes)
            model will be trained only on its non-zero indices
            its values will not be used.
        label_padding_idx (int, optional): the index used to pad all label_indices
            to the same length. Default -1
        max_labels (int, optional): max number of labels considered for each
            instance, will subsample from existing label indices if need to.
            Default None to ignore.


    Return values depend on the Y and M:
        1. Both Y and M are not None (train on middle layer):
            data[i] = (feature[0][i], feature[1][i], ..., label_values[i], label_indices[i])
        2. Both Y and M are None (inference on top layer):
            data[i] = (feature[0][i], feature[1][i], ...)
        2. Y is not None, M is None (train on top layer):
            data[i] = (feature[0][i], feature[1][i], ..., label_values[i])
        3. Y is None, M is not None (inference on middle layer):
            data[i] = (feature[0][i], feature[1][i], ..., label_indices[i])
    """

    def __init__(
        self,
        input_tensorizer,
        output_tensorizer=None,
    ):
        if output_tensorizer is None:
            output_tensorizer = XMCLabelTensorizer()

        if len(output_tensorizer) > 0:
            if len(input_tensorizer) != len(output_tensorizer):
                raise ValueError(
                    f"Dimension 0 mismatch: {len(input_tensorizer)} != {len(output_tensorizer)}"
                )

        self.input_tensorizer = input_tensorizer
        self.output_tensorizer = output_tensorizer

    def __len__(self):
        return len(self.input_tensorizer)

    def get_shard(self, start, end):
        return self.__class__(
            self.input_tensorizer.get_shard(start, end),
            self.output_tensorizer.get_shard(start, end),
        )

    def save(self, save_dir, num_shards=None, init_shard_idx=0):
        if num_shards is None:
            num_shards = 1

        os.makedirs(save_dir, exist_ok=True)
        param = {
            "model": self.__class__.__name__,
            "num_shards": num_shards,
            "num_instances": len(self),
        }
        with open(f"{save_dir}/config.json", "w") as f:
            f.write(json.dumps(param, indent=True))

        chunk_size = (len(self) + num_shards - 1) // num_shards
        for sid in range(init_shard_idx, init_shard_idx + num_shards):
            cur_chunk_dir = f"{save_dir}/{sid}"
            start = chunk_size * sid
            end = min(chunk_size * (sid + 1), len(self))
            torch.save(self.get_shard(start, end), cur_chunk_dir, pickle_protocol=4)
            LOGGER.info(f"Shard{sid} saved to {cur_chunk_dir}, len={end - start}")

    @classmethod
    def get_data_stats(cls, load_dir):
        with open(f"{load_dir}/config.json", "r") as f:
            config = json.load(f)
        return config

    @classmethod
    def load(cls, load_dir, shard=0):
        nr_shards = cls.get_data_stats(load_dir)["num_shards"]
        if shard >= nr_shards:
            raise ValueError(f"Loading shard#{shard} where there are only {nr_shards} available")
        return torch.load(f"{load_dir}/{shard}")

    @property
    def has_ns(self):
        return self.output_tensorizer.has_ns

    @property
    def num_active_labels(self):
        return self.output_tensorizer.num_active_labels

    def __getitem__(self, i):
        return self.input_tensorizer[i] + self.output_tensorizer[i]
