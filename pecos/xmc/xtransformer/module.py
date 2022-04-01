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
import numpy as np
import torch
import scipy.sparse as smat
from pecos.utils import smat_util
from torch.utils.data import Dataset, TensorDataset
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


class XMCTensorDataset(Dataset):
    """Dataset to hold feature and label tensors for XMC training and prediction.

    Args:
        *features (tensors): feature tensors, required to have same first
            dimension: nr_inst
        label_values (tensor or None): label values with shape = (nr_inst, num_active_labels)
        label_indices (tensor or None): label indices with shape = (nr_inst, num_active_labels)

    Return values depend on the label_values and label_indices:
        if label_values is None and label_indices is not None:
            data[i] = (feature[0][i], feature[1][i], ..., label_values[i], label_indices[i])
        elif label_values is not None:
            data[i] = (feature[0][i], feature[1][i], ..., label_values[i])
        elif label_indices is not None:
            data[i] = (feature[0][i], feature[1][i], ..., label_indices[i])
        else:
            data[i] = (feature[0][i], feature[1][i], ...)
    """

    def __init__(self, *features, label_values=None, label_indices=None):
        self.nr_inst = features[0].size(0)
        self.data = TensorDataset(*features)
        if label_values is not None and label_values.size(0) != self.nr_inst:
            raise ValueError("First dimension mismatch between features and label_values")
        if label_indices is not None and label_indices.size(0) != self.nr_inst:
            raise ValueError("First dimension mismatch between features and label_indices")

        self.label_values = label_values
        self.label_indices = label_indices

    @property
    def num_active_labels(self):
        if self.label_indices is None:
            return None
        else:
            return self.label_indices.shape[1]

    def __getitem__(self, index):
        if self.label_values is not None and self.label_indices is not None:
            return self.data[index] + (self.label_values[index], self.label_indices[index])
        elif self.label_indices is not None:
            return self.data[index] + (self.label_indices[index],)
        elif self.label_values is not None:
            return self.data[index] + (self.label_values[index],)
        else:
            return self.data[index]

    def __len__(self):
        return self.nr_inst

    def refresh_labels(self, label_values=None, label_indices=None):
        """Refresh label-values and label-indices from given tensors"""
        self.label_values = label_values
        self.label_indices = label_indices


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
        idx_padding (int, optional): the index used to pad all label_indices
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
        text,
        input_transform,
        feature_keys,
        Y=None,
        M=None,
        idx_padding=-1,
        max_labels=None,
    ):
        self.text = text
        self.input_transform = input_transform
        self.feature_keys = feature_keys
        self.idx_padding = idx_padding

        self.lbl_mat = None
        self.has_label = Y is not None
        self.has_ns = M is not None

        self.offset = 0

        if M is None and Y is None:
            # 1.inference at top layer
            self.label_width = None
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

    def __len__(self):
        return len(self.text)

    @property
    def num_active_labels(self):
        return self.label_width

    def get_input_tensors(self, i):
        ret = self.input_transform(self.text[i])
        ret["instance_number"] = torch.IntTensor([i])
        return tuple(ret[kk].squeeze(dim=0) for kk in self.feature_keys)

    def get_output_tensors(self, i):
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

            label_indices = torch.zeros((self.label_width,), dtype=torch.int) + self.idx_padding
            label_indices[:nr_active] = torch.from_numpy(self.lbl_mat.indices[rng])

            if not self.has_label:
                return (label_indices,)
            else:
                label_values = torch.zeros((self.label_width,), dtype=torch.float32)
                label_values[:nr_active] = torch.from_numpy(self.lbl_mat.data[rng] - self.offset)
                return (label_values, label_indices)

    def __getitem__(self, index):
        return self.get_input_tensors(index) + self.get_output_tensors(index)
