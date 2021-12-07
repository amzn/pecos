#!/usr/bin/env python
import argparse
import glob
import os
import pickle as pkl
import sys
from collections import defaultdict

import numpy as np
import scipy as sp
import scipy.sparse as sps


def combine_Y(mapper, new_nr_labels, y):
    """Create the new ground-truth label matrix given the label mapper.

    Args:
        mapper: the mapper returned from `combine_from_cluster` function.
        new_nr_labels: the number of new labels returned from `combined_from_cluster` function.
        y: the old label matrix of shape nr_instance x nr_labels.

    Returns:
        the new label matrix of shape nr_instance x new_nr_labels.
    """
    new_y_cols = []
    new_y_rows = []
    y = y.tocsr()
    for inst_id in range(y.shape[0]):
        orig_labels = y.indices[y.indptr[inst_id] : y.indptr[inst_id + 1]]
        for orig_label in orig_labels:
            new_label = mapper[orig_label]
            new_y_cols.append(new_label)
            new_y_rows.append(inst_id)
    new_y = sps.coo_matrix(
        (np.ones_like(new_y_cols), (new_y_rows, new_y_cols)),
        shape=(y.shape[0], new_nr_labels),
    )
    return new_y.tocsr()


def cluster_chain(path, level_from_bottom=0):
    """Obtain the clustering matrix given the model checkpoint.

    Args:
        path: path to the model checkpoint.
        level_from_bottom: the level of label tree starting from the bottom.

    Returns:
        the clustering matrix `C` to be used for `combine_from_cluster` function.
    """
    Cs = []
    for cf in sorted(glob.glob(path)):
        C = sps.load_npz(cf)
        Cs.append(C)
    C_from_bottom = [Cs[-1]]
    for C in Cs[::-1][1:]:
        C_from_bottom.append(C_from_bottom[-1].dot(C))
    return C_from_bottom[level_from_bottom]


def combine_from_cluster(C, bin_size):
    """Combine the labels in the same cluster.

    Given the clustering matrix `C`, we randomly group `bin_size` number of labels as a synthetic composite label,
    which is further saved to our synthetic dataset.

    Args:
        C: A scipy sparse matrix.
        bin_size: An integer specifying how many labels to group into a new fake label.

    Returns:
        A dictionary `mapper` that maps the id of original label to the id of new label.
        As well as `new_label_count` meaning the number of labels in the new synthetic dataset.
    """
    C = C.tocsc()
    new_label_count = 0
    mapper = {}
    for group_id in range(C.shape[1]):
        # which labels are in the same cluster?
        row_ids = C.indices[C.indptr[group_id] : C.indptr[group_id + 1]]
        row_ids = row_ids.copy()
        np.random.shuffle(row_ids)
        # randomly group labels n-by-n
        nr_new_labels_in_cluster = len(row_ids) // bin_size + len(row_ids) % bin_size
        for id_of_bin in range(nr_new_labels_in_cluster):
            for id_in_bin in range(bin_size):
                idx = id_of_bin * bin_size + id_in_bin
                if idx < len(row_ids):
                    mapper[row_ids[idx]] = id_of_bin + new_label_count
        new_label_count += nr_new_labels_in_cluster
    return mapper, new_label_count


def inversion_mapper(mapper):
    """Find the list of label pairs that are mapped to the same bins.

    This amounts to finding the keys with the same labels in mapper. The results are sorted in increasing order.

    Args:
        mapper: the result from `combined_from_cluster` function.

    Returns:
        the inversion of mapper.
    """
    invert_mapper = defaultdict(list)
    for k, v in mapper.items():
        invert_mapper[v].append(k)
    invert_mapper = {k: sorted(v) for k, v in invert_mapper.items()}
    return invert_mapper


if __name__ == "__main__":
    np.random.seed(0)
    parser = argparse.ArgumentParser("Create dataset for Section 5.1 in our paper.")
    parser.add_argument(
        "--data", type=str, default="eurlex-4k", help="Name of the dataset."
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="Select the level at which labels in the same clusters are combined.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=2,
        help="How many labels to group for each synthetic composite label.",
    )
    args = parser.parse_args()

    data = args.data
    bin_size = args.bin_size
    combine_level = args.level
    folder = f"./dataset/xmc-base/{data}/"
    out_folder = f"./dataset-binned/{data}/"
    C = cluster_chain(f"./model/{data}/ranker/**/C.npz", combine_level)
    os.makedirs(out_folder, exist_ok=True)

    ytr = sps.load_npz(folder + "Y.trn.npz")
    yte = sps.load_npz(folder + "Y.tst.npz")

    mapper, new_nr_labels = combine_from_cluster(C, bin_size)
    new_ytr = combine_Y(mapper, new_nr_labels, ytr)
    new_yte = combine_Y(mapper, new_nr_labels, yte)
    invert_mapper = inversion_mapper(mapper)
    with open(out_folder + "mapper.pkl", "wb") as writer:
        pkl.dump(invert_mapper, writer)
    sps.save_npz(out_folder + "Y.trn.npz", new_ytr)
    sps.save_npz(out_folder + "Y.tst.npz", new_yte)
