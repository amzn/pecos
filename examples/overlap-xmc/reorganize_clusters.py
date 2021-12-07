import argparse
import os
import pickle as pkl
from collections import defaultdict

import numpy as np
import scipy.sparse as smat
from numba import njit
from numba.core import types
from numba.typed import Dict
from pecos.core.base import clib
from pecos.utils import smat_util
from pecos.utils.cluster_util import ClusterChain
from pecos.xmc import MLModel
from pecos.xmc.xlinear import XLinearModel


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="Reorganize the clusters, move some of the labels around to "
        "improve recall."
    )
    parser.add_argument(
        "-x",
        "--inst-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to npz file of feature matrix",
    )
    parser.add_argument(
        "-y",
        "--label-path",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the npz file of the label matrix",
    )
    parser.add_argument(
        "-m",
        "--model-folder",
        type=lambda p: os.path.abspath(p),
        required=True,
        metavar="DIR",
        help="path to the model folder",
    )
    parser.add_argument(
        "-o",
        "--model-folder-output",
        type=lambda p: os.path.abspath(p),
        required=True,
        metavar="DIR",
        help="path to the model output folder",
    )
    parser.add_argument(
        "-b",
        "--beam-size",
        type=int,
        required=True,
        help="Beam size to calculate the matching matrix.",
    )
    parser.add_argument(
        "--n_copies", type=int, default=2, help="number of copies for each label.",
    )
    args = parser.parse_args()
    return args


def get_matching_matrix(xlinear_model, Xt, beam_size=10):
    """Get the matching matrix.
    
    The matching matrix indicates which cluster(s) are selected for data point in X. The 
    final results is a sparse matrix of shape N x C, where N is the number of data, and C
    is the number of clusters.

    Args:
        xlinear_model: the pretrained model.
        Xt: the feature matrix.
        beam_size: beam size for inference.
    
    Returns:
        The binarized matching matrix in CSR format.
    """
    matching_result = []
    batch_size = 8192 * 16
    kwargs = {
        "beam_size": beam_size,
        "only_topk": 30,
        "post_processor": "l3-hinge",
    }
    model_chain = xlinear_model.model.model_chain
    for i in range((Xt.shape[0] - 1) // batch_size + 1):
        beg, end = i * batch_size, (i + 1) * batch_size
        end = min(end, Xt.shape[0])
        X_selected = Xt[beg:end]
        csr_codes = None
        for level in range(len(model_chain) - 1):
            model_l = model_chain[level]
            level_pred = model_l.predict(
                X_selected,
                csr_codes=csr_codes,
                only_topk=beam_size,
                post_processor=kwargs["post_processor"],
            )
            csr_codes = level_pred
        matching_result.append(csr_codes)
    matching_result = smat.vstack(matching_result, format="csr").sign()
    return matching_result


@njit
def construct_new_C_and_Y(
    counts_rows,
    counts_cols,
    counts,
    row_ids,
    row_ranges,
    C_rows,
    sort_idx,
    nr_labels,
    max_cluster_size,
    n_copies,
):
    """Determine the new clustering matrix and the new label matrix given the couting matrix.

    This function implements Eq.(10) in our paper. I.e. given the couting matrix C = Y^T * M, 
    we select the correct cluster id for each label one by one, in descending order of C entries,
    possibly assign a label multiple times (`n_copies`) to different clusters. Finally, the new 
    cluster and new label matrix is returned. Notice that Numba is used here, this prevents us 
    from passing scipy sparse matrix directly.
    
    Args:
        counts_rows, counts_cols, counts: The counting matrix in COO format.
        row_ids, row_ranges: The indices and indptr of original Y matrix in CSC format.
        C_rows: Clustering matrix C in LIL format, converted to list of numpy arrays.
        sort_idx: Index of counts_{rows,cols} to sort them in decending order.
        nr_labels: Number of original labels.
        max_cluster_size: (Unused for now) Hard constraints to limit the number of labels 
            in each cluster (to balance cluster size).
        n_copies: Max number of copies for each label (\lambda in our paper).

    Returns:
        New cluster matrix (`new_C_*`), new label matrix (`new_Y_*`), the replicated label
        assignment (`C_overlap_*`), number of duplicated labels (`nr_copied_labels`), a map 
        from new label id to the underlying label id (`mapper`), unused labels that never 
        show up in training (`unused_labels`), number of lightly used labels (`nr_tail_labels`).
    """
    # construct empty cluster matrix and label matrix
    nr_copied_labels = 0
    new_C_cols = []
    new_C_data = []
    new_Y_rows = []
    labels_included = set()
    mapper = Dict.empty(key_type=types.int64, value_type=types.int64,)
    cluster_size = Dict.empty(key_type=types.int64, value_type=types.int64,)
    pseudo_label_count = Dict.empty(key_type=types.int64, value_type=types.int64,)
    # results
    C_overlap_rows, C_overlap_cols = [], []
    max_count = n_copies
    # adding labels to clusters one by one in descending frequency
    for idx in sort_idx:
        label_id = counts_rows[idx]
        leaf_id = counts_cols[idx]
        if label_id in pseudo_label_count and pseudo_label_count[label_id] >= max_count:
            continue
        # If you need to contrain the max cluster size, then
        # uncomment following two lines
        # if label_count[leaf_id] >= max_cluster_size:
        #    continue
        if leaf_id not in cluster_size:
            cluster_size[leaf_id] = 1
        else:
            cluster_size[leaf_id] += 1

        if label_id not in pseudo_label_count:
            pseudo_label_count[label_id] = 1
        else:
            pseudo_label_count[label_id] += 1

        if label_id in labels_included:
            # add a pseudo label that duplicates label_id
            pseudo_label_id = nr_copied_labels + nr_labels
            mapper[pseudo_label_id] = label_id
            # add one more row to C (in lil format)
            new_C_cols.append([leaf_id])
            new_C_data.append([1])
            # add one more column to Yt
            examples = row_ids[row_ranges[label_id] : row_ranges[label_id + 1]]
            new_Y_rows.append(examples)
            nr_copied_labels += 1
        else:
            # add a new label
            labels_included.add(label_id)
            C_overlap_rows.append(label_id)
            C_overlap_cols.append(leaf_id)

        # exit early if we have too many effective labels
        if len(mapper) >= max_count * nr_labels:
            break
    # add missing labels back to clusters
    nr_tail_labels = 0
    for label_id in range(nr_labels):
        if label_id not in labels_included:
            original_leaf_id = C_rows[label_id][0]
            C_overlap_rows.append(label_id)
            C_overlap_cols.append(original_leaf_id)
            labels_included.add(label_id)
            nr_tail_labels += 1

    unused_labels = set()
    for label_id in range(nr_labels):
        if label_id not in labels_included:
            unused_labels.add(label_id)

    # new_Y elements
    new_Y_indptr = [0]
    new_Y_indices = []
    for rows in new_Y_rows:
        new_Y_indptr.append(new_Y_indptr[-1] + len(rows))
        new_Y_indices.extend(rows)
    new_Y_data = np.ones(len(new_Y_indices), dtype=np.int32)
    return (
        new_C_cols,
        new_C_data,
        new_Y_data,
        new_Y_indices,
        new_Y_indptr,
        C_overlap_cols,
        C_overlap_rows,
        nr_copied_labels,
        mapper,
        unused_labels,
        nr_tail_labels,
    )


def get_topk_clusters(xlinear_model, Xt, Yt, beam_size):
    """Iterate over all labels, for each label, gather the training samples
    from which the label is activated. Then for each training sample, gather
    the top-1 (beam_size=1) leaf node. The topk leaves will be returned for
    the reorganization step.

    Args:
        xlinear_model: the pre-trained model.
        Xt: traning features in csr format.
        Yt: training labels in csr format.

    Returns:
        None, xlinear_model will be modified in-place.
    """
    # Get the clustering matrix in the last level
    model_chain = xlinear_model.model.model_chain
    leaf_model = model_chain[-1]

    # Get the matching matrix
    M = get_matching_matrix(xlinear_model, Xt, beam_size=beam_size)
    C = leaf_model.pC.buf

    # Get the counting matrix by YtM
    counts = Yt.transpose().dot(M).tocoo()
    counts.eliminate_zeros()

    counts_rows, counts_cols, counts = counts.row, counts.col, counts.data
    sort_idx = np.argsort(counts)[::-1]

    Yt_csc = Yt.tocsc()
    row_ranges = Yt_csc.indptr
    row_ids = Yt_csc.indices

    # Cast C to lil format
    C = C.tolil()
    C_rows = C.rows
    max_cluster_size = int(1.0 * C.shape[0] / C.shape[1])
    (
        new_C_cols,
        new_C_data,
        new_Y_data,
        new_Y_indices,
        new_Y_indptr,
        C_overlap_cols,
        C_overlap_rows,
        out_labels,
        mapper,
        unused_labels,
        nr_tail_labels,
    ) = construct_new_C_and_Y(
        np.asarray(counts_rows, dtype=np.int32),
        np.asarray(counts_cols, dtype=np.int32),
        np.asarray(counts, dtype=np.int32),
        np.asarray(row_ids, dtype=np.int32),
        np.asarray(row_ranges, dtype=np.int32),
        [np.asarray(row, dtype=np.int32) for row in C_rows],
        sort_idx,
        Yt.shape[1],
        max_cluster_size,
        args.n_copies,
    )
    C_overlap = smat.coo_matrix(
        (np.ones_like(C_overlap_cols), (C_overlap_rows, C_overlap_cols)),
        shape=C.shape,
        dtype=C.dtype,
    ).tocsr()
    print(f"#copied labels: {out_labels}, #tail labels: {nr_tail_labels}")

    new_C = smat.lil_matrix((out_labels, C.shape[1]), dtype=C.dtype)
    new_C.data = new_C_data
    new_C.rows = new_C_cols
    C = smat.vstack((C_overlap, new_C.tocsc()), format="csc")

    new_Y = smat.csc_matrix(
        (new_Y_data, new_Y_indices, new_Y_indptr),
        shape=(Yt.shape[0], len(new_Y_indptr) - 1),
        dtype=Yt.dtype,
    )
    Yt = smat.hstack((Yt, new_Y), format="csr")

    assert C.shape[1] == leaf_model.pC.buf.shape[1]
    assert C.shape[0] == Yt.shape[1]
    return C, Yt, dict(mapper), unused_labels


def main(args):
    # Load Data
    Xt = XLinearModel.load_feature_matrix(args.inst_path)
    Yt = XLinearModel.load_label_matrix(args.label_path)

    # Model prediction
    xlinear_model = XLinearModel.load(args.model_folder)
    C, Yt, mapper, unused_labels = get_topk_clusters(
        xlinear_model, Xt, Yt, args.beam_size
    )

    # Extract the cluster chain from model_chain
    clusters = [m.pC.buf for m in xlinear_model.model.model_chain[:-1]] + [C]
    chain = ClusterChain(clusters)

    # Save to folder
    chain.save(args.model_folder_output)

    xt_out_name = os.path.join(
        args.model_folder_output, os.path.basename(args.inst_path)
    )
    smat.save_npz(xt_out_name, Xt)
    yt_out_name = os.path.join(
        args.model_folder_output, os.path.basename(args.label_path)
    )
    smat.save_npz(yt_out_name, Yt)
    mapper_file = os.path.join(args.model_folder_output, "pseudo_label_mapping.pkl")
    with open(mapper_file, "wb") as writer:
        pkl.dump(mapper, writer)
    unused_labels_file = os.path.join(args.model_folder_output, "unused_labels.pkl")
    with open(unused_labels_file, "wb") as writer:
        pkl.dump(unused_labels, writer)


if __name__ == "__main__":
    args = parse_arguments()
    # save to new directory, avoid overwritting
    assert (
        args.model_folder != args.model_folder_output
    ), "You can't set the model desitination path to be the same as source path."
    if not os.path.exists(args.model_folder_output):
        os.makedirs(args.model_folder_output)
    main(args)
