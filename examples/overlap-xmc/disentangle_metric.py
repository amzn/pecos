import argparse
import os
import pickle as pkl

import numpy as np
import scipy.sparse as smat
from pecos.core.base import clib
from pecos.utils import smat_util
from pecos.utils.cluster_util import ClusterChain
from pecos.xmc import MLModel
from pecos.xmc.xlinear import XLinearModel


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="Evaluate how well our model is good at semantic disentablement."
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
        "--y-origin",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the npz file of the original label matrix",
    )
    parser.add_argument(
        "--y-binned",
        type=str,
        required=True,
        metavar="PATH",
        help="path to the binned label matrix",
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
        "--binned-mapper", type=str, required=True, help="path to the mapper file",
    )
    parser.add_argument(
        "--pseudo-label-mapper",
        type=str,
        default=None,
        help="path to pseudo label mapper. If None, this variable is ignored.",
    )
    parser.add_argument(
        "--unused-labels",
        type=str,
        default=None,
        help="path to unused label set. If None, this variable is ignored.",
    )
    parser.add_argument(
        "-b",
        "--beam-size",
        type=int,
        required=True,
        help="Beam size to calculate the matching matrix.",
    )
    args = parser.parse_args()
    return args


def get_matching_matrix(xlinear_model, Xt, beam_size=10):
    """Compute the matching matrix.
    
    The matching matrix indicates which cluster(s) are selected for data point in X. The 
    final results is a sparse matrix of shape N x C, where N is the number of data, and C
    is the number of clusters.

    Args:
        xlinear_model: the pretrained model.
        Xt: the feature matrix.
        beam_size: beam size for inference.
        
    Returns:
        The matching matrix in CSR format.
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
            cur_model = model_chain[level]
            level_pred = cur_model.predict(
                X_selected,
                csr_codes=csr_codes,
                only_topk=beam_size,
                post_processor=kwargs["post_processor"],
            )
            csr_codes = level_pred
        matching_result.append(csr_codes)
    matching_result = smat.vstack(matching_result, format="csr")
    return matching_result


def positive_instances(Xt, Yt, underlying_label_ids):
    """Find the instances having some particular label ids.
    
    For all labels in `underlying_label_ids`, return the list of instances containing 
    that label as ground-truth.

    Args:
        Xt: The feature matrix of shape N x d, where N is number of instances, d is 
            feature dimension.
        Yt: The label matrix of shape N x L, L is the size of label space.
        underlying_label_ids: The set of target labels.
    
    Returns:
        A list of positive instance ids and their feature vectors.
    """
    row_ids_list = []
    Xt_subsets = []
    for label_id in underlying_label_ids:
        row_ids = Yt.indices[Yt.indptr[label_id] : Yt.indptr[label_id + 1]]
        Xt_subsets.append(Xt[row_ids])
        row_ids_list.append(row_ids)
    return row_ids_list, Xt_subsets


def label_id_to_cluster_id(label_id, C, unused_labels):
    """Map the label id to the cluster id according to clustering matrix.
    
    Args:
        label_id: the label id.
        C: the cluster matrix of shape L x C.
        unused_labels: used to adjust the label id.
    
    Returns:
        the cluster id.
    """
    # count how many unused labels that are smaller than label_id
    offset = sum([l < label_id for l in unused_labels])
    row_id = label_id - offset
    assert C.indptr[row_id] + 1 == C.indptr[row_id + 1]
    cluster_id = C.indices[C.indptr[row_id]]
    return cluster_id


def match(
    xlinear_model, beam_size, instance_ids_list, X_subsets, cid1, cid2,
):
    """Given two clusters, distribute all instances to two groups.

    Separate all input features `X_subsets` into two subsets `x_cid1` and `x_cid2`,
    according to the prediction results from `xlinear_model`. If the scores of an 
    instance in `cid1` is higher than `cid2`, than this instance is assigned to group1.

    Args:
        xlinear_model: the model.
        beam_size: beam size for inference.
        instance_ids_list: the instance id of `X_subsets`.
        X_subsets: the feature matrix.
        cid1, cid2: the cluster ids of two clusters.
    
    Returns:
        the instance ids of two subsets.
    """
    x_cid1 = []
    x_cid2 = []
    for instance_ids, X_subset in zip(instance_ids_list, X_subsets):
        matching_matrix = get_matching_matrix(
            xlinear_model, X_subset, beam_size,
        ).toarray()
        mask = matching_matrix[:, cid1] > matching_matrix[:, cid2]
        x_cid1.extend(instance_ids[mask])
        x_cid2.extend(instance_ids[~mask])
    return x_cid1, x_cid2


def random_baseline(S1, S2):
    """A random baseline that assigns all instances randomly to two groups.
    
    Args:
        S1, S2: the ground truth assignment according to their semantic meanings.
    
    Returns:
        VI scores of this random baseline.
    """
    S = np.concatenate((S1, S2), axis=0)
    experiment = []
    for _ in range(100):
        np.random.shuffle(S)
        selector = np.random.randn(len(S)) > 0
        K1 = S[selector]
        K2 = S[~selector]
        vi_sample = VI(S1, S2, K1, K2)
        experiment.append(vi_sample)
    return np.mean(experiment)


def VI(S1, S2, K1, K2):
    """Computes the Variation of Information(VI) between two clusters.
    
    See: https://en.wikipedia.org/wiki/Variation_of_information for more information.

    Args:
        S1, S2: the set of ground truth clusters.
        K1, K2: the predicted clusters.
    
    Returns:
        the VI score.
    """
    assert len(S1) + len(S2) == len(K1) + len(K2)
    n = len(S1) + len(S2)
    eps = 1.0e-8
    p1 = len(S1) / n + eps
    p2 = len(S2) / n + eps
    q1 = len(K1) / n + eps
    q2 = len(K2) / n + eps
    r11 = len(np.intersect1d(S1, K1)) / n + eps
    r12 = len(np.intersect1d(S1, K2)) / n + eps
    r21 = len(np.intersect1d(S2, K1)) / n + eps
    r22 = len(np.intersect1d(S2, K2)) / n + eps
    vi = (
        r11 * (np.log(r11 / p1) + np.log(r11 / q1))
        + r12 * (np.log(r12 / p1) + np.log(r12 / q2))
        + r21 * (np.log(r21 / p2) + np.log(r21 / q1))
        + r22 * (np.log(r22 / p2) + np.log(r22 / q2))
    )
    return -vi


def main(args):
    # Load Data
    Xt = XLinearModel.load_feature_matrix(args.inst_path)
    Yt_bin = XLinearModel.load_label_matrix(args.y_binned)
    Yt = XLinearModel.load_label_matrix(args.y_origin).tocsc()

    # Optionally load mapper
    mapper = {}
    if args.pseudo_label_mapper is not None:
        with open(args.pseudo_label_mapper, "rb") as reader:
            mapper = pkl.load(reader)
    inv_mapper = {v: k for k, v in mapper.items()}

    unused_label_set = {}
    if args.unused_labels is not None:
        with open(args.unused_labels, "rb") as reader:
            unused_label_set = pkl.load(reader)

    # Mapper that maps from binned label to its components
    with open(args.binned_mapper, "rb") as reader:
        binned_label_mapper = pkl.load(reader)

    # Model prediction
    xlinear_model = XLinearModel.load(args.model_folder)

    # label clustering matrix of size L x K
    leaf_model = xlinear_model.model.model_chain[-1]
    C = leaf_model.pC.buf.tocsr()
    for fake_label_id, underlying_label_ids in binned_label_mapper.items():
        if len(underlying_label_ids) < 2 or fake_label_id not in inv_mapper:
            continue
        # given the label ids, gather all positive instances
        instance_ids_list, X_subsets = positive_instances(Xt, Yt, underlying_label_ids)
        if min(len(s) for s in instance_ids_list) <= 10:
            continue

        pseudo_label_id = inv_mapper[fake_label_id]
        cid_fake_label = label_id_to_cluster_id(fake_label_id, C, unused_label_set)
        cid_pseudo_label = label_id_to_cluster_id(pseudo_label_id, C, unused_label_set)
        assert cid_fake_label != cid_pseudo_label
        x_cid1, x_cid2 = match(
            xlinear_model,
            args.beam_size,
            instance_ids_list,
            X_subsets,
            cid_fake_label,
            cid_pseudo_label,
        )
        vi = VI(x_cid1, x_cid2, *instance_ids_list)
        baseline_vi = random_baseline(x_cid1, x_cid2)
        print(vi, baseline_vi, baseline_vi - vi)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
