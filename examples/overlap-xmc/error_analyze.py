import argparse
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as smat
from pecos.core import clib
from pecos.utils import smat_util
from pecos.xmc.xlinear.model import XLinearModel


def parse_arguments():
    parser = argparse.ArgumentParser()
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
        type=str,
        required=True,
        metavar="DIR",
        help="path to the model folder",
    )
    parser.add_argument(
        "--mapper",
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
        "-b", "--beam-size", type=int, default=10, help="Beam size at inference time.",
    )
    args = parser.parse_args()
    return args


def build_label_mapping_matrix(mapper, n_labels, n_new_labels):
    """Build a matrix for label score aggregration.

    The label mapping matrix is of shape (n_labels + n_dup_labels) x (n_labels). 
    The content is 
    [
        I,
        S,
    ]
    where I is identity matrix of shape n_labels x n_labels, and S is of shape
    n_dup_labels x n_labels. S(i, j) = 1 iff (i + n_labels) is a duplication of
    label j.

    Args:
        mapper: A mapper from duplicated label id to its original id.
        n_labels: Number of original labels.
        n_new_labels: Equals to n_labels + n_dup_labels.
    
    Returns:
        The sparse matrix for label score aggregation.
    """
    eye = smat.diags(np.ones(n_labels), format="csc", dtype=np.float32)
    rows, cols, data = [], [], []
    for row in range(n_labels, n_new_labels):
        rows.append(row - n_labels)
        cols.append(mapper[row])
        data.append(1)
    more = smat.coo_matrix(
        (data, (rows, cols)),
        shape=(n_new_labels - n_labels, n_labels),
        dtype=np.float32,
    )
    mapper_mat = smat.vstack((eye, more), format="csc")
    # normalize columns
    # D = 1.0 / np.asarray(mapper_mat.sum(axis=0)).squeeze(axis=0)
    # normalized_mapper = mapper_mat.dot(smat.diags(D, format="csc"))
    # return normalized_mapper
    return mapper_mat


def forward_matcher(xlinear_model, X, **kwargs):
    """Forward through the matcher to get the leaf id.
    
    Args:
        xlinear_model: The input xlinear model.
        X: Input feature matrix.
        kwargs: The config.
    
    Returns:
        The matching matrix.
    """
    csr_codes = None
    model_chain = xlinear_model.model.model_chain
    for layer_id in range(len(model_chain) - 1):
        cur_model = model_chain[layer_id]
        level_pred = cur_model.predict(
            X,
            csr_codes=csr_codes,
            only_topk=kwargs["beam_size"],
            post_processor=kwargs["post_processor"],
        )
        csr_codes = level_pred
    return csr_codes


def forward_ranker(xlinear_model, X, csr_codes, **kwargs):
    """Forward the ranker using the matching results.
    
    Args:
        xlinear_model: The input xlinear model.
        X: Feature matrix.
        csr_codes: The matching matrix.
        kwargs: The config.
    
    Returns:
        Y_hat: The score matrix of shape n_data x n_labels.
    """
    model_chain = xlinear_model.model.model_chain[-1]
    Y_hat = model_chain.predict(
        X,
        csr_codes=csr_codes,
        only_topk=kwargs["beam_size"],
        post_processor=kwargs["post_processor"],
    )
    return Y_hat


def merge_pseudo_labels(mapper, truth, pred, merge_by="max"):
    """Merge the prediction results of truth and prediction, combining the real labels
    and pseudo labels. The combination algorithm can be max or mean.

    Args:
        mapper: The label mapping matrix. Find the actual label id from the new label id.
        truth: The groundtruth matrix.
        pred: The prediction results.
        merge_by: If "max" is used, aggregate the scores by max operator. If "mean" is 
            used, aggregated by average.
    
    Returns:
        The aggregated scores.
    """
    # pred: N x Lnew, Lnew x Lold ==> N x Lold
    n_labels = pred.shape[1] - len(mapper)
    n_new_labels = pred.shape[1]
    if len(mapper) == 0:
        return truth, pred
    if merge_by == "mean":
        normalized_mapper = build_label_mapping_matrix(mapper, n_labels, n_new_labels)
        pred1 = clib.sparse_matmul(pred, normalized_mapper)
        # debiasing
        pred_sign = pred.sign()
        bias = clib.sparse_matmul(pred_sign, normalized_mapper)
        bias.data = 1.0 / np.clip(bias.data, a_min=0.1, a_max=None)
        pred = pred1.multiply(bias)
    else:
        pred = pred.tolil()
        truth = truth.tolil()
        for pseudo_id, real_id in mapper.items():
            pred[:, real_id] = pred[:, real_id].maximum(pred[:, pseudo_id])
    return truth[:, :n_labels], pred[:, :n_labels]


def build_score_transformer(unused_label_set, total_labels):
    """Build a matrix that mask-out the unused labels.
    
    This matrix is essentially all zeros in the row i if label i
    is never being used. Otherwise D(i, i) = 1.
    """
    D = np.ones(total_labels)
    for unused_label in unused_label_set:
        D[unused_label] = 0
    transformer = smat.diags(
        D, shape=(total_labels, total_labels), format="csc", dtype=np.float32
    )
    return transformer


def do_analyze(args):
    # Load Data
    Xt = XLinearModel.load_feature_matrix(args.inst_path)
    Yt = XLinearModel.load_label_matrix(args.label_path)

    # Optionally load mapper
    mapper = {}
    if args.mapper is not None:
        with open(args.mapper, "rb") as reader:
            mapper = pkl.load(reader)
    unused_label_set = {}
    if args.unused_labels is not None:
        with open(args.unused_labels, "rb") as reader:
            unused_label_set = pkl.load(reader)

    # Model prediction
    xlinear_model = XLinearModel.load(args.model_folder)
    kwargs = {
        "beam_size": args.beam_size,
        "only_topk": 160,
        "post_processor": "l3-hinge",
    }

    pred = None
    batch_size = 8192 * 16
    pred_batches = []
    M_batches = []
    for i in range((Xt.shape[0] - 1) // batch_size + 1):
        beg, end = i * batch_size, (i + 1) * batch_size
        end = min(end, Xt.shape[0])
        X_batch = Xt[beg:end, :]
        M_batch = forward_matcher(xlinear_model, X_batch, **kwargs)
        # pred_batch = forward_ranker(xlinear_model, X_batch, M_batch, **kwargs)
        pred_batch = xlinear_model.predict(Xt[beg:end, :], **kwargs)
        M_batches.append(M_batch)
        pred_batches.append(pred_batch)

    Mb = smat_util.binarized(smat.vstack(M_batches))
    C = xlinear_model.model.model_chain[-1].pC.buf
    MC = clib.sparse_matmul(Mb, C.transpose())
    avg_inner_prod = MC.sum(axis=1).mean(axis=0)[0, 0]

    pred = smat.vstack(pred_batches)
    unused_label_transformer = build_score_transformer(unused_label_set, pred.shape[1])
    # we set j-th column of pred to zero, iff j is an unused label, this is prevent label j
    # from being accidentally ranked to the front.
    pred = clib.sparse_matmul(pred, unused_label_transformer)
    truth = Yt

    print("Merging pseudo labels")
    truth, pred = merge_pseudo_labels(mapper, truth, pred, merge_by="mean")
    truth = truth.tocsr()
    pred = pred.tocsr()
    print("Calculating metrics")
    metric = smat_util.Metrics.generate(truth, pred, topk=10)
    print(metric)
    print("Average #inner prod: ", avg_inner_prod)


if __name__ == "__main__":
    args = parse_arguments()
    do_analyze(args)
