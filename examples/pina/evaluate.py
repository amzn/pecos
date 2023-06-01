# Example to evaluate
import sys
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils
from scipy.sparse import load_npz
import scipy.sparse as sparse
import numpy as np
import json
import os


def load_overlap(data_dir, filter_label_file='filter_labels.txt'):
    docs = np.asarray([])
    lbs = np.asarray([])
    if os.path.exists(os.path.join(data_dir, filter_label_file)):
        filter_lbs = np.loadtxt(os.path.join(
            data_dir, filter_label_file), dtype=np.int32)
        if filter_lbs.size > 0:
            docs = filter_lbs[:, 0]
            lbs = filter_lbs[:, 1]
    else:
        print("Overlap not found")
    print("Overlap is:", docs.size)
    return docs, lbs


def _remove_overlap(score_mat, docs, lbs):
    score_mat[docs, lbs] = 0
    score_mat = score_mat.tocsr()
    score_mat.eliminate_zeros()
    return score_mat


def main(targets_label_file, train_label_file, predictions_file, A, B, docs, lbls):
    true_labels = _remove_overlap(
        data_utils.read_sparse_file(
            targets_label_file, force_header=True).tolil(),
        docs, lbls)
    trn_labels = data_utils.read_sparse_file(
        train_label_file, force_header=True)
    inv_propen = xc_metrics.compute_inv_propesity(trn_labels, A=A, B=B)
    acc = xc_metrics.Metrics(
        true_labels, inv_psp=inv_propen, remove_invalid=False)
    predicted_labels = _remove_overlap(
        load_npz(predictions_file+'.npz').tolil(),
        docs, lbls)
    rec = xc_metrics.recall(predicted_labels, true_labels, k=10)[-1]*100
    print("R@10=%0.2f" % (rec))
    args = acc.eval(predicted_labels, 5)
    print(xc_metrics.format(*args))


if __name__ == '__main__':
    train_label_file = sys.argv[1]
    targets_file = sys.argv[2]  # Usually test data file
    predictions_file = sys.argv[3]  # In mat format
    data_dir=sys.argv[4]
    # configs = json.load(open(sys.argv[5]))["DEFAULT"]
    A = 0.6
    B = 2.6
    filter_data = "filter_labels_test.txt"
    docs, lbls = load_overlap(data_dir, filter_label_file=filter_data)
    main(targets_file, train_label_file, predictions_file, A, B, docs, lbls)
