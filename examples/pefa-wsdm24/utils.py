
import numpy as np
from pecos.utils import smat_util


def get_eval_metric(Y_true, Y_pred, topk=100):
    Y_pred = smat_util.sorted_csr(Y_pred)
    metric = smat_util.Metrics.generate(Y_true, Y_pred, topk=topk)
    eval_str_fnc = lambda evals, topks: " ".join("{:.4f}".format(v) for i, v in enumerate(evals) if i in topks)
    eval_str_obj = "R@10,20,100 {}".format(eval_str_fnc(metric.recall, [9, 19, 99]))
    return eval_str_obj


def get_data_aug(
	X_trn, X_doc, X_d2q,
	Y_trn, Y_doc, Y_d2q,
	aug_type="v0",
):
    if aug_type == "v0":
        X_aug = [X_trn]
        Y_aug = [Y_trn]
    elif aug_type == "v1":
        X_aug = [X_doc]
        Y_aug = [Y_doc]
    elif aug_type == "v2":
        X_aug = [X_d2q]
        Y_aug = [Y_d2q]
    elif aug_type == "v3":
        X_aug = [X_trn, X_doc]
        Y_aug = [Y_trn, Y_doc]
    elif aug_type == "v4":
        X_aug = [X_trn, X_d2q]
        Y_aug = [Y_trn, Y_d2q]
    elif aug_type == "v5":
        X_aug = [X_trn, X_doc, X_d2q]
        Y_aug = [Y_trn, Y_doc, Y_d2q]
    else:
        raise ValueError(f"aug_type={aug_type} is not support")
    X_aug = np.vstack(X_aug)
    Y_aug = smat_util.vstack_csr(Y_aug)
    return X_aug, Y_aug


