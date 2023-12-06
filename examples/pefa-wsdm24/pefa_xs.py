
import logging
import sys
import random
import numpy as np
from pecos.ann.hnsw import HNSW
from pecos.utils import smat_util
from pecos.xmc import LabelEmbeddingFactory
from sentence_transformers import LoggingHandler
from sklearn.preprocessing import normalize

from utils import get_data_aug, get_eval_metric


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


# FIXED HYPER-PARAMETERS
threads = 64
M, efC = 32, 500
efS, topk = 300, 100
metric_type = "ip"


def run_pefa_xs(
    P_emb_npy,
	Q_emb_trn, Y_trn_npz,
	Q_emb_tst, Y_tst_npz,
    lambda_erm=0.5,
):
    if lambda_erm >= 0.0 and lambda_erm < 1.0:
        pifa_emb = LabelEmbeddingFactory.create(
            Y_trn_npz,
            Q_emb_trn,
            method="pifa",
            normalized_Y=False,
        )
        label_emb = lambda_erm * P_emb_npy + (1.0 - lambda_erm) * pifa_emb
    elif lambda_erm == 1.0:
        label_emb = P_emb_npy
    else:
        raise ValueError(f"lambda_erm={lambda_erm} should be in [0.0, 1.0]!")

    # build ANN index
    train_params = HNSW.TrainParams(M=M, efC=efC, metric_type=metric_type, threads=threads)
    index_P = HNSW.train(label_emb, train_params=train_params, pred_params=None)

    # inference
    pred_params = HNSW.PredParams(efS=efS, topk=topk)
    searchers = index_P.searchers_create(num_searcher=threads)
    Yp_tst = index_P.predict(Q_emb_tst, pred_params=pred_params, searchers=searchers, ret_csr=True)
    Yp_tst.data = 1.0 - Yp_tst.data

    # eval recall
    eval_recall_str = get_eval_metric(Y_tst_npz, Yp_tst, topk=100)
    logging.info("lambda_erm {:.2f} | {}".format(lambda_erm, eval_recall_str))


def main(input_xmc_dir, input_emb_dir, lambda_erm):
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)

    logging.info("Loading input-to-label matrix..")
    Y_trn = smat_util.load_matrix(f"{input_xmc_dir}/Y.trn.npz")
    Y_tst = smat_util.load_matrix(f"{input_xmc_dir}/Y.tst.npz")
    Y_abs = smat_util.load_matrix(f"{input_xmc_dir}/Y.trn.abs.npz")
    Y_doc = smat_util.load_matrix(f"{input_xmc_dir}/Y.trn.doc.npz")
    Y_d2q = smat_util.load_matrix(f"{input_xmc_dir}/Y.trn.d2q.npz")

    logging.info("Loading input embedding matrix..")
    X_trn = smat_util.load_matrix(f"{input_emb_dir}/X.trn.npy")     # trn set emb from real query text
    X_tst = smat_util.load_matrix(f"{input_emb_dir}/X.tst.npy")     # tst set emb from real query text
    X_abs = smat_util.load_matrix(f"{input_emb_dir}/X.trn.abs.npy") # trn set emb from doc's abstract+title text
    X_doc = smat_util.load_matrix(f"{input_emb_dir}/X.trn.doc.npy") # trn set emb from doc's content (first 512 tokens)
    X_d2q = smat_util.load_matrix(f"{input_emb_dir}/X.trn.d2q.npy") # trn set emb from docT5query using doc's content
    X_trn = normalize(X_trn, axis=1, norm="l2")
    X_tst = normalize(X_tst, axis=1, norm="l2")
    X_abs = normalize(X_abs, axis=1, norm="l2")
    X_doc = normalize(X_doc, axis=1, norm="l2")
    X_d2q = normalize(X_d2q, axis=1, norm="l2")

    logging.info("Gathering data augmentation..")
    P_emb = LabelEmbeddingFactory.create(Y_abs, X_abs, method="pifa", normalized_Y=False)
    X_aug, Y_aug = get_data_aug(X_trn, X_doc, X_d2q, Y_trn, Y_doc, Y_d2q, aug_type="v5")

    logging.info("Running PEFA-XS..")
    run_pefa_xs(P_emb, X_aug, Y_aug, X_tst, Y_tst, lambda_erm=lambda_erm)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("python pefa_xs.py [input_xmc_dir] [input_emb_dir] [lambda_erm]")
        exit(0)
    input_xmc_dir = sys.argv[1]
    input_emb_dir = sys.argv[2]
    lambda_erm = float(sys.argv[3])
    main(input_xmc_dir, input_emb_dir, lambda_erm)
