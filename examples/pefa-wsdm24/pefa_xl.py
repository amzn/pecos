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
efS, topk_p = 300, 500
metric_type = "ip"


def run_pefa_xl(
    P_emb, Q_emb_trn, Y_trn_npz, Q_emb_tst, Y_tst_npz,
    topk_q=32, lambda_erm=0.5, knn_type="v0",
):
    logging.info("Building HNSW Index for f_erm")
    train_params = HNSW.TrainParams(M=M, efC=efC, metric_type=metric_type, threads=threads)
    index_P = HNSW.train(P_emb, train_params=train_params, pred_params=None)
    pred_params = HNSW.PredParams(efS=efS, topk=topk_p)
    searchers = index_P.searchers_create(num_searcher=threads)
    Yp_erm = index_P.predict(Q_emb_tst, pred_params=pred_params, searchers=searchers, ret_csr=True)
    Yp_erm.data = 1.0 - Yp_erm.data

    logging.info("Building HSNW Index for f_knn")
    Yp_tst = None
    if lambda_erm >= 0.0 and lambda_erm < 1.0:
        train_params = HNSW.TrainParams(M=M, efC=efC, metric_type=metric_type, threads=threads)
        index_Q = HNSW.train(Q_emb_trn, train_params=train_params, pred_params=None)
        pred_params_q = HNSW.PredParams(efS=efS, topk=topk_q)
        searchers = index_Q.searchers_create(num_searcher=threads)
        qQT = index_Q.predict(Q_emb_tst, pred_params=pred_params_q, searchers=searchers, ret_csr=True)
        if knn_type == "v0":
            qQT.data = (1.0 - qQT.data) / float(topk_q) # normalizing
        elif knn_type == "v1":
            qQT.data = (1.0 - qQT.data)  # no normalizing
        else:
            raise ValueError(f"knn_type={knn_type} is not valid!")
        Yp_knn = qQT.dot(Y_trn_npz)
        Yp_tst = lambda_erm * Yp_erm + (1.0 - lambda_erm) * Yp_knn
    elif lambda_erm == 1.0:
        Yp_tst = Yp_erm
    else:
        raise ValueError(f"lambda_erm={lambda_erm} should be in [0.0, 1.0]!")
    Yp_tst = smat_util.sorted_csr(Yp_tst)

    # eval recall
    eval_recall_str = get_eval_metric(Y_tst_npz, Yp_tst, topk=100)
    logging.info("topk_q {:3d} lambda_erm {:.2f} knn_type {} | {}".format(topk_q, lambda_erm, knn_type, eval_recall_str))


def main(input_xmc_dir, input_emb_dir, lambda_erm, topk_q, knn_type):
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

    logging.info("Running PEFA-XL..")
    run_pefa_xl(
        P_emb, X_aug, Y_aug, X_tst, Y_tst,
        topk_q=topk_q, lambda_erm=lambda_erm, knn_type=knn_type,
    )


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("python pefa_xl.py [input_xmc_dir] [input_emb_dir] [lambda_erm] [topk_q] [knn_type]")
        exit(0)
    input_xmc_dir = sys.argv[1]
    input_emb_dir = sys.argv[2]
    lambda_erm = float(sys.argv[3])
    topk_q = int(sys.argv[4])
    knn_type = str(sys.argv[5])
    main(input_xmc_dir, input_emb_dir, lambda_erm, topk_q, knn_type)
