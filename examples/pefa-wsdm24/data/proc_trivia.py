
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as smat
from pecos.utils import smat_util


COL_NAME_LIST = [
    "query", "qid", "doc_id",
    "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3", "bert_k30_c30_4", "bert_k30_c30_5",
]

QG_COL_NAME_LIST = [
    "query", "doc_id",
    "bert_k30_c30_1", "bert_k30_c30_2", "bert_k30_c30_3", "bert_k30_c30_4", "bert_k30_c30_5",
]


def load_df(input_tsv_path):
    return pd.read_csv(
        input_tsv_path,
        encoding='utf-8', header=None, sep='\t',
        names=COL_NAME_LIST,
        dtype={"query": str, "qid": str, 'doc_id': str},
        on_bad_lines="skip",
        skip_blank_lines=True,
    ).loc[:, ["query", "qid", "doc_id"]]


def load_qg_df(input_tsv_path):
    return pd.read_csv(
        input_tsv_path,
        encoding='utf-8', header=None, sep='\t',
        names=QG_COL_NAME_LIST,
        dtype={"query": str, 'doc_id': str},
        on_bad_lines="skip",
        skip_blank_lines=True,
    ).loc[:, ["query", "doc_id"]]


def build_did2lid_map(df_inp, did_to_lid):
    for i in range(len(df_inp)):
        #did_str = df_inp["doc_id"][i]
        try:
            did_list = df_inp["doc_id"][i].split(",")
        except:
            print(i, type(df_inp["doc_id"][i]), df_inp["doc_id"][i])
            exit(0)
        for did_str in did_list:
            if did_str not in did_to_lid:
                did_to_lid[did_str] = len(did_to_lid)

def build_corpus_and_label_mat(df, did_to_lid, skip_same_lid=False):
    qry_to_qid = defaultdict(str)
    rows, cols = [], []
    inc_lid_set = set()
    for i in range(len(df)):
        query = df["query"][i]
        try:
            did_list = df["doc_id"][i].split(",")
        except:
            print(i, type(df["doc_id"][i]), df["doc_id"][i])
            exit(0)
        for did_str in did_list:
            lid = did_to_lid[did_str]
            if skip_same_lid and lid in inc_lid_set:
                continue
            inc_lid_set.add(lid)

            if query not in qry_to_qid:
                qry_to_qid[query] = len(qry_to_qid)
            qid = qry_to_qid[query]
            rows.append(qid)
            cols.append(lid)

    vals = [1.0 for _ in range(len(rows))]

    num_inp, num_out = len(qry_to_qid), len(did_to_lid)
    Y = smat.csr_matrix(
        (vals, (rows, cols)),
        shape=(num_inp, num_out),
        dtype=np.float32,
    )
    print("#Q {:7d} #L {:7d} NNZ {:9d}".format(num_inp, num_out, Y.nnz))
    id2query = [str(query).lower() for query, qid in sorted(qry_to_qid.items(), key=lambda x: x[1])]
    return id2query, Y


def write_qtxt(id2qtxt, output_path):
    with open(output_path, "w") as fout:
        for query_txt in id2qtxt:
            query_txt = query_txt.replace('\r\n', ' ')
            query_txt = query_txt.replace('\n', ' ')
            fout.write(f"{query_txt}\n")

def main():
    df_trn = load_df("./raw/trivia_newdata/train.tsv")
    df_tst = load_df("./raw/trivia_newdata/dev.tsv")
    df_abs = load_df("./raw/trivia_newdata/trivia_title_cont.tsv")
    df_doc = load_df("./raw/trivia_newdata/trivia_doc_aug.tsv")
    df_d2q = load_qg_df("./raw/trivia_newdata/trivia_512_qg.tsv")

    did_to_lid = defaultdict(str)
    build_did2lid_map(df_abs, did_to_lid)
    print("After df_abs, #uniq_label {:9d}".format(len(did_to_lid)))
    build_did2lid_map(df_doc, did_to_lid)
    print("After df_doc, #uniq_label {:9d}".format(len(did_to_lid)))
    build_did2lid_map(df_d2q, did_to_lid)
    print("After df_d2q, #uniq_label {:9d}".format(len(did_to_lid)))

    id2qtxt_trn, Y_trn_all = build_corpus_and_label_mat(df_trn, did_to_lid, skip_same_lid=False)
    id2qtxt_tst, Y_tst_all = build_corpus_and_label_mat(df_tst, did_to_lid, skip_same_lid=False)
    id2qtxt_abs, Y_trn_abs = build_corpus_and_label_mat(df_abs, did_to_lid, skip_same_lid=True)
    id2qtxt_doc, Y_trn_doc = build_corpus_and_label_mat(df_doc, did_to_lid, skip_same_lid=False)
    id2qtxt_d2q, Y_trn_d2q = build_corpus_and_label_mat(df_d2q, did_to_lid, skip_same_lid=False)

    output_dir = "./xmc/trivia"
    os.makedirs(output_dir, exist_ok=True)
    write_qtxt(id2qtxt_trn, f"{output_dir}/X.trn.txt")
    write_qtxt(id2qtxt_tst, f"{output_dir}/X.tst.txt")
    write_qtxt(id2qtxt_abs, f"{output_dir}/X.trn.abs.txt")
    write_qtxt(id2qtxt_doc, f"{output_dir}/X.trn.doc.txt")
    write_qtxt(id2qtxt_d2q, f"{output_dir}/X.trn.d2q.txt")
    smat_util.save_matrix(f"{output_dir}/Y.trn.npz", Y_trn_all)
    smat_util.save_matrix(f"{output_dir}/Y.tst.npz", Y_tst_all)
    smat_util.save_matrix(f"{output_dir}/Y.trn.abs.npz", Y_trn_abs)
    smat_util.save_matrix(f"{output_dir}/Y.trn.doc.npz", Y_trn_doc)
    smat_util.save_matrix(f"{output_dir}/Y.trn.d2q.npz", Y_trn_d2q)


if __name__ == "__main__":
    main()
