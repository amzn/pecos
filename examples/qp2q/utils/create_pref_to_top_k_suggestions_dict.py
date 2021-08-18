import os
import sys
import json
import argparse
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pygtrie import CharTrie

from qp2q.preprocessing.session_data_processing import parallel_get_qp2q_sparse_data
from qp2q.preprocessing.sparse_data_processing import SparseDataFrame

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def get_top_k_w_label_trie(prefix, label_trie, k):
    if label_trie.has_subtrie(prefix):  # if prefix exists in trie
        sorted_labels = [
            (label, freq)
            for label, freq in sorted(
                label_trie.iteritems(prefix=prefix), key=lambda x: x[1], reverse=True
            )
        ]

        return sorted_labels[:k]
    else:  # if prefix does not exist in trie then get suggestions using smaller prefix
        if len(prefix) > 0:
            return get_top_k_w_label_trie(prefix=prefix[:-1], label_trie=label_trie, k=k)
        else:
            return []


def create_pref_to_topk_dict(fdir, f_compressed, fname, k):

    i2r, i2c, smat = parallel_get_qp2q_sparse_data(fdir=fdir, compressed=f_compressed, n_jobs=16)
    sdf = SparseDataFrame(data_matrix=smat, columns=i2c, rows=i2r)

    LOGGER.info("Created sparsedataframe")

    flat_mat = np.sum(sdf.data_matrix, axis=0)  # Compute sum for each col
    LOGGER.info("Computed sum of each col")
    assert flat_mat.shape == (1, sdf.data_matrix.shape[1])

    LOGGER.info("Creating label_freq dictionary")
    label_freq = {c: int(flat_mat[0, sdf.c2i[c]]) for c in sdf.c2i}
    LOGGER.info("Created label_freq dictionary")

    all_prefs = {label[:i]: [] for label in label_freq for i in range(len(label) + 1)}
    LOGGER.info("Number of prefixes = {}".format(len(all_prefs)))

    label_trie = CharTrie()
    label_trie.update(label_freq)
    LOGGER.info("Created label trie")

    all_prefs_to_topk = {
        pref: get_top_k_w_label_trie(prefix=pref, label_trie=label_trie, k=k)
        for pref in tqdm(all_prefs)
    }

    res_dir = os.path.dirname(fname)
    Path(res_dir).mkdir(exist_ok=True, parents=True)

    with open(fname, "w") as f:
        json.dump(all_prefs_to_topk, f)

    LOGGER.info("Saved top-k dict in {}".format(fname))


def main():
    parser = argparse.ArgumentParser(
        description="Create a dict where key = prefix of a label and value = top-k suggestions matching the given prefix"
    )
    parser.add_argument("--fdir", type=str, required=True, help="path to data files")
    parser.add_argument("--f_compressed", type=int, default=0, help="are data files compressed?")
    parser.add_argument(
        "--out_fname",
        type=str,
        default="data",
        help="name of trie model file used when saving trie",
    )
    parser.add_argument(
        "--k", type=int, required=True, help="max number of suggestions to store for each prefix"
    )

    args = parser.parse_args()
    fdir = args.fdir
    f_compressed = bool(args.f_compressed)
    fname = args.out_fname
    k = args.k

    create_pref_to_topk_dict(fdir=fdir, f_compressed=f_compressed, fname=fname, k=k)


if __name__ == "__main__":
    main()
