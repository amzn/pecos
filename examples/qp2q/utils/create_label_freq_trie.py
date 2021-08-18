import os
import sys
import pickle
import logging
import argparse
import numpy as np
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


def create_query_freq_based_trie(fdir, f_compressed, fname):

    i2r, i2c, smat = parallel_get_qp2q_sparse_data(fdir=fdir, compressed=f_compressed, n_jobs=16)
    sdf = SparseDataFrame(data_matrix=smat, columns=i2c, rows=i2r)

    LOGGER.info("Created sparsedataframe")

    trie = CharTrie()

    LOGGER.info("Computed sum of each col")
    flat_mat = np.sum(sdf.data_matrix, axis=0)  # Compute sum for each col
    assert flat_mat.shape == (1, sdf.data_matrix.shape[1])

    LOGGER.info("Creating label_freq dictionary")
    label_freq = {c: flat_mat[0, sdf.c2i[c]] for c in sdf.c2i}
    LOGGER.info("Created label_freq dictionary")

    trie.update(label_freq)
    LOGGER.info("Created trie")

    res_dir = os.path.dirname(fname)
    Path(res_dir).mkdir(exist_ok=True, parents=True)
    with open("{}".format(fname), "wb") as trie_file:
        pickle.dump(trie, trie_file, protocol=pickle.HIGHEST_PROTOCOL)

    LOGGER.info("Saved trie in {}/{}".format(res_dir, fname))


def main():
    parser = argparse.ArgumentParser(description="Create label frequency trie")
    parser.add_argument("--fdir", type=str, required=True, help="path to data files")
    parser.add_argument("--f_compressed", type=int, default=0, help="are data files compressed?")
    parser.add_argument(
        "--out_fname",
        type=str,
        default="data",
        help="name of trie model file used when saving trie",
    )

    args = parser.parse_args()
    fdir = args.fdir
    f_compressed = bool(args.f_compressed)
    fname = args.out_fname

    create_query_freq_based_trie(fdir=fdir, f_compressed=f_compressed, fname=fname)


if __name__ == "__main__":
    main()
