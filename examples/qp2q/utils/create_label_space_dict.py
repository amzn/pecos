import os
import sys
import json
import logging
import argparse
from pathlib import Path

from qp2q.preprocessing.session_data_processing import parallel_get_qp2q_sparse_data
from qp2q.preprocessing.sparse_data_processing import SparseDataFrame

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def get_suffixes(query_text):

    q_token_list = query_text.split()
    suffixes = [" ".join(q_token_list[ctr:]) for ctr in range(len(q_token_list))]

    return suffixes


def create_label_dict(fdir, f_compressed, out_file, add_suffix):

    i2r, i2c, smat = parallel_get_qp2q_sparse_data(fdir=fdir, compressed=f_compressed, n_jobs=8)
    sdf = SparseDataFrame(data_matrix=smat, columns=i2c, rows=i2r)
    train_next_queries = {q: 1 for q in sdf.c2i}

    out_dir = os.path.dirname(out_file)
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    if add_suffix:
        LOGGER.info("num_train_labels {}".format(len(train_next_queries)))
        train_next_queries = {
            q_suffix: 1 for q in train_next_queries for q_suffix in get_suffixes(q)
        }
        LOGGER.info("num_train_labels with suffixes {}".format(len(train_next_queries)))
        json.dump(train_next_queries, open(out_file, "w"))
    else:
        LOGGER.info("num_train_labels {}".format(len(train_next_queries)))
        json.dump(train_next_queries, open(out_file, "w"))


def main():
    parser = argparse.ArgumentParser(description="Create a dictionary of labels in training data")
    parser.add_argument("--fdir", type=str, required=True, help="training data file directory")
    parser.add_argument(
        "--out_file", type=str, required=True, help="name of output label dictionary file"
    )
    parser.add_argument(
        "--f_compressed",
        type=int,
        choices=[1, 0],
        default=0,
        help="are files in train data directory compressed. 1 for yes and 0 for no",
    )
    parser.add_argument(
        "--add_suffix",
        type=int,
        choices=[1, 0],
        default=0,
        help="Add suffixes of all labels in the label dictionary. 1 for yes and 0 for no",
    )

    args = parser.parse_args()
    _fdir = args.fdir
    _out_file = args.out_file
    _f_compressed = bool(args.f_compressed)
    _add_suffix = bool(args.add_suffix)

    create_label_dict(
        fdir=_fdir, out_file=_out_file, f_compressed=_f_compressed, add_suffix=_add_suffix
    )


if __name__ == "__main__":
    main()
