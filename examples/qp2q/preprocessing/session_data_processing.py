"""The module contains functions to preprocess input
datasets into usable format."""
import gc
import gzip
import json
import logging
import multiprocessing as mp
import pathlib
import sys
from itertools import repeat
import numpy as np
import scipy.sparse as smat

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

_FUNC = None  # place holder to Pool functions.


def _worker_init(func):
    "init method to invoke Pool."
    global _FUNC
    _FUNC = func


def _worker(x):
    "init function to invoke pool"
    return _FUNC(x)


def open_file_helper(filename, compressed, mode="rt"):
    """
    Supports reading of gzip compressed or uncompressed file.

    Parameters:
    ----------
    filename : str
        Name of the file to open.
    compressed : bool
        If true, treat filename as gzip compressed.
    mode : str
        Reading mode.

    Returns:
    --------
    file handle to the opened file.
    """
    return gzip.open(filename, mode=mode) if compressed else open(filename, mode)


def _get_unique_rows_cols(filename, compressed, delim="<@@>"):
    """Function to load a json file in the format of processed session-data
    for qp2q. Then it returns dictionary of query<delim>prefix as r2i and next_query
    as c2i.
    """
    r2i = {}
    c2i = {}
    logger.info("Processing file for rows and columns: {}".format(filename))
    with open_file_helper(filename, compressed) as fp:
        for line in fp:
            try:
                pline = json.loads(line)
            except json.decoder.JSONDecodeError:
                logger.warn(f"Failed to parse: {line}")
                continue
            query_prefix = delim.join([pline["prev_query"], pline["prefix"]])
            kw = pline["next_query"]
            if query_prefix not in r2i:
                r2i[query_prefix] = 1
            if kw not in c2i:
                c2i[kw] = 1
    return r2i, c2i


def _transform_file_to_matrix_qp2q(filename, compressed, delim, g_r2i, g_c2i):
    """
    Helper Function to extract qp2q matrix from input_file which was generated
    as a output of the function parallel_process_session_data_qp2p.
    Parameters:
    ----------
    input_file: filename
        full filepath of input dataframe
    compressed: bool
        compressed or not
    delim: str
        delim separating query and prefix
    g_r2i: dictionary
        mapping for input items
    g_c2i: dictionary
        mapping of output item

    Returns:
    -------
    qp2q count matrix
    """
    rows = []
    cols = []
    data = []
    logger.info("Processing file for matrix: {}".format(filename))
    with open_file_helper(filename, compressed) as fp:
        for line in fp:
            try:
                pline = json.loads(line)
            except json.decoder.JSONDecodeError:
                logger.warn(f"Failed to parse: {line}")
                continue
            query_prefix = delim.join([pline["prev_query"], pline["prefix"]])
            kw = pline["next_query"]
            freq = 1
            data.append(freq)
            rows.append(g_r2i[query_prefix])
            cols.append(g_c2i[kw])
    matrix = smat.coo_matrix((data, (rows, cols)), shape=(len(g_r2i), len(g_c2i)), dtype=np.float32)
    return matrix


def parallel_get_qp2q_sparse_data(fdir, compressed, delim="<@@>", n_jobs=4):
    """Process session data to sparse matrix and dictionaries mapping rows and columns.

    Parameters:
    ----------
    fdir: str
        path to directory having all the files in json format
    compressed: bool
        files being compressed or not
    delim: str
        delimiter between query and prefix
    n_jobs: int
        number of threads to be used

    Returns:
    -------
    dictionary mapping row index to row names
    dictionary mapping col index to col names
    qp2q sparse csr matrix containing freq. of occurences.

    """
    if compressed:
        extension = "*.gz"
    else:
        extension = "*.json"

    if pathlib.Path(fdir).is_dir():
        files = pathlib.Path(fdir).glob(extension)
    else:
        raise ValueError(f"{fdir} is not a valid directory")

    files = [str(f) for f in files]

    logger.info("Getting qp2q unique rows and columns from files in {}".format(fdir))
    if n_jobs > 1:
        with mp.Pool(processes=n_jobs) as pool:
            dicts = pool.starmap(
                _get_unique_rows_cols,
                zip(files, repeat(compressed), repeat(delim)),
            )
    else:
        dicts = [_get_unique_rows_cols(file, compressed, delim) for file in files]

    g_r2i = {}
    g_c2i = {}
    for dic in dicts:
        g_r2i.update(dic[0])
        g_c2i.update(dic[1])

    g_i2r = {}
    g_i2c = {}
    for i, k in enumerate(g_r2i.keys()):
        g_r2i[k] = i
        g_i2r[i] = k
    for i, k in enumerate(g_c2i.keys()):
        g_c2i[k] = i
        g_i2c[i] = k

    del dicts
    gc.collect()
    logger.info("Number of unique rows: {}".format(len(g_r2i)))
    logger.info("Number of unique cols: {}".format(len(g_c2i)))
    if n_jobs > 1:
        with mp.Pool(
            processes=n_jobs,
            initializer=_worker_init,
            initargs=(
                lambda x: _transform_file_to_matrix_qp2q(x, compressed, delim, g_r2i, g_c2i),
            ),
        ) as pool:
            matrices = pool.map(_worker, files)
    else:
        matrices = [
            _transform_file_to_matrix_qp2q(x, compressed, delim, g_r2i, g_c2i) for x in files
        ]
    matrices = [m.tocsr() for m in matrices]
    qp2q_matrix = matrices[0]
    for i in range(1, len(matrices)):
        qp2q_matrix += matrices[i]

    del matrices
    gc.collect()

    return g_i2r, g_i2c, qp2q_matrix
