"""
This module contains the SparseDataFrame object
"""
import gc
import logging
import os
import pathlib
import pickle
import random
import copy
from collections import Counter
from multiprocessing import Pool

import numpy as np
import scipy.sparse as smat


SEED = 111
np.random.seed(SEED)
random.seed(SEED)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

_FUNC = None  # place holder to Pool functions.


def worker_init(func):
    "init method to invoke Pool."
    global _FUNC
    _FUNC = func


def worker(x):
    "init function to invoke pool"
    return _FUNC(x)


def parallel_array_apply(arr, func, threads=8, len_threshold=None):
    """
    Apply function in parallel to array.

    Parameters:
    ----------
    arr: np.array
        input array
    func: function vectorized
        function to be applied
    threads: int
        number of processes to use
    len_threshold: int
        no parallelism for arr len less than len_threshold

    Returns:
    -------
    Function applied to the array.
    """
    if len_threshold is None:
        len_threshold = threads
    if len(arr) < len_threshold or threads == 1:
        return func(arr)
    chunks = np.array_split(arr, threads)
    with Pool(processes=threads, initializer=worker_init, initargs=(func,)) as pool:
        results = pool.map(worker, chunks)
    return np.concatenate(results)


class SparseDataFrame(object):
    """
    The goal for this class object is to mimic a
    sparse dataframe (limited functionality as of now). It is basically a sparse
    matrix that can be indexed by other indenitifiers,
    and not just row and column numbers.
    For instance if the rows are asins and columns are queries, then we can get a
    sparse sub-matrix as follows:

    R = ['B07K569FH8', 'B075PK7V2T']
    C = ['revvl 2 plus case', 'radio for 2011 gmc sierra']

    SD[R,C] will give the corresponding sub-matrix, where SD is the sparse data frame.
    """

    def __init__(self, data_matrix, columns, rows):
        """
        Parameters
        ------------
        data: sparse csr matrix
            input data matrix
        columns: list/iterable
            iterable with column names numbered from index 0 to end
        rows: list/iterable
            iterable with row names numbered from index 0 to end
        Returns
        ------------
        Nothing
        """
        assert isinstance(
            data_matrix, smat.csr_matrix
        ), "data_matrix has to be scipy sparse csr matrix"
        self.data_matrix = data_matrix  # the actual sparse matrix
        self.i2r = {}  # indextorow
        self.r2i = {}  # rowtoindex
        self.i2c = {}  # indextocolumn
        self.c2i = {}  # columntoindex
        self.shape = self.data_matrix.shape
        assert data_matrix.shape[0] == len(
            rows
        ), "Number of rows are not equal to data_matrix dim 0 shape"
        assert data_matrix.shape[1] == len(
            columns
        ), "Number of columns are not equal to data_matrix dim 1 shape"
        for i in range(len(rows)):
            self.i2r[i] = rows[i]
            self.r2i[rows[i]] = i
        for i in range(len(columns)):
            self.i2c[i] = columns[i]
            self.c2i[columns[i]] = i

    def save(self, folder_path):
        """
        Function to save sparse data frame in the goven folder path.
        It creates the folder if it does not exists
        Parameters
        -------------
        folder_path: str
            folder path to save the object in
        """
        os.makedirs(folder_path, exist_ok=True)
        i2rpath = str(pathlib.Path(folder_path, "i2r.pkl"))
        i2cpath = str(pathlib.Path(folder_path, "i2c.pkl"))
        with open(i2rpath, "wb") as f:
            pickle.dump(self.i2r, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(i2cpath, "wb") as f:
            pickle.dump(self.i2c, f, protocol=pickle.HIGHEST_PROTOCOL)

        matrix_path = str(pathlib.Path(folder_path, "matrix.npz"))
        smat.save_npz(matrix_path, self.data_matrix)

    @classmethod
    def load(cls, folder_path):
        """
        Method to load an object of the class given a folder_path

        Parameters
        ------------
        folder_path: str
            path containing a saved sdf

        Returns
        -------------
        An object of the class
        """
        data_matrix = smat.load_npz(str(pathlib.Path(folder_path, "matrix.npz")))
        i2r = pickle.load(open(str(pathlib.Path(folder_path, "i2r.pkl")), "rb"))
        i2c = pickle.load(open(str(pathlib.Path(folder_path, "i2c.pkl")), "rb"))
        return cls(data_matrix=data_matrix, columns=i2c, rows=i2r)

    @classmethod
    def load_from_dataframe(cls, dataframe, col_row, col_col, col_val):
        """
        Class method to convert a pandas
        dataframe into a sparse dataframe
        Parameters
        ------------
        dataframe: pandas dataframe
        input dataframe
        col_row: str
            the column mapped to the rows of the matrix
        col_col: str
            the column in the dataframe mapped to the columns of the matrix
        col_val: str
            the column in the dataframe which has the values
        Returns
        ------------
        parseDataFrame object, with the rows
        as col_row and columns as col_col
        and the corresponding values as col_val
        """
        pdata = dataframe.loc[dataframe[col_val] != 0]
        pdata = pdata.set_index([col_row, col_col])
        mat = smat.csr_matrix((pdata[col_val], (pdata.index.codes[0], pdata.index.codes[1])))
        return cls(
            data_matrix=mat,
            columns=pdata.index.levels[1],
            rows=pdata.index.levels[0],
        )

    def __getitem__(self, key):
        """
        Custom getitem method:
        self['hello','goodbye'] returns the value stored in the matrix
        corresponding to row 'hello'
        and column 'goodbye'
        self[['hello','world'],['good','indexing']] returns the sparse 2 by-2
        SparseDataFrame corresponding to rows
        ['hello','world'] and columns ['good','indexing']
        self[['hello'],:] will fetch all the columns corresponding to the row
        ['hello']
        Parameters
        ------------
        key: tuple
            tuple of row and column iterables
            sub-dataframe
        Returns
        ------------
        a sparse dataframe or a single scalar value
        """
        k0 = np.array(key[0]).reshape(-1)
        k1 = np.array(key[1]).reshape(-1)
        out_data, rows, columns = self.get_submatrix_data(k0, k1)
        if isinstance(out_data, smat.csr_matrix):
            return SparseDataFrame(data_matrix=out_data, rows=rows, columns=columns)
        return out_data

    def get_submatrix_data(self, rows, columns):
        """
        This method can be useful,
        if we only want to get the corresponding csr matrix.
        Parameters
        -------------
        rows: list(str)
            rows for which we need to fetch
            sub-matrix
        columns: list(str)
            rows for which we need to fetch
            sub-matrix
        Returns
        -------------
        output csr matrix/individual scalar value
        row iterator/dict
        col iterator/dict
        """
        if rows is None or isinstance(rows, slice):
            rs = np.arange(self.data_matrix.shape[0])
            rows = self.i2r
        elif rows[0] is None or isinstance(rows[0], slice):
            rs = np.arange(self.data_matrix.shape[0])
            rows = self.i2r
        else:
            rs = [self.r2i[i] for i in rows]
            rs = np.array(rs)
        if columns is None or isinstance(columns, slice):
            cs = np.arange(self.data_matrix.shape[1])
            columns = self.i2c
        elif columns[0] is None or isinstance(columns[0], slice):
            cs = np.arange(self.data_matrix.shape[1])
            columns = self.i2c
        else:
            cs = [self.c2i[i] for i in columns]
            cs = np.array(cs)
        if len(rs) == 1 and len(cs) == 1:
            out_data = self.data_matrix[rs[0], cs[0]]
        else:
            out_rows = self.data_matrix[rs, :]
            out_data = out_rows[:, cs]
        return out_data, rows, columns

    def get_index2rows(self, indices=None):
        """
        Parameters
        ------------
        indices: list/iterable
            indices that corresponds to rows
        Returns
        ------------
        row names corresponding to the indices
        """
        if indices is None:
            indices = np.arange(self.data_matrix.shape[0])
        return [self.i2r[i] for i in indices]

    def get_index2columns(self, indices=None):
        """
        Parameters
        ------------
        indices: list/iterable
            indices that corresponds to columns
        Returns
        ------------
        column names corresponding to the indices
        """
        if indices is None:
            indices = np.arange(self.data_matrix.shape[1])
        return [self.i2c[i] for i in indices]

    def get_rows2index(self, rows=None):
        """
        Parameters
        ------------
        indices: list/iterable
            list of row names
        Returns
        ------------
        indices corresponding to those rows
        """
        if rows is None:
            return np.arange(self.data_matrix.shape[0])
        return [self.r2i[r] for r in rows]

    def get_columns2index(self, columns=None):
        """
        Parameters
        ------------
        indices: list/iterable
            list of column names
        Returns
        ------------
        indices corresponding to those columns
        """
        if columns is None:
            return np.arange(self.data_matrix.shape[1])
        return [self.c2i[c] for c in columns]

    def transpose(self):
        """
        Out of place transpose of the whole object.
        Returns
        -----------
        sparse data-frame object transposed
        """
        data_matrix = smat.csc_matrix(self.data_matrix).transpose()
        return SparseDataFrame(data_matrix=data_matrix, rows=self.i2c, columns=self.i2r)

    def transpose_(self):
        """
        Inplace transpose operation
        Returns
        -----------
        void
        """
        self.data_matrix = smat.csc_matrix(self.data_matrix).transpose()
        self.i2r, self.i2c = self.i2c, self.i2r
        self.r2i, self.c2i = self.c2i, self.r2i

    def set_values(self, rows, columns, values, lil_matrix=False, merge_type="replace"):
        """
        Set the given indices with the given 'values'
        It might be more efficient to convert the matrix m in lil_matrix format
        before doing a lot of these operations
        for instance rows = [1,2] and columns = [4,5] means entries (1,4)
        and (2,5) will be edited
        values: the actual values to be written into the supplied indices
        Parameters
        -------------
        rows: list/iterable
            indices denoting rows
        columns: list/iterables
            indices denoting column
        lil_matrix: bool
            boolean value denoting whether to covert back and forth to
            lil_matrix while changing sparsity pattern
        merge_type: str
            'replace': original values are replaced
            'min' : minimum of the values are kept in place of collision
            'max' : maximum of the values are kept in place of collision
            'add' : add the two values
        """
        if lil_matrix:
            self.data_matrix = smat.lil_matrix(
                self.data_matrix
            )  # this may help to speed things up if we are modifying a lot
            # of values all at once
        row_indices = self.get_rows2index(rows)
        column_indices = self.get_columns2index(columns)
        prev_values = np.array(
            [self.data_matrix[row_indices[i], column_indices[i]] for i in range(len(row_indices))]
        )
        if merge_type == "replace":
            write_values = values
        elif merge_type == "max":
            write_values = np.maximum(values, prev_values)
        elif merge_type == "min":
            write_values = np.minimum(values, prev_values)
        elif merge_type == "add":
            write_values = values + prev_values
        else:
            raise NotImplementedError
        self.data_matrix[row_indices, column_indices] = write_values
        if lil_matrix:
            self.data_matrix = smat.csr_matrix(self.data_matrix)

    def shape(self):
        """
        Returns the shape of the data_matrix
        """
        return self.data_matrix.shape

    def join(self, sdf, merge_type="replace", threads=1):
        """
        Parameters:
        ----------
        sdf: SparseDataFrame
            another sparse dataframe
        merge_type: str
            'replace': original values are replaced
            'min' : minimum of the values are kept in place of collision
            'max' : maximum of the values are kept in place of collision
            'add' : add the two values
        threads: int
            number of processes to use.

        Returns:
        -------
        The values in sdf are used to replace/add to the corresponding
        values in the original dataframe
        """
        new_i2r, new_r2i = self._merge_index_dictionaries(self.r2i, sdf.r2i)
        new_i2c, new_c2i = self._merge_index_dictionaries(self.c2i, sdf.c2i)
        self.data_matrix = smat.coo_matrix(self.data_matrix)
        sdf.data_matrix = smat.coo_matrix(
            sdf.data_matrix
        )  # this will be changed back to original format
        gc.collect()
        data_dic_self = self._map_rows_cols_join(
            self.data_matrix,
            self.i2r,
            self.i2c,
            new_r2i,
            new_c2i,
            threads=threads,
        )
        LOGGER.info("Finished mapping rows and columns from self.")
        del self.data_matrix
        gc.collect()

        data_dic_sdf = self._map_rows_cols_join(
            sdf.data_matrix,
            sdf.i2r,
            sdf.i2c,
            new_r2i,
            new_c2i,
            threads=threads,
        )
        LOGGER.info("Finished mapping rows and columns from sdf.")
        gc.collect()

        if merge_type == "replace":
            data_dic_self.update(data_dic_sdf)
        else:
            data_dic_self = Counter(data_dic_self)
            data_dic_sdf = Counter(data_dic_sdf)
            if merge_type == "add":
                data_dic_self += data_dic_sdf
            elif merge_type == "min":
                data_dic_self &= data_dic_sdf
            elif merge_type == "max":
                data_dic_self |= data_dic_sdf
            else:
                raise NotImplementedError("Merge type not implemented.")
        del data_dic_sdf
        gc.collect()
        row, col = zip(*list(data_dic_self.keys()))
        values = list(data_dic_self.values())
        del data_dic_self
        gc.collect()
        self.data_matrix = smat.coo_matrix(
            (values, (row, col)), shape=(len(new_i2r), len(new_i2c))
        ).tocsr()
        sdf.data_matrix = sdf.data_matrix.tocsr()  # changed back to original format
        self.i2r = new_i2r
        self.r2i = new_r2i
        self.c2i = new_c2i
        self.i2c = new_i2c
        self.shape = self.data_matrix.shape
        gc.collect()

    def vstack(self, sdf):
        """
        In place version of vstack. The columns of sdf are assumed
        to be a subset of columns of self.
        If there is a row in sdf which is already present in self,
        it is ignored and not added to self.
        A usecase of this is where self contains the phrasedocs
        for original + expanded + pecos inferred and
        sdf points to predictions on unseen queries. In this use case,
        the columns (asins) are assumed to be the
        same whereas unseen queries (by definition) dont have phrase doc scores.
        If there are unseen queries which
        are already present in the set of (original + expanded + pecos inferred),
        we ignore them and keep the original
        data unchanged.

        Parameters
        ------------
        sdf (sparseDataFrame): The sparse dataframe to vstack.
        Sife Effect
        ------------
        Inplace update of self such that the resulting sparse matrix is a
        vertical stack of self and sdf.
        Returns
        -----------
        Nothing
        """
        num_rows_original = len(self.i2r)

        # Get the rows which are not overlapping in the same order as in original.
        non_overlapping_rows = []
        non_overlapping_indices = []
        for i in range(len(sdf.i2r)):
            row = sdf.i2r[i]
            if row not in self.r2i:
                non_overlapping_rows.append(row)
                non_overlapping_indices.append(i)

        # Append the rows from sdf to the end of the rows of self.data_matrix
        # by updating the mapping.
        for i, r in enumerate(non_overlapping_rows):
            self.i2r[num_rows_original + i] = r
            self.r2i[r] = num_rows_original + i

        non_overlapping_m = sdf.data_matrix[non_overlapping_indices].tocoo()
        # Map the columns to the indices of self.
        col_indices = [self.c2i[sdf.i2c[i]] for i in non_overlapping_m.col]
        updated_m = smat.csr_matrix(
            (non_overlapping_m.data, (non_overlapping_m.row, col_indices)),
            shape=(len(non_overlapping_rows), len(self.c2i)),
        )
        self.data_matrix = smat.vstack([self.data_matrix, updated_m])

    @staticmethod
    def _merge_index_dictionaries(dic_one, dic_two):
        """
        Merge dic_one and dic_two and create new index dictionaries.
        """
        new_o2i = copy.deepcopy(dic_one)
        new_o2i.update(dic_two)
        new_i2o = dict()
        rows = new_o2i.keys()
        for i, r in enumerate(rows):
            new_o2i[r] = i
            new_i2o[i] = r
        del rows
        gc.collect()
        return new_i2o, new_o2i

    @staticmethod
    def _map_rows_cols_join(data_matrix, i2r, i2c, new_r2i, new_c2i, threads=1):
        """
        Helper function to map rows and columns to new indices,
        during join.
        """
        global _FUNC
        row_mapper = np.vectorize(lambda x: new_r2i[i2r[x]])
        column_mapper = np.vectorize(lambda x: new_c2i[i2c[x]])
        LOGGER.info("Created vectorized mappings of rows and columns")
        rows, cols = (
            parallel_array_apply(data_matrix.row, row_mapper, threads=threads),
            parallel_array_apply(data_matrix.col, column_mapper, threads=threads),
        )
        _FUNC = None
        gc.collect()
        keys = tuple(zip(rows, cols))
        del rows, cols
        gc.collect()
        data_dic = dict()
        data_dic.update(zip(keys, data_matrix.data))
        del keys
        gc.collect()
        return data_dic
