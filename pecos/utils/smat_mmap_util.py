#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.
import numpy as np
import scipy.sparse as smat
import zipfile


def load_matrix_mmap(src):
    """
    Load matrix into memory map.
    """
    if not isinstance(src, str):
        raise ValueError("src for load_matrix must be a str")

    if src.endswith(".npy"):
        return np.load(src, mmap_mode="r+")
    elif src.endswith(".npz"):
        return _MMapSmatNpzLoader(src).load()
    else:
        raise ValueError("File name must end with .npy or .npz")


class _MMapSmatNpzLoader(object):
    """
    Class to load scipy sparse matrix saved in non-compressed npz format into memory map.
    """

    _SMAT_FORMAT_FN = "format.npy"
    _SMAT_FORMAT_TYPE_LIST = ["csc", "csr"]
    _SMAT_SHAPE_FN = "shape.npy"
    _SMAT_DATA_FN = "data.npy"
    _SMAT_INDPTR_FN = "indptr.npy"
    _SMAT_INDICES_FN = "indices.npy"

    _ZIP_HEADER_MAX_OFFSET = 480
    _NUMPY_MAGIC_PREFIX = b"\x93NUMPY"

    def __init__(self, src) -> None:
        if not isinstance(src, str):
            raise ValueError("src for load_matrix must be a str")

        if not src.endswith(".npz"):
            raise ValueError("File name must end with .npz")

        npz_f = zipfile.ZipFile(src, mode="r")
        npy_fn_list = [f"{zinfo.filename}" for zinfo in npz_f.infolist()]
        if self._SMAT_FORMAT_FN not in npy_fn_list:
            raise ValueError(f"Format file is not in .npz file: {list(npy_fn_list)}")
        if self._SMAT_SHAPE_FN not in npy_fn_list:
            raise ValueError(f"Shape file is not in .npz file: {list(npy_fn_list)}")

        self._npz_f = npz_f
        self._npy_fn_list = npy_fn_list

    def load(self):
        """
        Load from src file path.
        """
        smat_format = self._load_npy_mmap(self._SMAT_FORMAT_FN).item().decode()
        if smat_format not in self._SMAT_FORMAT_TYPE_LIST:
            raise ValueError(
                f"Invalid sparse matrix format: {smat_format}, should be in: {self._SMAT_FORMAT_TYPE_LIST}"
            )

        smat_shape = self._load_npy_mmap(self._SMAT_SHAPE_FN).tolist()
        smat_data = self._load_npy_mmap(self._SMAT_DATA_FN)
        smat_indptr = self._load_npy_mmap(self._SMAT_INDPTR_FN)
        smat_indices = self._load_npy_mmap(self._SMAT_INDICES_FN)

        smat_cls = getattr(smat, f"{smat_format}_matrix")
        return smat_cls((smat_data, smat_indices, smat_indptr), shape=smat_shape)

    def _load_npy_mmap(self, npy_fn):
        """
        Load one single npy numpy array from given npz's file pointer.
        """
        # figure out offset of .npy in .npz
        npy_info = self._npz_f.NameToInfo[npy_fn]
        assert npy_info.compress_type == 0, f"{npy_fn} is a compressed file, cannot read."
        npy_fp = self._npz_f.fp

        # search for the start of .npy
        len_numpy_prefix = len(self._NUMPY_MAGIC_PREFIX)
        for offset in range(self._ZIP_HEADER_MAX_OFFSET):
            npy_fp.seek(npy_info.header_offset + offset)
            if npy_fp.read(len_numpy_prefix) == self._NUMPY_MAGIC_PREFIX:
                break
        else:
            raise ValueError(
                f"Cannot find numpy header for .npy file: {npy_fn} from .npz file: {self._npz_f.filename}"
            )
        npy_fp.seek(npy_info.header_offset + offset)

        # read .npy header
        version = np.lib.format.read_magic(npy_fp)
        np.lib.format._check_version(version)
        shape, fortran_order, dtype = np.lib.format._read_array_header(npy_fp, version)

        # create memmap
        return np.memmap(
            self._npz_f.filename,
            dtype=dtype,
            shape=shape,
            order="F" if fortran_order else "C",
            mode="r",
            offset=npy_fp.tell(),
        )


if __name__ == "__main__":
    import sys

    npz_path = sys.argv[1]

    npz_loaded = load_matrix_mmap(npz_path)
    print(npz_loaded.shape)
