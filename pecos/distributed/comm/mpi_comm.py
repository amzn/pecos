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
from mpi4py import MPI
from pecos.distributed.comm.abs_dist_comm import DistComm


MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()


class MPIBufferSizeExceedError(Exception):
    """Exception class for better interpretation of mpi4py buffer limit exceed errors."""

    _MSG = (
        "Object size exceeds the MPI buffer limit. One should reduce the object size."
        "For Distributed PECOS training, one could increasing the parameter for minimum number of sub-tree."
    )

    def __init__(self, msg="", *args, **kwargs):
        super().__init__(msg + self._MSG, *args, **kwargs)


class MPIComm(DistComm):
    """MPI Communicator"""

    def get_size(self):
        """Get distributed cluster size"""
        return MPI_SIZE

    def get_rank(self):
        """Get self machine rank"""
        return MPI_RANK

    def send(self, py_obj, dest, tag):
        """Point-to-point send Python object"""
        try:
            MPI_COMM.send(py_obj, dest=dest, tag=tag)
        except OverflowError:
            raise MPIBufferSizeExceedError(f"In {self.__class__}.send, ")

    def recv(self, source, tag):
        """Point-to-point receive Python object"""
        return MPI_COMM.recv(source=source, tag=tag)

    def bcast(self, py_obj, root=0):
        """Broadcast Python object from root"""
        try:
            return MPI_COMM.bcast(py_obj, root=root)
        except OverflowError:
            raise MPIBufferSizeExceedError(f"In {self.__class__}.bcast, ")

    def scatter(self, py_list, root=0):
        """Scatter Python list from root

        TODO: Scatter large array overflow failure now not captured,
        because mpi4py throw BAD TERMINATION exception at mpiexec/other low-level libs layer.
        Currently Distributed PECOS only use scatter for sending small data object so it is fine for now.
        Use at your own caution.
        """
        return MPI_COMM.scatter(py_list, root=root)

    def gather(self, py_obj, root=0):
        """Gather Python list to root

        TODO: same as scatter
        """
        return MPI_COMM.gather(py_obj, root=root)
