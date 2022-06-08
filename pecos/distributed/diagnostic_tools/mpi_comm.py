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
from pecos.distributed.comm.mpi_comm import MPIBufferSizeExceedError, MPIComm, MPI_COMM


# Functions for testing
def test_report(func):
    """Decorator for pretty report function prints"""

    def wrap(mpi_comm, *args, **kwargs):
        if mpi_comm.get_rank() == 0:
            flush_print(f"--------------------Testing {func.__name__}-------------------------")
        func(mpi_comm, *args, **kwargs)
        if mpi_comm.get_rank() == 0:
            flush_print("\n")

    return wrap


def flush_print(msg):
    """Flush when printing a message to keep the order"""
    print(msg, flush=True)


class GenLargeObj(object):
    """Class to create a large object that overflows MPI buffer size."""

    # MPI size limit is 2^31 - 1, should specify len > that
    LARGE_ARR_LEN = int(5e8) + 1

    @classmethod
    def gen_large_list(cls):
        """Generate a large list"""
        flush_print(
            f"Generating a large list with length {cls.LARGE_ARR_LEN}...May take a while..."
        )
        arr = list(range(cls.LARGE_ARR_LEN))
        flush_print(f"Generated a large list with length {cls.LARGE_ARR_LEN}.")
        return arr


def catch_mpi_buff_size_exceed_exception(func, **kwarg):
    """Catch MPIBufferSizeExceedError from the function runs"""
    try:
        func(**kwarg)
    except MPIBufferSizeExceedError as e:
        flush_print(f"Successfully caught exception MPIBufferSizeExceedError: {e}")


# Tests
@test_report
def test_mpi_comm_echo(mpi_comm):
    """Test get rank and size"""

    if mpi_comm.get_rank() == 0:
        flush_print(f"MPI Cluster Size: {mpi_comm.get_size()}")
    MPI_COMM.Barrier()

    flush_print(f"Echo from Rank {mpi_comm.get_rank()} machine, connection OK.")
    MPI_COMM.Barrier()


@test_report
def test_send_recv(mpi_comm):
    """Test send and recv"""

    if mpi_comm.get_rank() == 0:
        send_recv_list = list(range(5))
        flush_print(f"Sending array {send_recv_list} from Rank 0...")
        for idx in range(1, mpi_comm.get_size()):
            mpi_comm.send(send_recv_list, dest=idx, tag=idx)
    else:
        send_recv_list = mpi_comm.recv(source=0, tag=mpi_comm.get_rank())
        flush_print(f"Received array {send_recv_list} from Rank 0 on Rank {mpi_comm.get_rank()}.")

    MPI_COMM.Barrier()


@test_report
def test_bcast(mpi_comm):
    """Test bcast"""

    bcast_list = None
    if mpi_comm.get_rank() == 0:
        bcast_list = list(range(5))
        flush_print(f"Broadcasting array {bcast_list} from Rank 0...")
    bcast_list = mpi_comm.bcast(bcast_list, root=0)
    flush_print(f"On Rank {mpi_comm.get_rank()}, received item {bcast_list} from broadcasting.")

    MPI_COMM.Barrier()


@test_report
def test_scatter_gather(mpi_comm):
    """Test scatter and gather"""

    scatter_list = list(range(mpi_comm.get_size()))
    if mpi_comm.get_rank() == 0:
        flush_print(f"Scattering array {scatter_list} from Rank 0...")
    recv_item = mpi_comm.scatter(scatter_list, root=0)
    flush_print(f"On Rank {mpi_comm.get_rank()}, received item {recv_item} from scattering.")

    if mpi_comm.get_rank() == 0:
        flush_print("Gathering array from all machines on Rank 0...")
    gather_list = mpi_comm.gather(recv_item, root=0)
    if mpi_comm.get_rank() == 0:
        flush_print(f"On Rank 0, gathered array: {gather_list}")

    MPI_COMM.Barrier()


# Test failures
@test_report
def test_send_recv_failure(mpi_comm):
    """Test send and recv failure capturing for large array"""

    if mpi_comm.get_rank() == 0:
        flush_print(f"Testing sending large array failure from Rank 0...")
        send_recv_list = GenLargeObj.gen_large_list()
        for idx in range(1, mpi_comm.get_size()):
            catch_mpi_buff_size_exceed_exception(
                mpi_comm.send, py_obj=send_recv_list, dest=idx, tag=idx
            )

    MPI_COMM.Barrier()


@test_report
def test_bcast_failure(mpi_comm):
    """Test bcast failure for a large array"""

    bcast_list = None
    if mpi_comm.get_rank() == 0:
        flush_print(f"Broadcasting large array from Rank 0...")
        bcast_list = GenLargeObj.gen_large_list()
        catch_mpi_buff_size_exceed_exception(mpi_comm.bcast, py_obj=bcast_list, root=0)

    MPI_COMM.Barrier()


if __name__ == "__main__":
    """Adhoc testing for connectivity and functioning of MPI communicator on cluster.

    Before testing, one should setup a cluster with >1 machines, and write IP list into `hostfile`.
    The testing is executed on main node.

    Example Command:
        mpiexec -n 2 -f hostfile python3 -m test_adhoc.adhoc_test_mpi_comm
    """
    mpi_comm = MPIComm()
    assert (
        mpi_comm.get_size() >= 2
    ), f"MPI test needs to be performed on a cluster with more than 1 machine."

    test_mpi_comm_echo(mpi_comm)

    # Test success
    test_send_recv(mpi_comm)
    test_bcast(mpi_comm)
    test_scatter_gather(mpi_comm)

    # Test failures
    test_send_recv_failure(mpi_comm)
    test_bcast_failure(mpi_comm)
