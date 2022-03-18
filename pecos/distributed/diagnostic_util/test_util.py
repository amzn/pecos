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
from pecos.distributed.comm.abs_dist_comm import DistComm


class DummyComm(DistComm):
    """Dummy communicator on single machine, for testing purpose."""

    def __init__(self):
        self._send_recv_dict = {}

    def get_size(self):
        """Dummy distributed cluster size"""
        return 1

    def get_rank(self):
        """Dummy self machine rank"""
        return 0

    def send(self, py_obj, dest, tag):
        """Point-to-point send Python object"""
        _ = dest
        assert tag not in self._send_recv_dict, (tag, self._send_recv_dict)
        self._send_recv_dict[tag] = py_obj

    def recv(self, source, tag):
        """Point-to-point receive Python object"""
        _ = source
        py_obj = self._send_recv_dict.pop(tag)
        return py_obj

    def bcast(self, py_object, root=0):
        """Broadcast Python object from root"""
        _ = root
        return py_object

    def scatter(self, py_list, root=0):
        """Scatter Python list from root"""
        _ = root
        return py_list[0]

    def gather(self, py_obj, root=0):
        """Gather Python object as a list to root"""
        _ = root
        return [py_obj]
