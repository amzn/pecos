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
from abc import ABC, abstractmethod


class DistComm(ABC):
    """Distributed communication class.

    An abstract class wraps around distributed machine commumincation backends.
    """

    @abstractmethod
    def get_size(self):
        """Get distributed cluster size

        Returns:
            size (int)
        """

    @abstractmethod
    def get_rank(self):
        """Get self machine rank

        Returns:
            rank (int)
        """

    @abstractmethod
    def send(self, py_obj, dest, tag):
        """Point-to-point send Python object

        Parameters:
            py_obj (any object): Any Python object sent.
            dest (int): Destination rank
            tag (int): Any information to mark the object.
        """

    @abstractmethod
    def recv(self, source, tag):
        """Point-to-point receive Python object

        Parameters:
            source (int): Source rank
            tag (int): Any information to mark the object.

        Returns:
            py_obj (any object): Any Python object sent from source with given tag.
        """

    @abstractmethod
    def bcast(self, py_obj, root=0):
        """Broadcast Python object from root

        Parameters:
            py_obj (any object): Any Python object broadcasted from root.
            root (int): Root machine rank. Optional, default=0.

        Returns:
            py_obj (any object): Any Python object broadcasted to all machines.
        """

    @abstractmethod
    def scatter(self, py_list, root=0):
        """Scatter Python list from root.

        List length should equal number of distributed machines.

        Parameters:
            py_list (List): List to scatter to all machines.
            root (int): Root machine rank. Optional, default=0.

        Returns:
            py_obj (any object): Python object scattered from the `py_list` at root to all machines.
        """

    @abstractmethod
    def gather(self, py_obj, root=0):
        """Gather Python objects into list to root.

        Parameters:
            py_obj (any object): Python object gathered from all machines.
            root (int): Root machine rank. Optional, default=0.

        Returns:
            py_list (list): List gathered from all `py_obj` into list to root.
        """
