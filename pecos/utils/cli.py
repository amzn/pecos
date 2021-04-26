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
def comma_separated_type(type):
    """Create a function that parses a comma-separated string into a list.

    Args:
        type (type): The type to convert each element of the string into.

    Returns:
        function: Parses a comma-separated string into a list of elements of type `type`.
    """

    return lambda x: [type(y) for y in x.split(",")]


def str2bool(x):
    """Convert a string to a boolean.

    Args:
        x (str)

    Returns:
        bool: True if `x.lower()` is 'y', 'yes', '1', 't', or 'true'; False if `x.lower()` is 'n', 'no', '0', 'f', or 'false'.

    Raises:
        ValueError: If `x.lower()` is not any of the values above.
    """

    if x.lower() in set(["y", "yes", "1", "t", "true"]):
        return True
    elif x.lower() in set(["n", "no", "0", "f", "false"]):
        return False
    else:
        raise ValueError


class SubCommand(object):
    """Interface class for building commands."""

    def __init__(self):
        pass

    @classmethod
    def add_parser(cls, super_parser):
        """Add a parser for the commands."""
        pass

    @staticmethod
    def add_arguments(parser):
        """Add arguments for the parser."""
        pass
