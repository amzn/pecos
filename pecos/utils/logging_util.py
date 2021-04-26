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
import logging

log_levels = {
    0: logging.ERROR,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG,
}


def setup_logging_config(level=1):
    """Configure logging module.

    Args:
        level (int, optional): verbose level, 0 for ERROR, 1 for WARNING (default), 2 for INFO, 3 for DEBUG
    """

    try:
        logging_level = log_levels[level]
    except KeyError:
        raise ValueError(f"expect level to be one of {log_levels.keys()}, but got {level}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging_level,
    )
