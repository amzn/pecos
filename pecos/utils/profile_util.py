import logging


LOGGER = logging.getLogger(__name__)


class MemInfo(object):
    """Memory information class for profiling.

    Return memory infomation string if psutil is installed. Otherwise, return empty string.
    """

    _IS_PSUTIL_INSTALLED = None

    @classmethod
    def _check_psutil(cls):
        """Check whether module psutil is installed and available."""
        import importlib

        try:
            importlib.import_module("psutil")
            LOGGER.info("psutil module installed, will print memory info.")
            return True
        except ModuleNotFoundError:
            LOGGER.info("psutil module NOT installed, will NOT print memory info.")

        return False

    @classmethod
    def mem_info(cls):
        """Return memory information upon request."""
        if cls._IS_PSUTIL_INSTALLED is None:
            cls._IS_PSUTIL_INSTALLED = cls._check_psutil()

        if not cls._IS_PSUTIL_INSTALLED:
            return ""

        import os
        import psutil

        full_mem_info = psutil.Process(os.getpid()).memory_info()
        rss_in_mb = "{:.1f}".format(full_mem_info.rss / 1024**2)
        return f"RSS {rss_in_mb} MB. Full mem info: {full_mem_info}"
