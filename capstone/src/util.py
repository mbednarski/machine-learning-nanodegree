from __future__ import print_function, division

import errno
import os


def ensure_dir_exists(path):
    """
    Ensures that given directory exists. Equivalent of Python 3 os.mkdirs(exists_ok=True)
    :param path: directory
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
