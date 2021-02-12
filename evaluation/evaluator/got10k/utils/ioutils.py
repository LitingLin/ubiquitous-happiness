from __future__ import absolute_import, division

import shutil


def compress(dirname, save_file):
    """Compress a folder to a zip file.
    
    Arguments:
        dirname {string} -- Directory of all files to be compressed.
        save_file {string} -- Path to store the zip file.
    """
    shutil.make_archive(save_file, 'zip', dirname)
