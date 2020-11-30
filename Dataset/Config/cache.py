import os

_cache_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'cache'))


def getCacheDir():
    global _cache_dir
    return _cache_dir


def setCacheRootDir(path: str):
    global _cache_dir
    _cache_dir = path
