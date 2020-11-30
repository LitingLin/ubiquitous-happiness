import hashlib
from Utils.slugify import slugify
import pickle
import os

from Dataset.Config.cache import getCacheDir


def _getCachePath(dataset):
    m = hashlib.md5()
    m.update(bytes(dataset.name, encoding='utf-8'))
    m.update(bytes(str(dataset.filters), encoding='utf-8'))
    cache_dir = getCacheDir()

    if len(dataset.filters) > 0:
        cache_file_name_prefix = '{}-filtered-{}-{}'.format(slugify(dataset.name), str(dataset.data_split), m.digest().hex())
    else:
        cache_file_name_prefix = '{}-{}-{}'.format(slugify(dataset.name), str(dataset.data_split), m.digest().hex())
    subdirectory_name = dataset.__class__.__name__
    return os.path.join(cache_dir, subdirectory_name), cache_file_name_prefix


def makeCache(dataset):
    path, cache_file_name_prefix = _getCachePath(dataset)
    if not os.path.exists(path):
        os.mkdir(path)

    cache_file_path = os.path.join(path, cache_file_name_prefix + '.p')
    tmp_file_name = cache_file_path + '.tmp'
    if os.path.exists(tmp_file_name):
        os.remove(tmp_file_name)
    if getattr(dataset, "__getstate__", None) is None:
        with open(tmp_file_name, 'wb') as fid:
            pickle.dump(dataset.__dict__, fid)
    else:
        with open(tmp_file_name, 'wb') as fid:
            pickle.dump(dataset.__getstate__(), fid)
    os.rename(tmp_file_name, cache_file_path)


def tryLoadCache(dataset):
    structure_version = dataset.structure_version
    data_version = dataset.data_version
    filters = dataset.filters

    path, cache_file_name_prefix = _getCachePath(dataset)

    cache_path = os.path.join(path, cache_file_name_prefix + '.p')
    if not os.path.exists(cache_path):
        return False
    try:
        with open(cache_path, 'rb') as fid:
            dataset_object_dict = pickle.load(fid)

        if getattr(dataset, "__setstate__", None) is None:
            if dataset_object_dict['structure_version'] == structure_version and dataset_object_dict['data_version'] == data_version and dataset_object_dict['filters'] == filters:
                dataset.__dict__.clear()
                dataset.__dict__.update(dataset_object_dict)
                return True
        else:
            dataset.__setstate__(dataset_object_dict)
            if dataset.structure_version == structure_version and dataset.data_version == data_version and dataset.filters == filters:
                return True

    except Exception as e:
        print('Failed to load cache.')
        print(str(e))

    os.remove(cache_path)
    return False
