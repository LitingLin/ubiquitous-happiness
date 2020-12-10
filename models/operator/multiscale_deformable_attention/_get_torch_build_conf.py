import torch
from torch.utils.cpp_extension import include_paths, library_paths, CUDAExtension, COMMON_NVCC_FLAGS


def _get_torch_cuda_flags():
    return COMMON_NVCC_FLAGS


def _get_torch_cuda_archs():
    config = torch.__config__.show()
    configs = config.split('\n')
    archs = set()
    for conf in configs:
        if 'NVCC arch' in conf:
            ss = conf.split(';')
            for s in ss:
                s = s.strip()
                if s.startswith('arch='):
                    cs = s[5:].split(',')
                    for c in cs:
                        v = c.split('_')
                        archs.add(int(v[1]))
    return archs


def _get_torch_include_paths():
    return [path.replace('\\', '/') for path in include_paths(False)]


def _get_torch_library_paths():
    return [path.replace('\\', '/') for path in library_paths(False)]


def _get_torch_libraries():
    return CUDAExtension('', []).libraries

