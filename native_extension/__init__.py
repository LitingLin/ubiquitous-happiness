import torch
try:
    from native_extension._C import *
except ImportError:
    from .build import build_extension_cmake
    build_extension_cmake()
    from native_extension._C import *
