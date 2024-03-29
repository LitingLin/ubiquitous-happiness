/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "cuda/ms_deform_attn_cuda.h"

PYBIND11_MODULE(_C, m) {
  m.def("ms_deform_attn_cuda_forward", &ms_deform_attn_cuda_forward);
  m.def("ms_deform_attn_cuda_backward", &ms_deform_attn_cuda_backward);
}
