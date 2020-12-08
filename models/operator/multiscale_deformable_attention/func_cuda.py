
from torch.autograd import Function
from torch.autograd.function import once_differentiable


try:
    from .C import ms_deform_attn_cuda_forward, ms_deform_attn_cuda_backward
except ImportError:
    from .build import build_extension_cmake

    build_extension_cmake()
    from .C import ms_deform_attn_cuda_forward, ms_deform_attn_cuda_backward


class MSDeformAttnCUDAFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = ms_deform_attn_cuda_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            ms_deform_attn_cuda_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
