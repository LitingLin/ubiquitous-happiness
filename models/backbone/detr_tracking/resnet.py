import torchvision
import torch
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from Utils.detr_misc import is_main_process


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneWrapper(nn.Module):
    def __init__(self, backbone, output_layers):
        super().__init__()
        train_backbone = True
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        num_channels_output = [64, 256, 512, 1024, 2048]
        self.num_channels_output = [num_channels_output[i] for i in output_layers]
        self.layer_getter = IntermediateLayerGetter(backbone, return_layers={f'layer{output_layer}': str(i) for i, output_layer in enumerate(output_layers)})

    def forward(self, x):
        x = self.layer_getter(x)
        x = list(x.values())
        if len(x) == 1:
            return x[0]
        else:
            return x

    def reset_parameters(self):
        pass

    def load_pretrained(self):
        pass


def construct_resnet50(load_pretrained, dilation=False, output_layers=(4,)):
    net = torchvision.models.resnet50(pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d, replace_stride_with_dilation=[False, False, dilation])
    return BackboneWrapper(net, output_layers)
