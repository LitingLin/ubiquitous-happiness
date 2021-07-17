from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_fn=F.relu):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act_fn = act_fn
        self.apply(_init_weights)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act_fn(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
