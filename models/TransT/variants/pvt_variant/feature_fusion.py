import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math
from .feature_fusion_module import FeatureFusionEncoderLayer, FeatureFusionDecoderLayer


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class FeatureFusionNetwork(nn.Module):
    def __init__(self, template_input_dim, search_input_dim, hidden_dim, num_heads=8,
                 mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_layers=4,
                 sr_ratio=2):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # stochastic depth decay rule

        self.encoder = nn.ModuleList(
            [FeatureFusionEncoderLayer(template_input_dim if i == 0 else hidden_dim,
                                       search_input_dim if i == 0 else hidden_dim,
                                       hidden_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate,
                                       attn_drop_rate, dpr[i], sr_ratio, act_layer, norm_layer)
             for i in range(num_layers)])
        self.decoder = FeatureFusionDecoderLayer(hidden_dim, hidden_dim, hidden_dim, num_heads, mlp_ratio, qkv_bias,
                                                 qk_scale, drop_rate, attn_drop_rate, drop_path_rate, sr_ratio,
                                                 act_layer, norm_layer)
        self.apply(_init_weights)

    def forward(self, x, y, x_H, x_W, y_H, y_W):
        for encoder in self.encoder:
            x, y = encoder(x, y, x_H, x_W, y_H, y_W)
        return self.decoder(x, y, x_H, x_W, y_H, y_W)


def build_pvt_feature_fusion(network_config: dict):
    transformer_config = network_config['transformer']

    enable_dim_projection = network_config['transformer']['enable_dim_projection']

    hidden_dim = transformer_config['hidden_dim']
    if enable_dim_projection:
        template_input_dim = hidden_dim
        search_input_dim = hidden_dim
    else:
        template_input_dim = transformer_config['backbone_output_layers']['template']['dim']
        search_input_dim = transformer_config['backbone_output_layers']['search']['dim']

    num_heads = transformer_config['num_heads']
    mlp_ratio = transformer_config['mlp_ratio']
    qkv_bias = transformer_config['qkv_bias']
    drop_rate = transformer_config['drop_rate']
    attn_drop_rate = transformer_config['attn_drop_rate']
    drop_path_rate = transformer_config['drop_path_rate']
    num_layers = transformer_config['num_layers']
    sr_ratio = transformer_config['sr_ratio']

    return FeatureFusionNetwork(template_input_dim, search_input_dim, hidden_dim, num_heads, mlp_ratio, qkv_bias, None,
                                drop_rate, attn_drop_rate, drop_path_rate, num_layers=num_layers, sr_ratio=sr_ratio)
