from models.backbone.swint.swin_transformer import SwinTransformer, SwinTransformerBlock
import torch
from torch import nn
from models.TransT.module.swin_feature_fusion import InterPatchCrossAttention, CrossAttention, CrossAttentionDecoder
import copy
from timm.models.layers import trunc_normal_


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class CrossAttentionModule(nn.Module):
    def __init__(self, cross_attentions, self_attentions, z_size, x_size):
        super(CrossAttentionModule, self).__init__()
        self.z_size = z_size
        self.x_size = x_size
        if isinstance(cross_attentions, (list, tuple)):
            self.cross_attn = nn.ModuleList(cross_attentions)
        else:
            self.cross_attn = cross_attentions
        if isinstance(self_attentions, (list, tuple)):
            self.self_attn = nn.ModuleList(self_attentions)
        else:
            self.self_attn = self_attentions

    def forward(self, z, x):
        if self.cross_attn is not None:
            if isinstance(self.cross_attn, nn.ModuleList):
                for layer in self.cross_attn:
                    z, x = layer(z, x)
            else:
                z, x = self.cross_attn(z, x)
        if self.self_attn is not None:
            if isinstance(self.self_attn, nn.ModuleList):
                for layer in self.self_attn:
                    layer.H = self.z_size[0]
                    layer.W = self.z_size[1]
                    z = layer(z)
                    layer.H = self.x_size[0]
                    layer.W = self.x_size[1]
                    x = layer(x)
            else:
                self.self_attn.H = self.z_size[0]
                self.self_attn.W = self.z_size[1]
                z = self.self_attn(z)
                self.self_attn.H = self.x_size[0]
                self.self_attn.W = self.x_size[1]
                x = self.self_attn(x)
        return z, x


class SwinTransformerX(nn.Module):
    def __init__(self, swin_transformer: SwinTransformer,
                 template_branch_stage_parameters,
                 search_branch_stage_parameters,
                 stage_injection_parameters):
        super(SwinTransformerX, self).__init__()

        self.z_involves = template_branch_stage_parameters['involves']
        self.x_involves = search_branch_stage_parameters['involves']

        for index_of_stage, stage_injection_parameter in stage_injection_parameters.items():
            cross_attentions = None
            if 'cross_attention' in stage_injection_parameter:
                parameter = stage_injection_parameter['cross_attention']
                if parameter['type'] == 'window_cross_attention':
                    cross_attentions = InterPatchCrossAttention(**parameter['parameters'])
                elif parameter['type'] == 'cross_attention':
                    cross_attentions = CrossAttention(**parameter['parameters'])
                else:
                    raise NotImplementedError(f'Unknown layer type {parameter["type"]}')
                if 'num_layers' in parameter:
                    cross_attentions = [copy.deepcopy(cross_attentions) for _ in range(parameter['num_layers'])]

            self_attentions = None
            if 'self_attention' in stage_injection_parameter:
                parameter = stage_injection_parameter['self_attention']
                if parameter['type'] == 'swin_transformer_block':
                    self_attentions = SwinTransformerBlock(**parameter['parameters'])
                else:
                    raise NotImplementedError(f'Unknown layer type {parameter["type"]}')
                if 'num_layers' in parameter:
                    self_attentions = [copy.deepcopy(self_attentions) for _ in range(parameter['num_layers'])]

            cross_attn_module = CrossAttentionModule(cross_attentions, self_attentions, stage_injection_parameter['z_size'],
                                 stage_injection_parameter['x_size'])
            if 'num_layers' in stage_injection_parameter:
                cross_attn_module = nn.ModuleList([copy.deepcopy(cross_attn_module) for _ in range(stage_injection_parameter['num_layers'])])
            if cross_attentions is not None or self_attentions is not None:
                self.add_module(f'stage{index_of_stage}_cross_attn', cross_attn_module)
            if 'decoder' in stage_injection_parameter:
                parameter = stage_injection_parameter['decoder']
                if parameter['type'] == 'cross_attention_decoder':
                    cross_attn_decoder = CrossAttentionDecoder(**parameter['parameters'])
                else:
                    raise NotImplementedError(f'Unknown layer type {parameter["type"]}')
                self.add_module(f'stage{index_of_stage}_cross_attn_decoder', cross_attn_decoder)

        self.apply(_init_weights)
        self.swin_transformer_stages = copy.deepcopy(swin_transformer.stages)

    def forward(self, z, x):
        _, _, z_H, z_W = z.shape
        _, _, x_H, x_W = x.shape
        for index_of_stage in range(len(self.swin_transformer_stages)):
            layer = self.swin_transformer_stages[index_of_stage]
            if index_of_stage in self.z_involves:
                z, z_H, z_W = layer(z, z_H, z_W)
            if index_of_stage in self.x_involves:
                x, x_H, x_W = layer(x, x_H, x_W)
            if hasattr(self, f'stage{index_of_stage}_cross_attn'):
                cross_attn = getattr(self, f'stage{index_of_stage}_cross_attn')
                if isinstance(cross_attn, nn.ModuleList):
                    for cross_attn_layer in cross_attn:
                        z, x = cross_attn_layer(z, x)
                else:
                    z, x = cross_attn(z, x)
            if hasattr(self, f'stage{index_of_stage}_cross_attn_decoder'):
                cross_attn_decoder = getattr(self, f'stage{index_of_stage}_cross_attn_decoder')
                x = cross_attn_decoder(z, x)
        return x


class SwinTransformerXTracker(nn.Module):
    def __init__(self, backbone, head):
        super(SwinTransformerXTracker, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, samples):
        z, x = samples
        x = self.backbone(z, x)
        return self.head(x.unsqueeze(0))

    @torch.no_grad()
    def template(self, z):
        return z

    @torch.no_grad()
    def track(self, z, x):
        return self.forward((z, x))


def _get_config_in_order(name, conf1, conf2):
    if name in conf1:
        return conf1[name]
    if name in conf2:
        return conf2[name]
    return None


def _parse_parameters(parameters, inferred_parameters, window_size, transformer_parameters):
    parsed_parameter_dict = {}
    parsed_parameter_dict['type'] = parameters['type']
    if 'num_layers' in parameters:
        parsed_parameter_dict['num_layers'] = parameters['num_layers']

    cross_attention_class_parameter = {'dim': inferred_parameters[0]}

    if parsed_parameter_dict['type'] == 'window_cross_attention' or parsed_parameter_dict['type'] == 'cross_attention' or parsed_parameter_dict['type'] == 'cross_attention_decoder':
        cross_attention_class_parameter['z_size'] = inferred_parameters[1]
        cross_attention_class_parameter['x_size'] = inferred_parameters[2]
    if parsed_parameter_dict['type'] == 'window_cross_attention':
        cross_attention_class_parameter['patch_size'] = window_size
    elif parsed_parameter_dict['type'] == 'swin_transformer_block':
        cross_attention_class_parameter['window_size'] = window_size

    num_heads = _get_config_in_order('num_heads', parameters, transformer_parameters)
    if num_heads is None:
        num_heads = inferred_parameters[3]
    cross_attention_class_parameter['num_heads'] = num_heads

    def _try_get_parameter(name):
        parameter = _get_config_in_order(name, parameters, transformer_parameters)
        if parameter is not None:
            cross_attention_class_parameter[name] = parameter

    _try_get_parameter('mlp_ratio')
    _try_get_parameter('qkv_bias')
    _try_get_parameter('qk_scale')
    _try_get_parameter('drop')
    _try_get_parameter('attn_drop')
    _try_get_parameter('drop_path')
    _try_get_parameter('act_layer')
    _try_get_parameter('norm_layer')
    if 'act_layer' in cross_attention_class_parameter:
        if cross_attention_class_parameter['act_layer'] == 'relu':
            cross_attention_class_parameter['act_layer'] = nn.ReLU
        elif cross_attention_class_parameter['act_layer'] == 'gelu':
            cross_attention_class_parameter['act_layer'] = nn.GELU
        else:
            raise NotImplementedError(cross_attention_class_parameter["act_layer"])
    if 'norm_layer' in cross_attention_class_parameter:
        if cross_attention_class_parameter['norm_layer'] == 'layer_norm':
            cross_attention_class_parameter['norm_layer'] = nn.LayerNorm
        else:
            raise NotImplementedError(cross_attention_class_parameter["norm_layer"])

    parsed_parameter_dict['parameters'] = cross_attention_class_parameter
    return parsed_parameter_dict


def build_swin_transformer_x_tracker(network_config: dict, load_pretrained=True):
    from models.backbone.swint.swin_transformer import build_swin_transformer_backbone
    swin_transformer = build_swin_transformer_backbone(network_config['backbone']['parameters']['name'], load_pretrained=load_pretrained)

    backbone_cross_attention_injection_parameters = network_config['transformer']['backbone_cross_attention_injection']

    template_branch_involves_indices = network_config['transformer']['template_branch_involves']
    search_branch_involves_indices = network_config['transformer']['search_branch_involves']

    assert 0 in template_branch_involves_indices and 0 in search_branch_involves_indices

    stage_parameters = []

    z_size = network_config['data']['template_size']
    x_size = network_config['data']['search_size']

    z_last_size = z_size
    x_last_size = x_size

    first_stage_patch_size = swin_transformer.stages[0].pre_stage.patch_size
    window_size = swin_transformer.stages[0].window_size

    for index_of_stage in range(len(swin_transformer.stages)):
        stage = swin_transformer.stages[index_of_stage]
        dim = stage.blocks[0].dim
        num_heads = stage.blocks[0].num_heads
        if index_of_stage in template_branch_involves_indices:
            if index_of_stage == 0:
                z_last_size = ((z_last_size[0] + first_stage_patch_size[0] - 1) // first_stage_patch_size[0], (z_last_size[1] + first_stage_patch_size[1] - 1) // first_stage_patch_size[1])
            else:
                z_last_size = ((z_last_size[0] + 1) // 2, (z_last_size[1] + 1) // 2)
        if index_of_stage in search_branch_involves_indices:
            if index_of_stage == 0:
                x_last_size = ((x_last_size[0] + first_stage_patch_size[0] - 1) // first_stage_patch_size[0], (x_last_size[1] + first_stage_patch_size[1] - 1) // first_stage_patch_size[1])
            else:
                x_last_size = ((x_last_size[0] + 1) // 2, (x_last_size[1] + 1) // 2)
        stage_parameters.append((dim, z_last_size, x_last_size, num_heads))

    stage_injection_parameters = {}
    for stage_name, stage_injection_parameter in backbone_cross_attention_injection_parameters.items():
        assert stage_name.startswith('stage_')
        stage_index = int(stage_name[len('stage_'):])
        parsed_parameter_dict = {}

        parsed_parameter_dict['z_size'] = stage_parameters[stage_index][1]
        parsed_parameter_dict['x_size'] = stage_parameters[stage_index][2]

        if 'cross_attention' in stage_injection_parameter:
            parsed_parameter_dict['cross_attention'] = _parse_parameters(stage_injection_parameter['cross_attention'], stage_parameters[stage_index], window_size, network_config['transformer'])
        if 'self_attention' in stage_injection_parameter:
            parsed_parameter_dict['self_attention'] = _parse_parameters(stage_injection_parameter['self_attention'], stage_parameters[stage_index], window_size, network_config['transformer'])
        if 'num_layers' in stage_injection_parameter:
            parsed_parameter_dict['num_layers'] = stage_injection_parameter['num_layers']
        if 'decoder' in stage_injection_parameter:
            parsed_parameter_dict['decoder'] = _parse_parameters(stage_injection_parameter['decoder'], stage_parameters[stage_index], window_size, network_config['transformer'])

        stage_injection_parameters[stage_index] = parsed_parameter_dict
    backbone = SwinTransformerX(swin_transformer,
                                {'involves': template_branch_involves_indices},
                                {'involves': search_branch_involves_indices},
                                stage_injection_parameters)
    del swin_transformer
    network_config['transformer']['hidden_dim'] = dim
    from models.TransT.head.builder import build_head
    head = build_head(network_config)
    return SwinTransformerXTracker(backbone, head)
