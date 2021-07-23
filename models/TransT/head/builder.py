def build_head(network_config):
    transformer_config = network_config['transformer']
    head_config = network_config['head']
    if head_config['type'] == 'TransT':
        from .detr import DETRHead
        return DETRHead(transformer_config['hidden_dim'])
    elif head_config['type'] == 'exp-1':
        if head_config['quality_assessment_with'] == 'class':
            from .exp_1 import EXP1Head_WithClassBranch
            return EXP1Head_WithClassBranch(transformer_config['hidden_dim'])
        elif head_config['quality_assessment_with'] == 'regression':
            from .exp_1 import EXP1Head_WithRegBranch
            return EXP1Head_WithRegBranch(transformer_config['hidden_dim'])
        else:
            raise RuntimeError(f"Unknown value {head_config['quality_assessment_with']}")
    elif head_config['type'] == 'GFocal-v2':
        from .gfocal_v2 import GFocalV2Head
        head_parameters = head_config['parameters']

        position_encoding = head_parameters['position_encoding']
        enable_v2 = 'v2' in head_parameters

        if enable_v2:
            return GFocalV2Head(head_parameters['input_dim'], head_parameters['hidden_dim'], head_parameters['input_size'],
                                enable_v2, position_encoding,
                                head_parameters['reg_max'], head_parameters['v2']['topk'], head_parameters['v2']['reg_channels'],
                                head_parameters['v2']['add_mean'])
        else:
            return GFocalV2Head(head_parameters['input_dim'], head_parameters['hidden_dim'],
                                head_parameters['input_size'],
                                enable_v2, position_encoding,
                                head_parameters['reg_max'], None, None, None)
    else:
        raise RuntimeError(f"Unknown value {head_config['type']}")
