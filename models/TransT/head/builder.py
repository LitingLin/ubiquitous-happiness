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
        head_parameters = head_config['parameters']
        if 'pos_encoded' in head_parameters:
            cls_reg_shared = False
            if 'shared' in head_parameters:
                cls_reg_shared = head_parameters['shared']
            if cls_reg_shared:
                from .gfocal_pos_encoded_mlp_shared import GFocalV2Head
            else:
                from .gfocal_pos_encoded_mlp import GFocalV2Head
        else:
            cls_reg_shared = False
            if 'shared' in head_parameters:
                cls_reg_shared = head_parameters['shared']
            if cls_reg_shared:
                from .gfocal_shared_mlp import GFocalV2Head
            else:
                from .gfocal_v2 import GFocalV2Head
        return GFocalV2Head(head_parameters['input_dim'], head_parameters['hidden_dim'], head_parameters['input_size'],
                            head_parameters['reg_max'], head_parameters['topk'], head_parameters['reg_channels'],
                            head_parameters['add_mean'])
    else:
        raise RuntimeError(f"Unknown value {head_config['type']}")
