def build_head(network_config):
    transformer_config = network_config['transformer']
    head_config = transformer_config['head']
    if head_config['type'] == 'detr':
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
    else:
        raise RuntimeError(f"Unknown value {head_config['type']}")
