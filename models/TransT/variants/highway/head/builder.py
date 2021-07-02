def build_highway_head(network_config):
    head_type = network_config['head']['type']
    if head_type == 'TransT':
        from .detr import DETRHead

        hidden_dim = network_config['transformer']['hidden_dim']
        classification_branch_hidden_dim = network_config['transformer']['highway']['classification']['hidden_dim']
        regression_branch_hidden_dim = network_config['transformer']['highway']['regression']['hidden_dim']

        return DETRHead(hidden_dim + classification_branch_hidden_dim, hidden_dim + regression_branch_hidden_dim)
