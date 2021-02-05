from .transformer import Transformer


def build_transformer(net_config: dict):
    transformer_config = net_config['transformer']
    return Transformer(
        d_model=transformer_config['hidden_dim'],
        dropout=transformer_config['dropout'],
        nhead=transformer_config['num_heads'],
        dim_feedforward=transformer_config['feed_forward']['dim'],
        num_encoder_layers=transformer_config['encoder']['num_layers'],
        num_decoder_layers=transformer_config['decoder']['num_layers']
    )
