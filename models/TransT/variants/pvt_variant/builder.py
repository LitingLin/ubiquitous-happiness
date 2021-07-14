from models.TransT.backbone import build_backbone
from .feature_fusion import build_pvt_feature_fusion
from models.TransT.head.builder import build_head
from .network import PVTFeatureFusionNetwork


def build_pvt_feature_fusion_tracker(network_config: dict, load_pretrained=True):
    backbone = build_backbone(network_config, load_pretrained)
    transformer = build_pvt_feature_fusion(network_config)
    head = build_head(network_config)
    template_output_stage = network_config['transformer']['backbone_output_layers']['template']['stage']
    template_output_shape = network_config['transformer']['backbone_output_layers']['template']['shape']
    search_output_stage = network_config['transformer']['backbone_output_layers']['search']['stage']
    search_output_shape = network_config['transformer']['backbone_output_layers']['search']['shape']

    enable_projection = network_config['transformer']['enable_dim_projection']
    template_output_dim = network_config['transformer']['backbone_output_layers']['template']['dim']
    search_output_dim = network_config['transformer']['backbone_output_layers']['search']['dim']
    template_hidden_dim = network_config['transformer']['hidden_dim']

    return PVTFeatureFusionNetwork(backbone, transformer, head,
                                   template_hidden_dim, enable_projection,
                                   template_output_stage, template_output_dim, template_output_shape,
                                   search_output_stage, search_output_dim, search_output_shape)
