from models.transformer.position_encoding import build_position_encoding
from models.transformer.deformable.deform_detr_transformer import build_deform_transformer
from models.network.siamfc.multires import SiamFCMultiResNet
from models.network.deformable_detr_tracking.siamfc_frontend.wrapper import DETRSiamFCWrapper
from models.network.deformable_detr_tracking.network import DeformableDETRTracking


def _build_backbone(config: dict):
    backbone_config = config['backbone']
    position_encoding = build_position_encoding(config)
    if backbone_config['type'] == 'alexnet':
        from models.backbone.pysot.alexnet import construct_alexnet
        backbone = construct_alexnet(output_layers=backbone_config['output_layers'])
    elif backbone_config['type'] == 'resnet50':
        from models.backbone.pysot.resnet_atrous import construct_resnet50_atrous
        backbone = construct_resnet50_atrous(output_layers=backbone_config['output_layers'])
    else:
        raise Exception(f'unsupported {backbone_config["type"]}')

    return backbone, position_encoding


def initialize_siamfc_multires_deform_atten_track(net: DeformableDETRTracking, backbone_load_pretrained=True):
    siamfc_frontend: DETRSiamFCWrapper = net.backbone
    siamfc_net: SiamFCMultiResNet = siamfc_frontend.siamfc
    postition_encoder = siamfc_frontend.position_encoder
    if hasattr(postition_encoder, 'reset_parameters'):
        postition_encoder.reset_parameters()
    backbone = siamfc_net.backbone
    if backbone_load_pretrained:
        backbone.load_pretrained()
    else:
        backbone.reset_parameters()
    for head in siamfc_net.heads:
        head.reset_parameters()
    net.reset_parameters()


def build_siamfc_multires_deform_atten_track(config: dict):
    backbone, position_encoding = _build_backbone(config)
    deformable_transformer = build_deform_transformer(config)

    output_layers = config['backbone']['output_layers']
    num_feature_levels = len(output_layers)

    if config['xcross_head'] == 'linear':
        from models.head.siamfc.siamfc import SiamFCLinearHead
        heads = [SiamFCLinearHead() for _ in range(num_feature_levels)]
    elif config['xcross_head'] == 'batch_norm':
        from models.head.siamfc.siamfc import SiamFCBNHead
        heads = [SiamFCBNHead() for _ in range(num_feature_levels)]
    else:
        raise Exception(f'unsupported xcross_head {config["xcross_head"]}')
    siamfc_net = SiamFCMultiResNet(backbone, heads)
    siamfc_frontend = DETRSiamFCWrapper(siamfc_net, position_encoding)

    return DeformableDETRTracking(siamfc_frontend, deformable_transformer, config['transformer']['num_queries'], output_layers)
