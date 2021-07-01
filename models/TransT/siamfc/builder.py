import torch.nn


def build_backbone(network_config: dict, load_pretrained=True):
    backbone_config = network_config['backbone']

    if 'load_pretrained' in backbone_config:
        load_pretrained = backbone_config['load_pretrained']

    backbone_type = backbone_config['type']
    if backbone_type == 'Alexnet-BN':
        from models.backbone.pysot.alexnet import construct_alexnet
        backbone = construct_alexnet(load_pretrained, output_layers=backbone_config['output_layers'])
    elif backbone_type == 'resnet-18-pytracking':
        from models.backbone.pytracking.resnet import resnet18
        backbone = resnet18(pretrained=load_pretrained, output_layers=backbone_config['output_layers'])
    elif backbone_type == 'resnet-18-vggm':
        from models.backbone.pytracking.resnet18_vggm import resnet18_vggmconv1
        backbone = resnet18_vggmconv1(output_layers=backbone_config['output_layers'])
    elif backbone_type == 'resnet-50-atrous':
        from models.backbone.pysot.resnet_atrous import construct_resnet50_atrous
        backbone = construct_resnet50_atrous(load_pretrained, backbone_config['output_layers'])
    elif backbone_type == 'Alexnet-SiamFC-v1':
        from models.backbone.siamfc.alexnet import AlexNetV1
        backbone = AlexNetV1()
    elif backbone_type == 'Alexnet-SiamFC-v2':
        from models.backbone.siamfc.alexnet import AlexNetV2
        backbone = AlexNetV2()
    else:
        raise NotImplementedError(f"Unknown backbone type {backbone_type}")
    return backbone


def build_neck(network_config: dict):
    neck_config = network_config['neck']
    neck_type = neck_config['type']
    if neck_type == 'XCorr':
        if network_config['type'] == 'SiamFCDualPath':
            from models.TransT.neck.dual_path.xcorr import SiamFCXCorr
        else:
            from models.TransT.neck.xcorr import SiamFCXCorr
        return SiamFCXCorr(**neck_config['parameters'])
    elif neck_type == 'DualPathXCorr':
        from models.TransT.neck.dual_path.dual_path_xcorr import SiamFCDualPathXCorr
        return SiamFCDualPathXCorr(**neck_config['parameters'])
    elif neck_type == 'SiamFCLinearXCorr':
        from models.TransT.neck.classical import SiamFCLinearNeck
        return SiamFCLinearNeck()
    elif neck_type == 'SiamFCBNXCorr':
        from models.TransT.neck.classical import SiamFCBNNeck
        return SiamFCBNNeck()
    else:
        raise NotImplementedError(f"Unknown neck type {neck_type}")


def build_head(network_config: dict):
    head_config = network_config['head']
    head_type = head_config['type']
    head_parameters = head_config['parameters']
    if head_type == 'TransT':
        from .head.detr import DETRHead
        head = DETRHead(head_parameters['input_dim'], head_parameters['hidden_dim'])
    elif head_type == 'SiamFC':
        head = torch.nn.Identity()
    else:
        raise NotImplementedError(f"Unknown head type {head_type}")
    return head


def build_siamfc(network_config: dict, load_pretrained=True):
    backbone = build_backbone(network_config, load_pretrained)
    neck = build_neck(network_config)
    head = build_head(network_config)

    from .network import SiamFCNetwork, SiamFCDualPathNetwork

    if network_config['type'] == 'SiamFC':
        return SiamFCNetwork(backbone, neck, head)
    elif network_config['type'] == 'SiamFCDualPath':
        return SiamFCDualPathNetwork(backbone, neck, head)
    else:
        raise NotImplementedError(f'Unknown network type {network_config["type"]}')
