def build_backbone(network_config: dict, load_pretrained=True):
    backbone_config = network_config['backbone']

    if 'load_pretrained' in backbone_config:
        load_pretrained = backbone_config['load_pretrained']

    backbone_type = backbone_config['type']
    if backbone_type == 'Alexnet-BN':
        from models.backbone.alexnet_bn import AlexNet
        backbone = AlexNet()
    elif backbone_type == 'resnet-18-pytracking':
        from models.backbone.pytracking.resnet import resnet18
        backbone = resnet18(pretrained=load_pretrained)
    else:
        raise NotImplementedError(f"Unknown backbone type {backbone_type}")
    return backbone


def build_neck(network_config: dict):
    neck_config = network_config['neck']
    neck_type = neck_config['type']
    if neck_type == 'XCorr':
        from models.TransT.neck.xcorr import SiamFCXCorr
        return SiamFCXCorr(**neck_config['parameters'])
    elif neck_type == 'DualPathXCorr':
        from models.TransT.neck.dual_path_xcorr import SiamFCDualPathXCorr
        return SiamFCDualPathXCorr(**neck_config['parameters'])
    else:
        raise NotImplementedError(f"Unknown neck type {neck_type}")


def build_head(network_config: dict):
    head_config = network_config['head']
    head_type = head_config['type']
    head_parameters = head_config['parameters']
    if head_type == 'DETR':
        from .head.detr import DETRHead
        head = DETRHead(head_parameters['input_dim'], head_parameters['hidden_dim'])
    else:
        raise NotImplementedError(f"Unknown head type {head_type}")
    return head


def build_siamfc(network_config: dict, load_pretrained=True):
    backbone = build_backbone(network_config, load_pretrained)
    neck = build_neck(network_config)
    head = build_head(network_config)

    from .network import SiamFCNetwork
    return SiamFCNetwork(backbone, neck, head)
