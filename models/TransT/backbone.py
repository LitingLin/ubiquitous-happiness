def build_backbone(net_config: dict, load_pretrained=True):
    backbone_config = net_config['backbone']
    if 'parameters' in backbone_config:
        backbone_build_params = backbone_config['parameters']
        if load_pretrained and 'pretrained' in backbone_build_params:
            load_pretrained = backbone_build_params['pretrained']
            del backbone_build_params['pretrained']
    else:
        backbone_build_params = ()
    if backbone_config['type'] == 'alexnet':
        from models.backbone.pysot.alexnet import construct_alexnet
        backbone = construct_alexnet(load_pretrained, **backbone_build_params)
    elif backbone_config['type'] == 'resnet50_atrous':
        from models.backbone.pysot.resnet_atrous import construct_resnet50_atrous
        backbone = construct_resnet50_atrous(load_pretrained, **backbone_build_params)
    elif backbone_config['type'] == 'resnet50_detr':
        from models.backbone.detr_tracking.resnet import construct_resnet50
        backbone = construct_resnet50(load_pretrained, **backbone_build_params)
    elif backbone_config['type'] == 'swin_transformer':
        from models.backbone.swint.swin_transformer import build_swin_transformer_backbone
        if 'embed_dim' in backbone_build_params:
            backbone_build_params['overwrite_embed_dim'] = backbone_build_params['embed_dim']
            del backbone_build_params['embed_dim']
        backbone = build_swin_transformer_backbone(load_pretrained=load_pretrained, **backbone_build_params)
    elif backbone_config['type'] == 'resnet50_pytracking':
        from .resnet50 import resnet50
        backbone = resnet50(pretrained=load_pretrained, **backbone_build_params)
    else:
        raise Exception(f'unsupported {backbone_config["type"]}')

    return backbone
