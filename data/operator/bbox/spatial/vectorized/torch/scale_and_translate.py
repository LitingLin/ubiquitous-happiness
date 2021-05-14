import torch


def bbox_scale_and_translate_vectorized(bbox, scale, input_center, output_center):
    """
    (i - input_center) * scale = o - output_center
    Args:
        bbox (torch.Tensor): (n, 4)
        scale (torch.Tensor): (n, 2)
        input_center (torch.Tensor): (n, 2)
        output_center (torch.Tensor): (n, 2)
    Returns:
        torch.Tensor: scaled torch tensor, (n, 4)
    """
    out_bbox_x1 = (bbox[..., 0] - input_center[..., 0]) * scale[..., 0] + output_center[..., 0]
    out_bbox_y1 = (bbox[..., 1] - input_center[..., 1]) * scale[..., 1] + output_center[..., 1]
    out_bbox_x2 = (bbox[..., 2] - input_center[..., 0]) * scale[..., 0] + output_center[..., 0]
    out_bbox_y2 = (bbox[..., 3] - input_center[..., 1]) * scale[..., 1] + output_center[..., 1]
    return torch.stack((out_bbox_x1, out_bbox_y1, out_bbox_x2, out_bbox_y2), dim=-1)
