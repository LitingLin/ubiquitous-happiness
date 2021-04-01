import torch
import torch.nn.functional
from data.operator.bbox.scale_and_translate import bbox_scale_and_translate
from data.operator.bbox.align_corner.validity import is_bbox_validity
from data.operator.bbox.intersection import bbox_compute_intersection


def torch_scale_and_translate_align_corners_nchw(img: torch.Tensor, output_size, scale, input_center=(0, 0), output_center=(0, 0), background_color=(0, 0, 0), mode='bilinear'):
    n, c, h, w = img.shape
    dtype = img.dtype
    if not isinstance(background_color, torch.Tensor):
        background_color = torch.tensor(background_color, dtype=dtype)

    background_color = background_color.to(img.device)
    assert background_color.ndim == 1

    c_repeat_times = c / background_color.shape[0]
    assert int(c_repeat_times) == c_repeat_times
    c_repeat_times = int(c_repeat_times)
    background_color = background_color.reshape(1, background_color.shape[0], 1, 1)
    output_img = background_color.repeat(n, c_repeat_times, output_size[1], output_size[0])

    output_img = output_img.float()

    output_bbox = bbox_scale_and_translate((0, 0, w - 1, h - 1), scale, input_center, output_center)
    output_bbox = tuple(round(v) for v in output_bbox)

    in_range_bbox = bbox_compute_intersection(output_bbox, (0, 0, output_size[0] - 1, output_size[1] - 1))
    if not is_bbox_validity(in_range_bbox):
        return output_img, output_bbox

    input_bbox = bbox_scale_and_translate(in_range_bbox, (1 / scale[0], 1 / scale[1]), output_center, input_center)
    input_bbox = tuple(round(v) for v in input_bbox)
    transformed_img = torch.nn.functional.interpolate(img[:, :, input_bbox[1]: input_bbox[3] + 1, input_bbox[0]: input_bbox[2] + 1].float(), (in_range_bbox[3] - in_range_bbox[1] + 1, in_range_bbox[2] - in_range_bbox[0] + 1), mode=mode, align_corners=True)
    output_img[:, :, in_range_bbox[1]: in_range_bbox[3] + 1, in_range_bbox[0]: in_range_bbox[2] + 1] = transformed_img
    return output_img, output_bbox


def torch_scale_and_translate_align_corners(img: torch.Tensor, output_size, scale, input_center=(0, 0), output_center=(0, 0), background_color=(255, 255, 255), mode='bilinear'):
    # n, h, w, c => n, c, h, w
    img = img.permute(0, 3, 1, 2)
    output_img, output_bbox = torch_scale_and_translate_align_corners_nchw(img, output_size, scale, input_center, output_center, background_color, mode)
    # n, c, h, w => n, h, w, c
    return output_img.permute(0, 2, 3, 1), output_bbox
