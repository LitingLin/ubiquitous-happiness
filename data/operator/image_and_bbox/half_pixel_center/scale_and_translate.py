from data.operator.bbox.half_pixel_center.validity import is_bbox_validity
from data.operator.image.half_pixel_center.scale_and_translate import tf_image_scale_and_translate
from data.operator.bbox.scale_and_translate import bbox_scale_and_translate
from data.operator.bbox.intersection import bbox_compute_intersection
import tensorflow as tf
import sys


def _generate_background_regions_rasterized(foreground_bbox, image_size):
    # foreground_bbox in standard XYXY format: [begin, end]
    # return in format: [begin, end)
    regions = [
        [-sys.float_info.max, -sys.float_info.max, foreground_bbox[0] - sys.float_info.min, sys.float_info.max],
        [foreground_bbox[0] + sys.float_info.min, -sys.float_info.max, foreground_bbox[2] - sys.float_info.min, foreground_bbox[1] - sys.float_info.min],
        [foreground_bbox[0] + sys.float_info.min, foreground_bbox[3] + sys.float_info.min, foreground_bbox[2] - sys.float_info.min, sys.float_info.max],
        [foreground_bbox[2] + sys.float_info.min, -sys.float_info.max, sys.float_info.max, sys.float_info.max]
    ]

    regions = [bbox_compute_intersection(region, [0, 0, image_size[0], image_size[1]]) for region in regions]

    # rasterization policy: round
    regions = [[round(v) for v in region] for region in regions]

    regions = [region for region in regions if is_bbox_validity(region)]

    return regions


def tf_scale_and_translate_numpy(img, output_size, scale, input_center=(0, 0), output_center=(0, 0), background_color=(0, 0, 0), kernel_type='triangle', antialias=False):
    n, h, w, c = img.shape
    assert c == len(background_color)

    output_bbox = bbox_scale_and_translate((0, 0, w - 1, h - 1), scale, input_center, output_center)
    if not is_bbox_validity(bbox_compute_intersection(output_bbox, (0, 0, output_size[0] - 1, output_size[1] - 1))):
        return tf.tile(tf.constant(background_color, shape=(1, 1, 1, c)), (n, h, w, 1))
    img = tf_image_scale_and_translate(img, output_size, scale, input_center, output_center, kernel_type, antialias)
    img = img.numpy()
    regions = _generate_background_regions_rasterized(output_bbox, output_size)
    for region in regions:
        img[:, region[1]: region[3], region[0]: region[2], :] = background_color
    return img
