from data.operator.image.align_corner.scale_and_translate import tf_image_scale_and_translate_align_corners
from data.operator.bbox.spatial.scale_and_translate import bbox_scale_and_translate
from data.operator.bbox.validity import bbox_is_valid
from data.operator.bbox.intersection import bbox_get_intersection
from data.operator.bbox.transform.rasterize.aligned import bbox_rasterize_aligned
from data.operator.bbox.spatial.utility.aligned.image import get_image_bounding_box
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

    regions = [bbox_get_intersection(region, get_image_bounding_box(image_size)) for region in regions]
    regions = [region for region in regions if bbox_is_valid(region)]

    regions = [bbox_rasterize_aligned(region) for region in regions]

    # in interval: [begin, end)
    regions = [(region[0], region[1], region[2] + 1, region[3] + 1) for region in regions]
    return regions


def tf_scale_and_translate_numpy(img, output_size, scale, input_center=(0, 0), output_center=(0, 0), background_color=(0, 0, 0), kernel_type='keyscubic', antialias=False):
    n, h, w, c = img.shape
    assert c == len(background_color)

    output_bbox = bbox_scale_and_translate(get_image_bounding_box((w, h)), scale, input_center, output_center)
    if not bbox_is_valid(bbox_get_intersection(output_bbox, get_image_bounding_box(output_size))):
        return tf.tile(tf.constant(background_color, shape=(1, 1, 1, c)), (n, h, w, 1)).numpy()
    img = tf_image_scale_and_translate_align_corners(img, output_size, scale, input_center, output_center, kernel_type, antialias)
    img = img.numpy()
    regions = _generate_background_regions_rasterized(output_bbox, output_size)
    for region in regions:
        img[:, region[1]: region[3], region[0]: region[2], :] = background_color
    return img
