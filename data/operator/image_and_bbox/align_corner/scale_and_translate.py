from data.operator.image.align_corner.scale_and_translate import tf_image_scale_and_translate_align_corners
from data.operator.bbox.scale_and_translate import bbox_scale_and_translate
from data.operator.bbox.align_corner.validity import is_bbox_validity
from data.operator.bbox.intersection import bbox_compute_intersection
import tensorflow as tf
import math


def _generate_background_regions_rasterized(foreground_bbox, image_size):
    bboxs = [] # in xywh
    if foreground_bbox[0] > 0:
        x2 = math.floor(foreground_bbox[0])
        if x2 == foreground_bbox[0]:
            x2 -= 1

        bbox = [0, 0, foreground_bbox[0] + 1, image_size[1]]
        bboxs.append(bbox)
    if foreground_bbox[1] > 0:
        x1 =



def tf_scale_and_translate_align_corners_numpy(img, output_size, scale, input_center=(0, 0), output_center=(0, 0), background_color=(0, 0, 0), kernel_type='triangle', antialias=False):
    n, h, w, c = img.shape
    assert c == len(background_color)

    output_bbox = bbox_scale_and_translate((0, 0, w - 1, h - 1), scale, input_center, output_center)
    if not is_bbox_validity(bbox_compute_intersection(output_bbox, (0, 0, output_size[0] - 1, output_size[1] - 1))):
        return tf.tile(tf.constant(background_color, shape=(1, 1, 1, c)), (n, h, w, 1))
    img = tf_image_scale_and_translate_align_corners(img, output_size, scale, input_center, output_center, kernel_type, antialias)
    img = img.numpy()
