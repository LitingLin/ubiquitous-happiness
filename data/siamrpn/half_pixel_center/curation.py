import numpy as np
from data.operator.bbox.half_pixel_offset.center import get_center_point_from_xywh
from data.operator.image_and_bbox.half_pixel_center.torch_scale_and_translate import torch_scale_and_translate


def get_siamfc_curation_center_and_scale(bounding_box, context, exemplar_size):
    assert len(bounding_box) == 4
    bounding_box = np.array(bounding_box, dtype=np.float32)
    target_size = bounding_box[2:4]

    context = context * np.sum(target_size)
    scale = np.sqrt(exemplar_size ** 2 / np.prod(target_size + context))

    scale = (scale, scale)

    center = get_center_point_from_xywh(bounding_box)
    return center, scale


def siamfc_curation_with_shift_scale_augmentation(image, center, scale, out_size, max_translation=None, max_stretch_ratio=None):
    out_center = [out_size / 2, out_size / 2]

    if max_translation is not None:
        out_center[0] += np.random.uniform(-max_translation, max_translation)
        out_center[1] += np.random.uniform(-max_translation, max_translation)

    if max_stretch_ratio is not None:
        ratio_x = np.random.uniform(1 - max_stretch_ratio, 1 + max_stretch_ratio)
        ratio_y = np.random.uniform(1 - max_stretch_ratio, 1 + max_stretch_ratio)
        scale[0] *= ratio_x
        scale[1] *= ratio_y



    return torch_scale_and_translate(image, (out_size, out_size), scale, center, out_center, )

def curate_image_like_siamfc_with_aug(image, center, scale, out_size, max_translation=None, max_stretch_ratio=None):
    out_center = [out_size / 2, out_size / 2]

    if max_translation is not None:
        max_translation_x = max_translation / scale[0]
        max_translation_y = max_translation / scale[1]
        center[0] += np.random.uniform(-max_translation_x, max_translation_x)
        center[1] += np.random.uniform(-max_translation_y, max_translation_y)

    if max_stretch_ratio is not None:
        ratio_x = np.random.uniform(1 - max_stretch_ratio, 1 + max_stretch_ratio)
        ratio_y = np.random.uniform(1 - max_stretch_ratio, 1 + max_stretch_ratio)
        scale[0] *= ratio_x
        scale[1] *= ratio_y

    avg_color = np.round(cv2.mean(image)[0:3]).astype(np.uint8)

    image = RGBImageTranslateAndScale(image, [out_size, out_size], center, out_center, scale, avg_color, InterpolationMethod.INTER_LINEAR)
    return image

