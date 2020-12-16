import numpy
import numpy.random
import torch
from NativeExtension import InterpolationMethod, RGBImageTranslateAndScale, RGBImageToGrayScale
import cv2


def curate_image_like_siamfc_with_aug(image, center, scale, out_size, max_translation=None, max_stretch_ratio=None, rgb_variance=None, gray_scale=False):
    if gray_scale:
        image = RGBImageToGrayScale(image)

    out_center = [out_size / 2, out_size / 2]

    if max_translation is not None:
        max_translation_x = max_translation / scale[0]
        max_translation_y = max_translation / scale[1]
        center[0] += numpy.random.uniform(-max_translation_x, max_translation_x)
        center[1] += numpy.random.uniform(-max_translation_y, max_translation_y)

    if max_stretch_ratio is not None:
        ratio_x = numpy.random.uniform(1 - max_stretch_ratio, 1 + max_stretch_ratio)
        ratio_y = numpy.random.uniform(1 - max_stretch_ratio, 1 + max_stretch_ratio)
        scale[0] *= ratio_x
        scale[1] *= ratio_y

    avg_color = numpy.round(cv2.mean(image)[0:3]).astype(numpy.uint8)

    interp = numpy.random.choice([
        InterpolationMethod.INTER_LINEAR,
        InterpolationMethod.INTER_CUBIC,
        InterpolationMethod.INTER_AREA,
        InterpolationMethod.INTER_NEAREST,
        InterpolationMethod.INTER_LANCZOS4])

    image = RGBImageTranslateAndScale(image, [out_size, out_size], center, out_center, scale, avg_color, interp)
    image = image.astype(numpy.float32)

    if rgb_variance is not None:
        image -= rgb_variance.dot(numpy.random.randn(3).astype(numpy.float32))
        image = numpy.clip(image, 0., 255.)

    return torch.from_numpy(image).permute((2, 0, 1))


def get_siamfc_curation_center_and_scale(bounding_box, context, exemplar_size):
    assert len(bounding_box) == 4
    bounding_box = numpy.array(bounding_box, dtype=numpy.float32)
    target_size = bounding_box[2:4]

    context = context * numpy.sum(target_size)
    scale = numpy.sqrt(exemplar_size ** 2 / numpy.prod(target_size + context))

    scale = [scale, scale]

    center = [bounding_box[0] - 1 + bounding_box[2] / 2, bounding_box[1] - 1 + bounding_box[3] / 2]
    return center, scale


def siamfc_z_curation(image, bounding_box, context, exemplar_size):
    return curate_image_like_siamfc_with_aug(image, *get_siamfc_curation_center_and_scale(bounding_box, context, exemplar_size), exemplar_size)


def siamfc_x_curation(image, bounding_box, context, exemplar_size, instance_size):
    return curate_image_like_siamfc_with_aug(image, *get_siamfc_curation_center_and_scale(bounding_box, context, exemplar_size), instance_size)
