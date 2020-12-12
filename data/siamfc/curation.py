import numpy
import numpy.random
from NativeExtension import InterpolationMethod, RGBImageTranslateAndScale
import cv2


def _get_center_and_scale(bounding_box, context, exemplar_size):
    assert len(bounding_box) == 4
    bounding_box = numpy.array(bounding_box, dtype=numpy.float32)
    target_size = bounding_box[2:4]

    context = context * numpy.sum(target_size)
    scale = numpy.sqrt(exemplar_size ** 2 / numpy.prod(target_size + context))

    scale = [scale, scale]

    center = [bounding_box[0] - 1 + bounding_box[2] / 2, bounding_box[1] - 1 + bounding_box[3] / 2]
    return center, scale


def _process(image, center, scale, out_size):

    out_center = [out_size / 2, out_size / 2]

    avg_color = numpy.round(cv2.mean(image)[0:3]).astype(numpy.uint8)

    interp = numpy.random.choice([
        InterpolationMethod.INTER_LINEAR,
        InterpolationMethod.INTER_CUBIC,
        InterpolationMethod.INTER_AREA,
        InterpolationMethod.INTER_NEAREST,
        InterpolationMethod.INTER_LANCZOS4])

    image = RGBImageTranslateAndScale(image, [out_size, out_size], center, out_center, scale, avg_color, interp)
    return image


def siamfc_z_curation(image, bounding_box, context, exemplar_size):
    return _process(image, *_get_center_and_scale(bounding_box, context, exemplar_size), exemplar_size)
