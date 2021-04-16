from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
from Dataset.SOT.Seed.OTB import OTB_Seed


def get_standard_training_dataset_filter():
    from Dataset.Filter.DataCleaning.BoundingBox import DataCleaning_BoundingBox
    from Dataset.Filter.DataCleaning.Integrity import DataCleaning_Integrity
    from Dataset.Filter.DataCleaning.AnnotationStandard import DataCleaning_AnnotationStandard
    from data.types.bounding_box_format import BoundingBoxFormat
    from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
    from data.types.pixel_coordinate_system import PixelCoordinateSystem
    from data.types.pixel_definition import PixelDefinition
    filters = [DataCleaning_AnnotationStandard(BoundingBoxFormat.XYXY, PixelCoordinateSystem.Aligned, BoundingBoxCoordinateSystem.Spatial, PixelDefinition.Point),
               DataCleaning_BoundingBox(update_validity=True, remove_invalid_objects=True, remove_empty_objects=True),
               DataCleaning_Integrity()]
    return filters


def get_bbox_converter():
    from data.types.bounding_box_format import BoundingBoxFormat
    from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
    from data.types.pixel_coordinate_system import PixelCoordinateSystem
    from data.types.pixel_definition import PixelDefinition

    from data.operator.bbox.transform.compile import compile_bbox_transform

    return compile_bbox_transform(BoundingBoxFormat.XYXY, BoundingBoxFormat.XYWH, PixelCoordinateSystem.Aligned, PixelCoordinateSystem.Aligned, BoundingBoxCoordinateSystem.Spatial, BoundingBoxCoordinateSystem.Rasterized, PixelDefinition.Point)
import torch
from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
from data.operator.bbox.spatial.center import bbox_get_center_point
from data.operator.bbox.spatial.utility.aligned.image import get_image_center_point
from data.operator.image_and_bbox.align_corner.scale_and_translate import tf_scale_and_translate_numpy
from data.operator.image.mean import tf_get_image_mean
from data.operator.bbox.spatial.scale_and_translate import bbox_scale_and_translate
import math
import numpy as np
from data.operator.image.imagenet_normalize import image_torch_tensor_imagenet_normalize
from data.operator.image.rgb_to_gray import tf_image_rgb_to_gray_keep_channels
from torchvision.transforms.functional import adjust_brightness
from data.operator.image.numpy_pytorch_interop import image_numpy_to_torch_HWC_to_CHW
from data.operator.image.batchify import unbatchify
from data.operator.image.normalize import torch_image_normalize
from data.operator.bbox.spatial.utility.aligned.image import bounding_box_fit_in_image_boundary, bounding_box_is_intersect_with_image


def get_rand():
    return torch.tensor([0.3, 0.3])


def get_jittered_scaling_and_translate_factor(bbox, scaling, scaling_jitter_factor, translation_jitter_factor):
    scaling = scaling / torch.exp(get_rand() * scaling_jitter_factor)
    bbox = bbox_xyxy2xywh(bbox)
    max_translate = (torch.tensor(bbox[2:4]) * scaling).sum() * 0.5 * translation_jitter_factor
    translate = (get_rand() - 0.5) * max_translate
    return scaling, translate


def get_scaling_factor_from_area_factor(bbox, area_factor, output_size):
    bbox = bbox_xyxy2xywh(bbox)
    w, h = bbox[2: 4]
    w_z = w + (area_factor-1)*((w+h)*0.5)
    h_z = h + (area_factor-1)*((w+h)*0.5)
    scaling = math.sqrt((output_size[0] * output_size[1]) / (w_z * h_z))
    return torch.tensor((scaling, scaling))


def jittered_center_crop(image, bbox, area_factor, output_size, scaling_jitter_factor, translation_jitter_factor):
    while True:
        scaling = get_scaling_factor_from_area_factor(bbox, area_factor, output_size)
        scaling, translate = get_jittered_scaling_and_translate_factor(bbox, scaling, scaling_jitter_factor, translation_jitter_factor)

        source_center = bbox_get_center_point(bbox)
        target_center = get_image_center_point(output_size)
        target_center = (torch.tensor(target_center) - translate).tolist()
        scaling = scaling.tolist()

        output_bbox = bbox_scale_and_translate(bbox, scaling, source_center, target_center)

        if bounding_box_is_intersect_with_image(output_bbox, output_size):
            break

    output_image = tf_scale_and_translate_numpy(image, output_size, scaling, source_center, target_center, tf_get_image_mean(image).numpy())
    return output_image, output_bbox


def _get_jittered_box(box, scale_jitter_factor, center_jitter_factor):
    """ Jitter the input box
    args:
        box - input bounding box

    returns:
        torch.Tensor - jittered box
    """

    jittered_size = box[2:4] * torch.exp(get_rand() * scale_jitter_factor)
    max_offset = (jittered_size.sum() * 0.5 * torch.tensor(center_jitter_factor).float())
    jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (get_rand() - 0.5)
    return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)


def o_imp_(image, bbox, area_factor, output_size, scaling_jitter_factor, translation_jitter_factor):
    import tensorflow as tf
    image = tf.squeeze(image, 0)
    image = image.numpy()
    bbox = torch.tensor([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=torch.float)
    target_box = _get_jittered_box(bbox, scaling_jitter_factor, translation_jitter_factor)
    from data.TransT.o_impl import jittered_center_crop
    a, b, _ = jittered_center_crop([image], [target_box], [bbox], area_factor, output_size)
    b = b[0]
    b = b.tolist()
    b = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
    return a[0], b


if __name__ == '__main__':
    datasets = SingleObjectTrackingDatasetFactory([OTB_Seed()]).construct(filters=get_standard_training_dataset_filter())
    dataset = datasets[0]
    image_path = dataset[0][0].get_image_path()
    bounding_box = dataset[0][0].get_bounding_box()
    from data.operator.image.decoder import tf_decode_image
    image = tf_decode_image(image_path)

    from data.operator.image.batchify import tf_batchify
    image = tf_batchify(image)

    output_image, output_bbox = jittered_center_crop(image, bounding_box, 4, (256, 256), 0.25, 3)
    from data.operator.image.batchify import unbatchify
    output_image = unbatchify(output_image)

    from data.operator.image.dtype import image_round_to_uint8
    output_image = image_round_to_uint8(output_image)

    from Viewer.qt5_viewer import Qt5Viewer
    viewer = Qt5Viewer()
    painter = viewer.getPainter()
    from Miscellaneous.qt_numpy_interop import numpy_rgb888_to_qimage

    from PyQt5.QtGui import QPixmap
    painter.setCanvas(QPixmap(numpy_rgb888_to_qimage(output_image)))

    bbox_converter = get_bbox_converter()
    output_bbox = bbox_converter(output_bbox)
    with painter:
        painter.drawBoundingBox(output_bbox)

    painter.update()

    output_image, output_bbox = o_imp_(image, bounding_box, 4, 256, 0.25, 3)

    from data.operator.image.dtype import image_round_to_uint8
    output_image = image_round_to_uint8(output_image)

    painter = viewer.newCanvas()
    from Miscellaneous.qt_numpy_interop import numpy_rgb888_to_qimage

    from PyQt5.QtGui import QPixmap
    painter.setCanvas(QPixmap(numpy_rgb888_to_qimage(output_image)))

    output_bbox = bbox_converter(output_bbox)
    with painter:
        painter.drawBoundingBox(output_bbox)

    painter.update()

    output_image, output_bbox = jittered_center_crop(image, bounding_box, 2, (128, 128), 0, 0)
    output_image = unbatchify(output_image)

    from data.operator.image.dtype import image_round_to_uint8
    output_image = image_round_to_uint8(output_image)

    painter = viewer.newCanvas()
    from Miscellaneous.qt_numpy_interop import numpy_rgb888_to_qimage

    from PyQt5.QtGui import QPixmap
    painter.setCanvas(QPixmap(numpy_rgb888_to_qimage(output_image)))

    output_bbox = bbox_converter(output_bbox)
    with painter:
        painter.drawBoundingBox(output_bbox)

    painter.update()

    output_image, output_bbox = o_imp_(image, bounding_box, 2, 128, 0, 0)

    from data.operator.image.dtype import image_round_to_uint8
    output_image = image_round_to_uint8(output_image)

    painter = viewer.newCanvas()
    from Miscellaneous.qt_numpy_interop import numpy_rgb888_to_qimage

    from PyQt5.QtGui import QPixmap
    painter.setCanvas(QPixmap(numpy_rgb888_to_qimage(output_image)))

    output_bbox = bbox_converter(output_bbox)
    with painter:
        painter.drawBoundingBox(output_bbox)

    painter.update()

    viewer.runEventLoop()
