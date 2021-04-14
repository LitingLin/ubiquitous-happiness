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


if __name__ == '__main__':
    datasets = SingleObjectTrackingDatasetFactory([OTB_Seed()]).construct(filters=get_standard_training_dataset_filter())
    from data.TransT.pipeline import transt_preprocessing_pipeline
    dataset = datasets[0]
    image_path = dataset[0][0].get_image_path()
    bounding_box = dataset[0][0].get_bounding_box()
    from data.operator.image.decoder import tf_decode_image
    image = tf_decode_image(image_path)

    from data.operator.image.batchify import tf_batchify
    image = tf_batchify(image)

    output_image, output_bbox = transt_preprocessing_pipeline(image, bounding_box, 4, (224, 224), 0.25, 3, 0.2)

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
    viewer.runEventLoop()
