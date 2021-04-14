
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
               DataCleaning_Integrity()
               ]
    return filters

if __name__ == '__main__':
    datasets = SingleObjectTrackingDatasetFactory([OTB_Seed()]).construct(filters=get_standard_training_dataset_filter())

    # jittered_center_crop