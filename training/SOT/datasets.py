def get_standard_trainning_dataset_filter():
    from Dataset.Filter.DataCleaning.BoundingBox import DataCleaning_BoundingBox
    from Dataset.Filter.DataCleaning.Integrity import DataCleaning_Integrity
    from Dataset.Filter.DataCleaning.AnnotationStandard import DataCleaning_AnnotationStandard
    from data.types.bounding_box_format import BoundingBoxFormat
    from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
    from data.types.pixel_coordinate_system import PixelCoordinateSystem
    from data.types.pixel_definition import PixelDefinition
    filters = [DataCleaning_AnnotationStandard(BoundingBoxFormat.XYXY, PixelCoordinateSystem.Aligned, BoundingBoxCoordinateSystem.Spatial, PixelDefinition.Point),
               DataCleaning_BoundingBox(fit_in_image_size=True, update_validity=True, remove_invalid_objects=True, remove_empty_objects=True),
               DataCleaning_Integrity(remove_zero_annotation_objects=True, remove_zero_annotation_image=True, remove_zero_annotation_video_head_tail=True, remove_invalid_image=True)]
    return filters


def get_standard_training_datasets():
    from Dataset.SOT.Seed.OTB import OTB_Seed
    from Dataset.SOT.Seed.LaSOT import LaSOT_Seed
    from Dataset.SOT.Seed.TrackingNet import TrackingNet_Seed
    from Dataset.SOT.Seed.GOT10k import GOT10k_Seed
    from Dataset.SOT.Seed.UAV_Benchmark_S import UAV_Benchmark_S_Seed
    from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
    from Dataset.Type.data_split import DataSplit
    datasets = SingleObjectTrackingDatasetFactory(
        [OTB_Seed(), LaSOT_Seed(data_split=DataSplit.Training), TrackingNet_Seed(data_split=DataSplit.Training),
         GOT10k_Seed(data_split=DataSplit.Training), UAV_Benchmark_S_Seed()]).construct(get_standard_trainning_dataset_filter())

    from Dataset.MOT.Seed.ILSVRC_VID import ILSVRC_VID_Seed
    from Dataset.MOT.Seed.UAV_Benchmark_M import UAV_Benchmark_M_Seed
    from Dataset.MOT.Seed.YoutubeBB import YoutubeBB_Seed
    from Dataset.MOT.factory import MultipleObjectTrackingDatasetFactory

    mot_datasets = MultipleObjectTrackingDatasetFactory([ILSVRC_VID_Seed(), UAV_Benchmark_M_Seed(), YoutubeBB_Seed()]).construct(get_standard_trainning_dataset_filter())

    from Dataset.DET.Seed.COCO import COCO_Seed
    from Dataset.DET.Seed.ILSVRC_DET import ILSVRC_DET_Seed
    from Dataset.DET.Seed.BDD100k_Images import BDD100k_Images_Seed
    from Dataset.DET.Seed.KITTI_Detection import KITTI_Detection_Seed
    from Dataset.DET.Seed.OpenImages import OpenImages_Seed
    from Dataset.DET.factory import DetectionDatasetFactory
    det_datasets = DetectionDatasetFactory([COCO_Seed(),ILSVRC_DET_Seed(), BDD100k_Images_Seed(), KITTI_Detection_Seed(), OpenImages_Seed()]).construct(get_standard_trainning_dataset_filter())
    return datasets, mot_datasets, det_datasets


if __name__ == '__main__':
    datasets = get_standard_training_datasets()
