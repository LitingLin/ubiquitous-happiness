def get_standard_evaluation_dataset_filter():
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


def get_standard_evaluation_datasets():
    from Dataset.SOT.Seed.OTB import OTB_Seed
    from Dataset.SOT.Seed.LaSOT import LaSOT_Seed
    from Dataset.SOT.Seed.LaSOT_Extension import LaSOT_Extension_Seed
    from Dataset.SOT.Seed.GOT10k import GOT10k_Seed
    from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
    from Dataset.Type.data_split import DataSplit
    return SingleObjectTrackingDatasetFactory([OTB_Seed(), LaSOT_Seed(data_split=DataSplit.Validation), LaSOT_Extension_Seed(), GOT10k_Seed(data_split=DataSplit.Validation)]).construct(get_standard_evaluation_dataset_filter())


def run_standard_evaluation(tracker_name, tracker, output_path, run_times=None):
    datasets = get_standard_evaluation_datasets()
    from evaluation.SOT.protocol.ope import run_OPE_evalutation_and_generate_report, pack_OPE_result_and_report
    run_OPE_evalutation_and_generate_report(tracker_name, tracker, datasets, output_path, run_times)
    pack_OPE_result_and_report(tracker_name, output_path)
