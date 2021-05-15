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


def get_standard_non_public_evaluation_dataset_filter():
    from Dataset.Filter.DataCleaning.AnnotationStandard import DataCleaning_AnnotationStandard
    from data.types.bounding_box_format import BoundingBoxFormat
    from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
    from data.types.pixel_coordinate_system import PixelCoordinateSystem
    from data.types.pixel_definition import PixelDefinition
    filters = [DataCleaning_AnnotationStandard(BoundingBoxFormat.XYXY, PixelCoordinateSystem.Aligned, BoundingBoxCoordinateSystem.Spatial, PixelDefinition.Point)]
    return filters


def get_standard_evaluation_datasets():
    from Dataset.SOT.Seed.OTB import OTB_Seed
    from Dataset.SOT.Seed.LaSOT import LaSOT_Seed
    from Dataset.SOT.Seed.LaSOT_Extension import LaSOT_Extension_Seed
    from Dataset.SOT.Seed.GOT10k import GOT10k_Seed
    from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
    from Dataset.Type.data_split import DataSplit
    return SingleObjectTrackingDatasetFactory([OTB_Seed(), GOT10k_Seed(data_split=DataSplit.Validation), LaSOT_Seed(data_split=DataSplit.Validation), LaSOT_Extension_Seed()]).construct(get_standard_evaluation_dataset_filter())


def get_standard_non_public_evaluation_datasets():
    from Dataset.SOT.Seed.GOT10k import GOT10k_Seed
    from Dataset.SOT.Seed.TrackingNet import TrackingNet_Seed
    from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
    from Dataset.Type.data_split import DataSplit
    return SingleObjectTrackingDatasetFactory(
        [GOT10k_Seed(data_split=DataSplit.Testing)]).construct(get_standard_non_public_evaluation_dataset_filter())


def run_standard_evaluation(tracker_name, tracker, output_path, generate_report=True, run_times=None):
    datasets = get_standard_evaluation_datasets()
    from evaluation.SOT.protocol.ope import run_one_pass_evaluation, run_OPE_evalutation_and_report_generation, pack_OPE_result_and_report
    if generate_report:
        run_OPE_evalutation_and_report_generation(tracker_name, tracker, datasets, output_path, run_times)
    else:
        run_one_pass_evaluation(tracker_name, tracker, datasets, output_path, run_times)
    pack_OPE_result_and_report(tracker_name, output_path)


def run_standard_non_public_dataset_evaluation(tracker_name, tracker, output_path):
    datasets = get_standard_non_public_evaluation_datasets()
    from evaluation.SOT.protocol.ope import run_one_pass_evaluation_on_non_public_datasets_and_pack_for_submission
    run_one_pass_evaluation_on_non_public_datasets_and_pack_for_submission(tracker_name, tracker, datasets, output_path)


def run_standard_report_generation(tracker_name, output_path, run_times=None):
    datasets = get_standard_evaluation_datasets()
    from evaluation.SOT.protocol.ope import generate_one_pass_evaluation_report, pack_OPE_result_and_report
    generate_one_pass_evaluation_report(tracker_name, datasets, output_path, run_times)
    pack_OPE_result_and_report(tracker_name, output_path)


def visualize_sequence(tracker_name, result_path, sequence_name, output_video_file_path, run_time=None):
    from evaluation.SOT.protocol.ope import OPE_visualize_sequence

    datasets = get_standard_evaluation_datasets()
    for dataset in datasets:
        for sequence in dataset:
            if sequence.get_name() == sequence_name:
                OPE_visualize_sequence(tracker_name, result_path, sequence, output_video_file_path, run_time)
                return


def visualize_tracking_results(tracker_names, result_paths, sequence_name, output_video_file_path, run_time=None):
    from evaluation.SOT.protocol.ope import OPE_visualize_tracking_results

    datasets = get_standard_evaluation_datasets()
    for dataset in datasets:
        for sequence in dataset:
            if sequence.get_name() == sequence_name:
                OPE_visualize_tracking_results(tracker_names, result_paths, sequence, output_video_file_path, run_time)
                return


if __name__ == '__main__':
    datasets = get_standard_non_public_evaluation_datasets()
