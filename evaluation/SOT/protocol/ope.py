import os.path

from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from typing import List, Optional


class OPEEvaluationParameter:
    bins_of_center_location_error = 51
    bins_of_normalized_center_location_error = 51
    bins_of_intersection_of_union = 21


def run_OPE_evalutation_and_report_generation(tracker_name, tracker, datasets, output_path, run_times=None, parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    from evaluation.SOT.protocol.impl.ope_run_evalution import prepare_result_path, run_one_pass_evaluation_on_dataset
    from evaluation.SOT.protocol.impl.ope_report import prepare_report_path, generate_dataset_report_one_pass_evaluation, dump_datasets_report

    result_path = prepare_result_path(output_path, datasets, tracker_name)
    report_path, sequences_report_path = prepare_report_path(output_path, tracker_name)

    datasets_report = {}

    for dataset in datasets:
        run_one_pass_evaluation_on_dataset(dataset, tracker, result_path, run_times)
        datasets_report[dataset.get_name()] = generate_dataset_report_one_pass_evaluation(tracker_name, dataset, result_path, report_path, sequences_report_path, run_times, parameter)

    dump_datasets_report(report_path, datasets_report)


def run_one_pass_evaluation(tracker_name, tracker, datasets: List[SingleObjectTrackingDataset_MemoryMapped], output_path: str, run_times: Optional[int]=None):
    from evaluation.SOT.protocol.impl.ope_run_evalution import prepare_result_path, run_one_pass_evaluation_on_dataset
    result_path = prepare_result_path(output_path, datasets, tracker_name)

    for dataset in datasets:
        run_one_pass_evaluation_on_dataset(dataset, tracker, result_path, run_times)


def run_one_pass_evaluation_on_non_public_datasets_and_pack_for_submission(tracker_name, tracker, datasets: List[SingleObjectTrackingDataset_MemoryMapped], output_path: str):
    from Dataset.Type.data_split import DataSplit
    from evaluation.SOT.protocol.impl.ope_run_evalution import prepare_result_path, run_one_pass_evaluation_on_dataset
    result_path = prepare_result_path(output_path, datasets, tracker_name)

    for dataset in datasets:
        run_one_pass_evaluation_on_dataset(dataset, tracker, result_path)
        if dataset.get_name() == 'GOT-10k' and dataset.get_data_split() == DataSplit.Testing:
            from evaluation.SOT.protocol.utils.as_got10k_format import convert_dataset_tracking_result_to_got10k_format
            target_path = os.path.join(output_path, 'submit')
            convert_dataset_tracking_result_to_got10k_format(tracker_name, dataset, result_path, target_path, True)
        else:
            print('Warn: Unsupported non-public dataset, packing disabled.')


def generate_one_pass_evaluation_report(tracker_name, datasets: List[SingleObjectTrackingDataset_MemoryMapped],
                                        output_path: str, run_times: Optional[int] = None,
                                        parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    from evaluation.SOT.protocol.impl.ope_run_evalution import prepare_result_path
    from evaluation.SOT.protocol.impl.ope_report import prepare_report_path, generate_dataset_report_one_pass_evaluation, dump_datasets_report

    report_path, sequences_report_path = prepare_report_path(output_path, tracker_name)
    result_path = prepare_result_path(output_path, datasets, tracker_name)

    datasets_report = {}

    for dataset in datasets:
        datasets_report[dataset.get_name()] = generate_dataset_report_one_pass_evaluation(tracker_name, dataset, result_path, report_path, sequences_report_path, run_times, parameter)

    dump_datasets_report(report_path, datasets_report)


def generate_multiple_tracker_one_pass_evaluation_report(search_paths, datasets: List[SingleObjectTrackingDataset_MemoryMapped],
                                        output_path: str, run_times: Optional[int] = None,
                                        parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    import evaluation.SOT.protocol.impl.ope_multi_tracker_report
    return evaluation.SOT.protocol.impl.ope_multi_tracker_report.generate_multiple_tracker_one_pass_evaluation_report(search_paths, datasets, output_path, run_times, parameter)


def pack_OPE_result_and_report(tracker_name, output_path):
    from Miscellaneous.pack_directory import make_tarfile
    import os
    make_tarfile(os.path.join(output_path, f'{tracker_name}.tar.xz'), os.path.join(output_path, 'ope', tracker_name))


def OPE_visualize_sequence(tracker_name, result_path, sequence, output_video_file_path, run_time=None):
    from evaluation.SOT.protocol.impl.ope_run_evalution import get_sequence_result_path
    from evaluation.SOT.protocol.impl.ope_report import _load_predicted_bounding_boxes
    sequence_result_path, _ = get_sequence_result_path(result_path, sequence, run_time)
    predicted_bounding_boxes = _load_predicted_bounding_boxes(sequence_result_path)
    from evaluation.SOT.visualization.video_generation import generate_sequence_video, get_standard_bounding_box_rasterizer
    bounding_box_rasterizer = get_standard_bounding_box_rasterizer()
    generate_sequence_video(tracker_name, sequence, predicted_bounding_boxes, bounding_box_rasterizer, output_video_file_path)


def OPE_visualize_tracking_results(tracker_names, result_paths, sequence, output_video_file_path, run_time=None):
    from evaluation.SOT.protocol.impl.ope_run_evalution import get_sequence_result_path
    from evaluation.SOT.protocol.impl.ope_report import _load_predicted_bounding_boxes
    bounding_boxes = []
    for result_path in result_paths:
        sequence_result_path, _ = get_sequence_result_path(result_path, sequence, run_time)
        bounding_boxes.append(_load_predicted_bounding_boxes(sequence_result_path))
    from evaluation.SOT.visualization.video_generation import visualize_tracking_results, get_standard_bounding_box_rasterizer
    bounding_box_rasterizer = get_standard_bounding_box_rasterizer()
    visualize_tracking_results(tracker_names, sequence, bounding_boxes, bounding_box_rasterizer, output_video_file_path)
