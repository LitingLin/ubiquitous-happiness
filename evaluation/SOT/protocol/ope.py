class OPEEvaluationParameter:
    bins_of_center_location_error = 51
    bins_of_normalized_center_location_error = 51
    bins_of_intersection_of_union = 21


def run_OPE_evalutation_and_generate_report(tracker_name, tracker, datasets, output_path, run_times=None, parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    from evaluation.SOT.protocol.impl.ope_run_evalution import prepare_result_path, run_one_pass_evaluation_on_dataset
    from evaluation.SOT.protocol.impl.ope_report import prepare_report_path, generate_dataset_report_one_pass_evaluation, dump_datasets_report

    result_path = prepare_result_path(output_path, datasets, tracker_name)
    report_path, sequences_report_path = prepare_report_path(output_path, tracker_name)

    datasets_report = {}

    for dataset in datasets:
        run_one_pass_evaluation_on_dataset(dataset, tracker, result_path, run_times)
        datasets_report[dataset.get_name()] = generate_dataset_report_one_pass_evaluation(tracker_name, dataset, result_path, report_path, sequences_report_path, run_times, parameter)

    dump_datasets_report(report_path, datasets_report)


def pack_OPE_result_and_report(tracker_name, output_path):
    from Miscellaneous.pack_directory import make_tarfile
    import os
    make_tarfile(os.path.join(output_path, f'{tracker_name}.tar.xz'), os.path.join(output_path, 'ope', tracker_name))


def OPE_visualize_sequence(result_path, sequence, output_video_file_path, run_time=None):
    from evaluation.SOT.protocol.impl.ope_run_evalution import get_sequence_result_path
    from evaluation.SOT.protocol.impl.ope_report import _load_predicted_bounding_boxes
    sequence_result_path, _ = get_sequence_result_path(result_path, sequence, run_time)
    predicted_bounding_boxes = _load_predicted_bounding_boxes(sequence_result_path)
    from evaluation.SOT.visualization.video_generation import generate_sequence_video, get_standard_bounding_box_rasterizer
    bounding_box_rasterizer = get_standard_bounding_box_rasterizer()
    generate_sequence_video(sequence, predicted_bounding_boxes, bounding_box_rasterizer, output_video_file_path)
