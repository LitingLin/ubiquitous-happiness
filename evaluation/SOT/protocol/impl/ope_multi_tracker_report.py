import os
import shutil
from typing import Optional, List
from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from evaluation.SOT.protocol.ope import OPEEvaluationParameter
from evaluation.SOT.protocol.impl.ope_draw_plots import draw_success_plot, draw_precision_plot, draw_normalized_precision_plot, draw_sequences_success_plot, draw_sequences_precision_plot, draw_sequences_norm_precision_plot, draw_sequences_average_overlap_plot, draw_sequences_success_rate_at_iou_0_5_plot, draw_sequences_success_rate_at_iou_0_75_plot, draw_sequences_fps_plot
import numpy as np
import csv


def generate_multiple_tracker_one_pass_evaluation_report(search_paths, datasets: List[SingleObjectTrackingDataset_MemoryMapped],
                                        output_path: str, run_times: Optional[int] = None,
                                        parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    from evaluation.SOT.protocol.impl.ope_report import generate_dataset_report_one_pass_evaluation, dump_datasets_report
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    tracker_names = []
    tracker_ope_evaluation_result_paths = []
    tracker_ope_report_paths = []
    for search_path in search_paths:
        ope_path = os.path.join(search_path, 'ope')
        tracker_names_ = os.listdir(ope_path)
        tracker_names_ = [tracker_name for tracker_name in tracker_names_ if os.path.isdir(os.path.join(ope_path, tracker_name))]
        for tracker_name in tracker_names_:
            assert tracker_name not in tracker_names
        tracker_names.extend(tracker_names_)
        tracker_ope_evaluation_result_paths.extend([os.path.join(ope_path, tracker_name, 'result') for tracker_name in tracker_names_])
        for tracker_name in tracker_names_:
            tracker_ope_report_path = os.path.join(output_path, tracker_name)
            os.mkdir(tracker_ope_report_path)
            tracker_sequence_report_path = os.path.join(tracker_ope_report_path, 'sequences')
            os.mkdir(tracker_sequence_report_path)
            tracker_ope_report_paths.append(tracker_ope_report_path)

    combined_report_path = os.path.join(output_path, 'combined')
    os.mkdir(combined_report_path)

    tracker_dataset_reports = {}

    dataset_tracker_reports = {}

    for dataset in datasets:
        dataset_tracker_reports[dataset.get_name()] = {}
        for tracker_name, tracker_result_path, tracker_report_path in zip(tracker_names, tracker_ope_evaluation_result_paths, tracker_ope_report_paths):
            tracker_sequence_report_path = os.path.join(tracker_report_path, 'sequences')
            dataset_report, sequences_report = generate_dataset_report_one_pass_evaluation(tracker_name, dataset, tracker_result_path, tracker_report_path, tracker_sequence_report_path, run_times, parameter)

            if tracker_name not in tracker_dataset_reports:
                tracker_dataset_reports[tracker_name] = {}
            tracker_dataset_reports[tracker_name][dataset.get_name()] = (dataset_report, sequences_report)

            dataset_tracker_reports[dataset.get_name()][tracker_name] = (dataset_report, sequences_report)

    for tracker_name, datasets_report in tracker_dataset_reports.items():
        tracker_report_path = os.path.join(output_path, tracker_name)
        dump_datasets_report(tracker_report_path, datasets_report)

    for dataset in datasets:
        tracker_reports = dataset_tracker_reports[dataset.get_name()]

        sequence_names = [sequence.get_name() for sequence in dataset]

        dataset_success_curve = []
        dataset_precision_curve = []
        dataset_norm_precision_curve = []

        dataset_success_score = []
        dataset_precision_score = []
        dataset_normalized_precision_score = []
        dataset_average_overlap = []
        dataset_success_rate_at_iou_0_5 = []
        dataset_success_rate_at_iou_0_75 = []
        dataset_fps = []

        sequences_success = []
        sequences_precision = []
        sequences_norm_precision = []
        sequences_average_overlap = []
        sequences_success_rate_at_iou_0_5 = []
        sequences_success_rate_at_iou_0_75 = []
        sequences_fps = []

        for tracker_name, (dataset_report, sequences_report) in tracker_reports.items():
            success_curve = dataset_report['success_curve']
            precision_curve = dataset_report['precision_curve']
            normalized_precision_curve = dataset_report['normalized_precision_curve']

            dataset_success_score.append(dataset_report['success_score'])
            dataset_precision_score.append(dataset_report['precision_score'])
            dataset_normalized_precision_score.append(dataset_report['normalized_precision_score'])
            dataset_average_overlap.append(dataset_report['average_overlap'])
            dataset_success_rate_at_iou_0_5.append(dataset_report['success_rate_at_iou_0.5'])
            dataset_success_rate_at_iou_0_75.append(dataset_report['success_rate_at_iou_0.75'])
            dataset_fps.append(dataset_report['fps'])

            dataset_success_curve.append(success_curve)
            dataset_precision_curve.append(precision_curve)
            dataset_norm_precision_curve.append(normalized_precision_curve)

            sequences_success.append([sequence_report['success_score'] for sequence_report in sequences_report])
            sequences_precision.append([sequence_report['precision_score'] for sequence_report in sequences_report])
            sequences_norm_precision.append([sequence_report['normalized_precision_score'] for sequence_report in sequences_report])
            sequences_average_overlap.append([sequence_report['average_overlap'] for sequence_report in sequences_report])
            sequences_success_rate_at_iou_0_5.append([sequence_report['success_rate_at_iou_0.5'] for sequence_report in sequences_report])
            sequences_success_rate_at_iou_0_75.append([sequence_report['success_rate_at_iou_0.75'] for sequence_report in sequences_report])
            sequences_fps.append([sequence_report['fps'] for sequence_report in sequences_report])
        sequences_success = np.array(sequences_success)
        sequences_precision = np.array(sequences_precision)
        sequences_norm_precision = np.array(sequences_norm_precision)
        sequences_average_overlap = np.array(sequences_average_overlap)
        sequences_success_rate_at_iou_0_5 = np.array(sequences_success_rate_at_iou_0_5)
        sequences_success_rate_at_iou_0_75 = np.array(sequences_success_rate_at_iou_0_75)
        sequences_fps = np.array(sequences_fps)

        combined_dataset_report_path = os.path.join(combined_report_path, dataset.get_name())
        os.mkdir(combined_dataset_report_path)

        dataset_success_curve = np.array(dataset_success_curve)
        dataset_precision_curve = np.array(dataset_precision_curve)
        dataset_norm_precision_curve = np.array(dataset_norm_precision_curve)
        draw_success_plot(dataset_success_curve, tracker_names, combined_dataset_report_path)
        draw_precision_plot(dataset_precision_curve, tracker_names, combined_dataset_report_path)
        draw_normalized_precision_plot(dataset_norm_precision_curve, tracker_names, combined_dataset_report_path)

        draw_sequences_success_plot(sequence_names, sequences_success, tracker_names, combined_dataset_report_path)
        draw_sequences_precision_plot(sequence_names, sequences_precision, tracker_names, combined_dataset_report_path)
        draw_sequences_norm_precision_plot(sequence_names, sequences_norm_precision, tracker_names, combined_dataset_report_path)
        draw_sequences_average_overlap_plot(sequence_names, sequences_average_overlap, tracker_names, combined_dataset_report_path)
        draw_sequences_success_rate_at_iou_0_5_plot(sequence_names, sequences_success_rate_at_iou_0_5, tracker_names, combined_dataset_report_path)
        draw_sequences_success_rate_at_iou_0_75_plot(sequence_names, sequences_success_rate_at_iou_0_75, tracker_names, combined_dataset_report_path)
        draw_sequences_fps_plot(sequence_names, sequences_fps, tracker_names, combined_dataset_report_path)

        with open(os.path.join(combined_dataset_report_path, 'performance.csv'), 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(('Tracker Name', 'Success Score', 'Precision Score', 'Normalized Precision Score',
                                 'Average Overlap', 'Success Rate @ IOU>=0.5', 'Success Rate @ IOU>=0.75', 'FPS'))
            [csv_writer.writerow(data) for data in zip(tracker_names, dataset_success_score, dataset_precision_score, dataset_normalized_precision_score, dataset_average_overlap, dataset_success_rate_at_iou_0_5, dataset_success_rate_at_iou_0_75, dataset_fps)]
