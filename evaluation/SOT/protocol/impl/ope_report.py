import os
from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped, \
    SingleObjectTrackingDatasetSequence_MemoryMapped
from data.operator.bbox.spatial.vectorized.iou import bbox_compute_iou_numpy_vectorized
from data.operator.bbox.spatial.vectorized.validity import bbox_is_valid_vectorized
from typing import List, Optional
import numpy as np
from Dataset.Base.Common.constructor import DatasetProcessBar
import shutil
import pickle
from evaluation.SOT.operator.center_location_error import calculate_center_location_error_torch_vectorized
import json
from evaluation.SOT.protocol.ope import OPEEvaluationParameter
import csv
from evaluation.SOT.protocol.impl.ope_draw_plots import draw_success_plot, draw_precision_plot, \
    draw_normalized_precision_plot
from evaluation.SOT.protocol.impl.ope_run_evalution import get_sequence_result_path


def _calc_curves(ious, center_errors, norm_center_errors, parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    ious = np.asarray(ious, float)[:, np.newaxis]
    center_errors = np.asarray(center_errors, float)[:, np.newaxis]
    norm_center_errors = np.asarray(norm_center_errors, float)[:, np.newaxis]

    thr_iou = np.linspace(0, 1, parameter.bins_of_intersection_of_union)[np.newaxis, :]
    thr_ce = np.arange(0, parameter.bins_of_center_location_error)[np.newaxis, :]
    thr_nce = np.linspace(0, 0.5, parameter.bins_of_normalized_center_location_error)[np.newaxis, :]

    bin_iou = np.greater_equal(ious, thr_iou)
    bin_ce = np.less_equal(center_errors, thr_ce)
    bin_nce = np.less_equal(norm_center_errors, thr_nce)

    succ_curve = np.mean(bin_iou, axis=0)
    prec_curve = np.mean(bin_ce, axis=0)
    norm_prec_curve = np.mean(bin_nce, axis=0)

    return succ_curve, prec_curve, norm_prec_curve


def _calc_ao_sr(ious):
    return np.mean(ious), np.mean(ious >= 0.5), np.mean(ious >= 0.75)


def _calculate_evaluation_metrics(predicted_bounding_boxes,
                                  sequence: SingleObjectTrackingDatasetSequence_MemoryMapped,
                                  parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    ious = bbox_compute_iou_numpy_vectorized(predicted_bounding_boxes, sequence.get_all_bounding_box())
    center_location_errors = calculate_center_location_error_torch_vectorized(predicted_bounding_boxes,
                                                                              sequence.get_all_bounding_box())
    normalized_center_location_errors = calculate_center_location_error_torch_vectorized(predicted_bounding_boxes,
                                                                                         sequence.get_all_bounding_box(),
                                                                                         True)

    predicted_bboxes_validity = bbox_is_valid_vectorized(predicted_bounding_boxes)
    ious[~predicted_bboxes_validity] = -1.0
    center_location_errors[~predicted_bboxes_validity] = float('inf')
    normalized_center_location_errors[~predicted_bboxes_validity] = float('inf')

    if sequence.has_bounding_box_validity_flag():
        ious = ious[sequence.get_all_bounding_box_validity_flag()]
        center_location_errors = center_location_errors[sequence.get_all_bounding_box_validity_flag()]
        normalized_center_location_errors = normalized_center_location_errors[
            sequence.get_all_bounding_box_validity_flag()]

    succ_curve, prec_curve, norm_prec_curve = _calc_curves(ious, center_location_errors,
                                                           normalized_center_location_errors, parameter)
    ious[ious < 0.] = 0
    ao, sr_at_0_5, sr_at_0_75 = _calc_ao_sr(ious)

    return ao, sr_at_0_5, sr_at_0_75, succ_curve, prec_curve, norm_prec_curve


def _generate_sequence_report(ao, sr_at_0_5, sr_at_0_75, succ_curve, prec_curve, norm_prec_curve, running_time):
    return {
        'average_overlap': ao,
        'success_rate_at_iou_0.5': sr_at_0_5,
        'success_rate_at_iou_0.75': sr_at_0_75,
        'success_curve': succ_curve.tolist(),
        'precision_curve': prec_curve.tolist(),
        'normalized_precision_curve': norm_prec_curve.tolist(),
        'success_score': np.mean(succ_curve),
        'precision_score': prec_curve[20],  # center location error @ 20 pix
        'normalized_precision_score': norm_prec_curve[20],
        'running_time': running_time.tolist(),
        'fps': 1.0 / np.mean(running_time)}


def _get_mean_of_multiple_run(result):
    result = zip(*result)
    return [np.mean(metric, axis=0) for metric in result]


def _load_predicted_bounding_boxes(result_path: str):
    with open(os.path.join(result_path, 'bounding_box.p'), 'rb') as f:
        predicted_bounding_boxes = pickle.load(f)
    return predicted_bounding_boxes


def _load_running_time(result_path: str):
    with open(os.path.join(result_path, 'time.p'), 'rb') as f:
        times = pickle.load(f)
    return times


def generate_sequence_report(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, report_path, bounding_boxes,
                             running_times, run_times=None, parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    if run_times is not None:
        metrics = []
        for run_time in range(run_times):
            metrics.append(_calculate_evaluation_metrics(bounding_boxes[run_time], sequence, parameter))
        ao, sr_at_0_5, sr_at_0_75, succ_curve, prec_curve, norm_prec_curve = _get_mean_of_multiple_run(metrics)
        running_times = np.mean(running_times, axis=0)
    else:
        ao, sr_at_0_5, sr_at_0_75, succ_curve, prec_curve, norm_prec_curve = _calculate_evaluation_metrics(
            bounding_boxes, sequence, parameter)
    sequence_report = _generate_sequence_report(ao, sr_at_0_5, sr_at_0_75, succ_curve, prec_curve, norm_prec_curve,
                                                running_times)

    report_file_path = os.path.join(report_path, 'report.json')
    if not os.path.exists(report_file_path):
        with open(os.path.join(report_path, 'report.json'), 'w') as f:
            json.dump(sequence_report, f, indent=2)
    return sequence_report


def generate_dataset_report(tracker_name, sequence_reports, dataset, report_path: str,
                            parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    with open(os.path.join(report_path, 'sequences_report.csv'), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(('Sequence Name', 'Success Score', 'Precision Score', 'Normalized Precision Score',
                             'Average Overlap', 'Success Rate @ IOU>=0.5', 'Success Rate @ IOU>=0.75', 'FPS'))
        for sequence, sequence_report in zip(dataset, sequence_reports):
            csv_writer.writerow((sequence.get_name(), sequence_report['success_score'],
                                 sequence_report['precision_score'], sequence_report['normalized_precision_score'],
                                 sequence_report['average_overlap'], sequence_report['success_rate_at_iou_0.5'],
                                 sequence_report['success_rate_at_iou_0.75'],
                                 sequence_report['fps']))

    success_curve = np.mean([sequence_report['success_curve'] for sequence_report in sequence_reports], axis=0)
    precision_curve = np.mean([sequence_report['precision_curve'] for sequence_report in sequence_reports], axis=0)
    normalized_precision_curve = np.mean(
        [sequence_report['normalized_precision_curve'] for sequence_report in sequence_reports], axis=0)

    draw_success_plot(success_curve[np.newaxis, :], [tracker_name], report_path, parameter=parameter)
    draw_precision_plot(precision_curve[np.newaxis, :], [tracker_name], report_path, parameter=parameter)
    draw_normalized_precision_plot(normalized_precision_curve[np.newaxis, :], [tracker_name], report_path,
                                   parameter=parameter)

    success_score = np.mean(success_curve)
    precision_score = precision_curve[20]
    normalized_precision_score = normalized_precision_curve[20]

    average_overlap = np.mean([sequence_report['average_overlap'] for sequence_report in sequence_reports])
    success_rate_at_iou_0_5 = np.mean(
        [sequence_report['success_rate_at_iou_0.5'] for sequence_report in sequence_reports])
    success_rate_at_iou_0_75 = np.mean(
        [sequence_report['success_rate_at_iou_0.75'] for sequence_report in sequence_reports])

    dataset_report = {'success_score': success_score,
                      'precision_score': precision_score,
                      'normalized_precision_score': normalized_precision_score,
                      'average_overlap': average_overlap,
                      'success_rate_at_iou_0.5': success_rate_at_iou_0_5,
                      'success_rate_at_iou_0.75': success_rate_at_iou_0_75}
    with open(os.path.join(report_path, 'report.json'), 'w') as f:
        json.dump(dataset_report, f, indent=2)
    return dataset_report


def generate_report_one_pass_evaluation(tracker_name, datasets: List[SingleObjectTrackingDataset_MemoryMapped],
                                        output_path: str, run_times: Optional[int] = None,
                                        parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    report_path = os.path.join(output_path, 'ope', tracker_name, 'report')
    if os.path.exists(report_path):
        shutil.rmtree(report_path)
    os.makedirs(report_path)
    sequences_report_path = os.path.join(report_path, 'sequences')
    os.mkdir(sequences_report_path)

    result_path = os.path.join(output_path, 'ope', tracker_name, 'result')

    dataset_reports = {}

    for dataset in datasets:
        process_bar = DatasetProcessBar()
        process_bar.set_dataset_name(dataset.get_name())
        process_bar.set_total(len(dataset))
        sequence_reports = []
        for sequence in dataset:
            process_bar.set_sequence_name(sequence.get_name())

            sequence_report_path = os.path.join(sequences_report_path, sequence.get_name())

            if not os.path.exists(sequence_report_path):
                os.mkdir(sequence_report_path)

            if run_times is not None:
                bounding_boxes = []
                running_times = []
                for run_time in range(run_times):
                    sequence_result_path, _ = get_sequence_result_path(result_path, sequence, run_time)
                    bounding_boxes.append(_load_predicted_bounding_boxes(sequence_result_path))
                    running_times.append(_load_running_time(sequence_result_path))
            else:
                sequence_result_path, _ = get_sequence_result_path(result_path, sequence)
                bounding_boxes = _load_predicted_bounding_boxes(sequence_result_path)
                running_times = _load_running_time(sequence_result_path)
            sequence_reports.append(
                generate_sequence_report(sequence, sequence_report_path, bounding_boxes, running_times, run_times))
            process_bar.update()

        print(f'Generating {dataset.get_name()} dataset report...', end=' ')
        dataset_report_path = os.path.join(report_path, dataset.get_name())
        os.mkdir(dataset_report_path)
        dataset_reports[dataset.get_name()] = generate_dataset_report(tracker_name, sequence_reports, dataset, dataset_report_path, parameter)
        print('done')

        process_bar.close()

    print(f'Generating all datasets report...', end=' ')
    with open(os.path.join(report_path, 'report.csv'), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(('Dataset Name', 'Success Score', 'Precision Score', 'Normalized Precision Score',
                             'Average Overlap', 'Success Rate @ IOU>=0.5', 'Success Rate @ IOU>=0.75', 'FPS'))
        for dataset_name, dataset_report in dataset_reports.items():
            csv_writer.writerow((dataset_name, dataset_report['success_score'],
                                 dataset_report['precision_score'], dataset_report['normalized_precision_score'],
                                 dataset_report['average_overlap'], dataset_report['success_rate_at_iou_0.5'],
                                 dataset_report['success_rate_at_iou_0.75'],
                                 dataset_report['fps']))
    print('done')
