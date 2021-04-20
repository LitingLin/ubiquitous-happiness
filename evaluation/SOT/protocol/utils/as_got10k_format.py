import os
import numpy as np


def get_bounding_box_converter():
    from data.operator.bbox.transform.compile import compile_bbox_transform

    from data.types.bounding_box_format import BoundingBoxFormat
    from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
    from data.types.pixel_coordinate_system import PixelCoordinateSystem
    from data.types.pixel_definition import PixelDefinition

    return compile_bbox_transform(BoundingBoxFormat.XYXY, BoundingBoxFormat.XYWH, PixelCoordinateSystem.Aligned, PixelCoordinateSystem.Aligned, BoundingBoxCoordinateSystem.Spatial, BoundingBoxCoordinateSystem.Spatial, PixelDefinition.Point)


def _convert_tracking_result_to_got10k_format(sequence, result_path, target_path, run_times=None):
    assert run_times is None
    from evaluation.SOT.protocol.impl.ope_run_evalution import get_sequence_result_path
    from evaluation.SOT.protocol.impl.ope_report import _load_predicted_bounding_boxes, _load_running_time
    sequence_result_path, _ = get_sequence_result_path(result_path, sequence)
    bounding_boxes = _load_predicted_bounding_boxes(sequence_result_path)
    bounding_box_converter = get_bounding_box_converter()
    bounding_boxes = np.array([bounding_box_converter(bounding_box.tolist()) for bounding_box in bounding_boxes])
    times = _load_running_time(sequence_result_path)

    target_sequence_path = os.path.join(target_path, sequence.get_name())
    os.mkdir(target_sequence_path)

    np.savetxt(os.path.join(target_sequence_path, f'{sequence.get_name()}_001.txt'), bounding_boxes, fmt='%.3f', delimiter=',')
    np.savetxt(os.path.join(target_sequence_path, f'{sequence.get_name()}_time.txt'), times, fmt='%.8f')


def convert_dataset_tracking_result_to_got10k_format(tracker_name, dataset, result_path, target_path, run_times=None):
    target_dataset_path = os.path.join(target_path, dataset.get_name(), tracker_name)
    os.makedirs(target_dataset_path, exist_ok=True)

    for sequence in dataset:
        _convert_tracking_result_to_got10k_format(sequence, result_path, target_dataset_path, run_times)


def convert_datasets_tracking_result_to_got10k_format(tracker_name, datasets, result_path, target_path, run_times=None):
    for dataset in datasets:
        convert_dataset_tracking_result_to_got10k_format(tracker_name, dataset, result_path, target_path, run_times)
