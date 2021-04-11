import os
from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped, SingleObjectTrackingDatasetSequence_MemoryMapped
from evaluation.SOT.operator.half_pixel_center.iou import calculate_iou_overlap_torch_vectorized
import torch
from evaluation.SOT.operator.auc import
from typing import List, Optional


def calculate_success_plot(predicted_bboxes, groundtruth_bboxes, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped):
    ious = calculate_iou_overlap_torch_vectorized(predicted_bboxes, groundtruth_bboxes)

    validity_flags = ((sequence.get_all_bounding_box()[:, 2:] > 0.0).sum(1) == 2)
    if sequence.has_bounding_box_validity_flag():
        validity_flags = sequence.get_all_bounding_box_validity_flag() & validity_flags

    ious[~validity_flags] = -1.0

    plot_bin_gap = 0.05
    threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)

    return ious


def calculate_precision_plot(predicted_bboxes, groundtruth_bboxes, dataset: Optional[SingleObjectTrackingDataset_MemoryMapped]=None):
    if dataset is not None:
        if 'LaSOT' in dataset.get_name():

            pass
    pass

def calculate_normalized_precision_plot(predicted_bboxes, groundtruth_bboxes, dataset: Optional[SingleObjectTrackingDataset_MemoryMapped]=None):
    if dataset is not None:
        if 'LaSOT' in dataset.get_name():

            pass
    pass


def run_one_pass_evaluation(tracker, datasets: List[SingleObjectTrackingDataset_MemoryMapped], result_path: str, run_times: Optional[int]=None):

    output_path = os.path.join(result_path, tracker.get_name(), 'ope')
    os.makedirs(output_path, exist_ok=True)




class OnePassEvaluation:
    def __init__(self):
        pass

    def run(self, tracker, datasets, result_path):
        pass

