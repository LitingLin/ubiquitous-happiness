import numpy as np


class PrecisionPlotCriterion:
    def __init__(self, groundtruth_bboxes: np.ndarray, predicted_bboxes: np.ndarray):
        groundtruth_center = np.stack([groundtruth_bboxes[:, 0] + groundtruth_bboxes[:, 2] / 2,
                                            groundtruth_bboxes[:, 1] + groundtruth_bboxes[:, 3] / 2])
        predicted_center = np.stack([predicted_bboxes[:, 0] + predicted_bboxes[:, 2] / 2,
                                            predicted_bboxes[:, 1] + predicted_bboxes[:, 3] / 2])
        self.precision_error = np.linalg.norm(groundtruth_center - predicted_center, axis=0)

    def at(self, euclidean_distance_threshold: float):
        return (self.precision_error < euclidean_distance_threshold).sum() / self.precision_error.shape[0]
