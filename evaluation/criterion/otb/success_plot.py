import numpy as np


def _valid_gt_mask(gt):
    w_valid = gt[:, 2] > 0
    h_valid = gt[:, 3] > 0
    return w_valid & h_valid


def _iou(anno_bb, pred_bb):
    tl = np.maximum(pred_bb[:, :2], anno_bb[:, :2])
    br = np.minimum(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = (br - tl + 1.0).clip(0)

    # Area
    intersection = sz.prod(axis=1)
    union = pred_bb[:, 2:].prod(axis=1) + anno_bb[:, 2:].prod(axis=1) - intersection

    return intersection / union


class SuccessPlotCriterion:
    def __init__(self, groundtruth_bboxes: np.ndarray, predicted_bboxes: np.ndarray):
        mask = _valid_gt_mask(groundtruth_bboxes)
        self.iou = np.ones(groundtruth_bboxes.shape[0], dtype=np.float)
        predicted_bboxes = predicted_bboxes.copy()
        predicted_bboxes[:, 0] += 1
        predicted_bboxes[:, 1] += 1
        predicted_bboxes = np.rint(predicted_bboxes)
        self.iou[mask] = _iou(groundtruth_bboxes[mask], predicted_bboxes[mask])

    def at(self, threshold: float):
        assert 0 <= threshold <= 1.
        return (self.iou > threshold).sum() / self.iou.shape[0]

    def auc(self):
        thresholds_overlap = np.arange(0, 1.05, 0.05)
        success = np.zeros(len(thresholds_overlap))
        for i in range(len(thresholds_overlap)):
            success[i] = np.sum(self.iou > thresholds_overlap[i]) / float(self.iou.shape[0])
        score = np.mean(success)
        return score
