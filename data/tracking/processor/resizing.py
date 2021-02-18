import numpy as np
import cv2


class RandomResizing_KeepingAspect_Processor:
    def __init__(self, min_total_size, max_total_size):
        self.min_total_size = min_total_size
        self.max_total_size = max_total_size

    def __call__(self, image: np.ndarray, bbox: np.ndarray):
        h, w = image.shape[0:2]

        target_size = np.random.randint(self.min_total_size, self.max_total_size, dtype=np.float)
        scaling_ratio = np.sqrt(target_size / (h * w)).item()
        target_h = int(round(scaling_ratio * h))
        target_w = int(round(scaling_ratio * w))

        target_bbox = np.around(bbox * scaling_ratio)

        return cv2.resize(image, (target_w, target_h)), target_bbox
