import numpy as np
import torch
from Utils.detr_misc import NestedTensor


class DETRDetector:
    def __init__(self, detr, postprocessor, transforms, device):
        self.detr = detr
        self.postprocessor = postprocessor
        self.transforms = transforms
        self.device = device

    def __call__(self, image: np.ndarray):
        image = torch.tensor(image)
        _, w, h = image.size()
        mask = torch.ones((1, h, w), dtype=torch.bool, device=self.device)
