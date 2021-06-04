import numpy as np
import torch
import torch.nn.functional as F


class TransTTrackingPostProcessing:
    def __init__(self, search_feat_size, window_penalty, device):
        window = np.outer(np.hanning(search_feat_size[1]), np.hanning(search_feat_size[0]))
        self.window = torch.tensor(window.flatten(), device=device)

        self.window_penalty = window_penalty

    def __call__(self, network_output):
        class_score_map, bounding_box_regression_map = network_output
        class_score_map = class_score_map.squeeze(0)
        class_score_map = F.softmax(class_score_map, dim=0)[0, :]

        # window penalty
        pscore = class_score_map * (1 - self.window_penalty) + \
                 self.window * self.window_penalty

        best_idx = torch.argmax(pscore)

        bounding_box_regression_map = bounding_box_regression_map.squeeze(0)
        bounding_box = bounding_box_regression_map[best_idx, :]
        bounding_box = bounding_box.cpu()
        return bounding_box
