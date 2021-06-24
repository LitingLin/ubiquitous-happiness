import numpy as np
import torch
import torch.nn.functional as F


class SiamFCTrackingPostProcessing:
    def __init__(self, response_sz, response_up, scale_penalty,
                 enable_gaussian_score_map_penalty, search_feat_size, device, window_penalty_ratio=None):
        self.upscale_sz = response_sz * response_up
        self.scale_penalty = scale_penalty
        self.enable_gaussian_score_map_penalty = enable_gaussian_score_map_penalty
        self.search_feat_size = search_feat_size

        if enable_gaussian_score_map_penalty:
            window = np.outer(np.hanning(search_feat_size[1]), np.hanning(search_feat_size[0]))
            self.window = torch.tensor(window.flatten(), device=device)
            self.window_penalty_ratio = window_penalty_ratio

    def __call__(self, response_map):
        s, c, h, w = response_map.shape

        response_map = F.interpolate(response_map, self.upscale_sz, mode='bicubic', align_corners=True)
        upscaled_h, upscaled_w = self.upscale_sz

        response_map[:s // 2] *= self.scale_penalty
        response_map[s // 2 + 1:] *= self.scale_penalty

        score, best_index = torch.max(response_map.view(-1), -1)
        best_scale_index = best_index // (c * upscaled_h * upscaled_w)


        class_score_map, bounding_box_regression_map, quality_assessment = network_output
        class_score_map = class_score_map.squeeze(0)

        if self.enable_gaussian_score_map_penalty:
            # window penalty
            class_score_map = class_score_map * (1 - self.window_penalty_ratio) + \
                     self.window * self.window_penalty_ratio

        confidence_score, best_idx = torch.max(class_score_map, 0)

        bounding_box_regression_map = bounding_box_regression_map.squeeze(0)
        bounding_box = bounding_box_regression_map[best_idx, :]
        return bounding_box.cpu(), confidence_score.cpu().item()
