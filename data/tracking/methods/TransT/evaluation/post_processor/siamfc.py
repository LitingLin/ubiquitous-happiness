import numpy as np
import torch
import torch.nn.functional as F


class SiamFCTrackingPostProcessing:
    def __init__(self, response_up, scale_penalty, window_influence,
                 search_feat_size, search_size, device):
        self.response_up = response_up
        search_feat_w, search_feat_h = search_feat_size
        self.upscale_size = (search_feat_h * response_up, search_feat_w * response_up)
        self.scale_penalty = scale_penalty
        self.window_influence = window_influence
        self.search_feat_size = search_feat_size
        search_w, search_h = search_size
        self.network_stride = (search_h - 1) / (search_feat_h - 1), (search_w - 1) / (search_feat_w - 1)
        hann_window = np.outer(np.hanning(self.upscale_size[0]), np.hanning(self.upscale_size[1]))
        self.window = torch.tensor(hann_window, device=device)
        self.response_map_to_search_image_mapping_ratio = (self.upscale_size[0] - 1) / (search_h - 1), (self.upscale_size[1] - 1) / (search_w - 1)

    def __call__(self, response_map):
        s, c, h, w = response_map.shape
        assert c == 1
        assert w == self.search_feat_size[0] and h == self.search_feat_size[1]

        response_map = F.interpolate(response_map, self.upscale_size, mode='bicubic', align_corners=True)
        response_map = response_map.squeeze(1)
        upscaled_h, upscaled_w = self.upscale_size

        response_map[:s // 2] *= self.scale_penalty
        response_map[s // 2 + 1:] *= self.scale_penalty

        score, best_index = torch.max(response_map.view(-1), -1)
        best_scale_index = best_index // (upscaled_h * upscaled_w)

        response = response_map[best_scale_index]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.window_influence) * response + \
                   self.window_influence * self.window

        max_response_index = torch.argmax(response).cpu()
        max_response_index = (max_response_index // upscaled_w) * self.response_map_to_search_image_mapping_ratio[0],\
                             (max_response_index % upscaled_w) * self.response_map_to_search_image_mapping_ratio[1]

        return (best_scale_index.cpu(), *max_response_index), score.cpu().item()
