import torch


class GFocalTrackingPostProcessing:
    def __init__(self, enable_gaussian_score_map_penalty, search_feat_size, device, window_penalty_ratio=None):
        self.enable_gaussian_score_map_penalty = enable_gaussian_score_map_penalty
        self.search_feat_size = search_feat_size

        if enable_gaussian_score_map_penalty:
            self.window = torch.flatten(torch.outer(torch.hann_window(search_feat_size[1], periodic=False, device=device),
                                                    torch.hann_window(search_feat_size[0], periodic=False, device=device)))

            self.window_penalty_ratio = window_penalty_ratio

    def __call__(self, network_output):
        class_score_map, predicted_bbox, _ = network_output  # shape: (N, 1, H, W), (N, H, W, 4)
        N, C, H, W = class_score_map.shape
        assert N == 1 and C == 1
        class_score_map = class_score_map.view(H * W)

        if self.enable_gaussian_score_map_penalty:
            # window penalty
            class_score_map = class_score_map * (1 - self.window_penalty_ratio) + \
                     self.window * self.window_penalty_ratio

        confidence_score, best_idx = torch.max(class_score_map, 0)

        predicted_bbox = predicted_bbox.view(H * W, 4)
        bounding_box = predicted_bbox[best_idx, :]
        return bounding_box.cpu(), confidence_score.cpu().item()
