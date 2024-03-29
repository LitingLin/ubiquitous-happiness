import torch


class TransTExp1TrackingPostProcessing:
    def __init__(self, enable_gaussian_score_map_penalty, with_quality_assessment, search_feat_size, device, window_penalty_ratio=None):
        self.enable_gaussian_score_map_penalty = enable_gaussian_score_map_penalty
        self.with_quality_assessment = with_quality_assessment
        self.search_feat_size = search_feat_size

        if enable_gaussian_score_map_penalty:
            # window = np.outer(np.hanning(search_feat_size[1]), np.hanning(search_feat_size[0]))
            # self.window = torch.tensor(window.flatten(), device=device)
            self.window = torch.flatten(torch.outer(torch.hann_window(search_feat_size[1], periodic=False, device=device),
                                                    torch.hann_window(search_feat_size[0], periodic=False, device=device)))

            self.window_penalty_ratio = window_penalty_ratio

    def __call__(self, network_output):
        class_score_map, bounding_box_regression_map, quality_assessment = network_output
        class_score_map = class_score_map.squeeze(0)

        if self.enable_gaussian_score_map_penalty:
            # window penalty
            class_score_map = class_score_map * (1 - self.window_penalty_ratio) + \
                     self.window * self.window_penalty_ratio
        if self.with_quality_assessment:
            class_score_map = class_score_map * quality_assessment.squeeze(0)

        confidence_score, best_idx = torch.max(class_score_map, 0)

        bounding_box_regression_map = bounding_box_regression_map.squeeze(0)
        bounding_box = bounding_box_regression_map[best_idx, :]
        return bounding_box.cpu(), confidence_score.cpu().item()
