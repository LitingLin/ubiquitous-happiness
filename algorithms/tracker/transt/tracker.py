import numpy as np
import torch.nn.functional as F


class TransTTracker(object):
    def __init__(self, model, device, window_penalty, template_size, search_size, search_feat_size, template_area_factor, search_area_factor):
        model = model.to(device)
        model.eval()
        self.net = model
        self.device = device
        self.window_penalty = window_penalty
        self.template_size = template_size
        self.search_size = search_size
        self.search_feat_size = search_feat_size
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor
        from data.TransT.pipeline import build_evaluation_transform
        self.image_transform = build_evaluation_transform()

    def _convert_score(self, score):
        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

    def _convert_bbox(self, delta):
        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()
        return delta

    def initialize(self, image, bbox):
        window = np.outer(np.hanning(self.search_feat_size[1]), np.hanning(self.search_feat_size[0]))
        self.window = window.flatten()
        self.object_bbox = bbox


        from data.TransT.pipeline import get_scaling_and_translation_parameters, transt_preprocessing_pipeline

        curation_scaling, curation_source_center_point, curation_target_center_point = \
            get_scaling_and_translation_parameters(self.object_bbox, self.template_area_factor, self.template_size)

        curated_template_image, _, self.image_mean = \
            transt_preprocessing_pipeline(image, self.object_bbox, self.template_size,
                                          curation_scaling, curation_source_center_point, curation_target_center_point,
                                          None, self.image_transform)
        curated_template_image = curated_template_image.to(self.device)
        # initialize template feature
        self.z = self.net.template(curated_template_image)

    def track(self, image):
        n, h, w, c = image.shape
        from data.TransT.pipeline import get_scaling_and_translation_parameters, transt_preprocessing_pipeline
        curation_scaling, curation_source_center_point, curation_target_center_point = \
            get_scaling_and_translation_parameters(self.object_bbox, self.search_area_factor, self.search_size)

        curated_search_image, curated_search_image_object_bbox, _ = \
            transt_preprocessing_pipeline(image, self.object_bbox, self.search_size,
                                          curation_scaling, curation_source_center_point, curation_target_center_point,
                                          self.image_mean, self.image_transform)

        curated_search_image = curated_search_image.to(self.device)
        # track
        predicted_classes, predicted_boxes = self.net.track(self.z, curated_search_image)
        score = self._convert_score(predicted_classes)
        pred_bbox = self._convert_bbox(predicted_boxes)

        # window penalty
        pscore = score * (1 - self.window_penalty) + \
                 self.window * self.window_penalty

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx]

        from data.TransT.label_generation import get_bounding_box_from_label
        bbox = get_bounding_box_from_label(bbox, self.search_size)
        from data.operator.bbox.spatial.scale_and_translate import bbox_scale_and_translate
        bbox = bbox_scale_and_translate(bbox, [1.0 / curation_scaling_ for curation_scaling_ in curation_scaling], curation_target_center_point, curation_source_center_point)
        from data.operator.bbox.spatial.utility.aligned.image import bounding_box_fit_in_image_boundary
        bbox = bounding_box_fit_in_image_boundary(bbox, (w, h))
        self.object_bbox = bbox

        return bbox
