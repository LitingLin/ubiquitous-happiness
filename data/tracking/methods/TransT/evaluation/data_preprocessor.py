from data.tracking.processor.siamfc_curation import do_SiamFC_curation_CHW
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


def build_evaluation_transform():
    return transforms.Normalize(
            mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
            std=torch.tensor(IMAGENET_DEFAULT_STD))


class TransTEvaluationDataProcessor:
    def __init__(self, template_area_factor, search_area_factor, template_size, search_size, device, preprocessing_on_device, bounding_box_post_processor):
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor
        self.template_size = template_size
        self.search_size = search_size
        self.device = device
        self.preprocessing_on_device = preprocessing_on_device
        self.transform = build_evaluation_transform()
        self.bounding_box_post_processor = bounding_box_post_processor

    def initialize(self, image, bbox):
        if self.preprocessing_on_device:
            image = image.to(self.device, non_blocking=True)
        image = image.float() / 255.
        curated_template_image, _, self.image_mean, _, _, _ = do_SiamFC_curation_CHW(image, bbox, self.template_area_factor, self.template_size)

        curated_template_image = self.transform(curated_template_image)

        if not self.preprocessing_on_device:
            curated_template_image = curated_template_image.to(self.device, non_blocking=True)
        return curated_template_image

    def track(self, image, last_frame_bbox):
        if self.preprocessing_on_device:
            image = image.to(self.device, non_blocking=True)

        image = image.float() / 255.
        curated_search_image, _, _, curation_scaling, curation_source_center_point, curation_target_center_point = do_SiamFC_curation_CHW(image, last_frame_bbox, self.search_area_factor, self.search_size, self.image_mean)

        curated_search_image = self.transform(curated_search_image)

        if not self.preprocessing_on_device:
            curated_search_image = curated_search_image.to(self.device, non_blocking=True)

        c, h, w = image.shape
        self.last_frame_image_size = (w, h)
        self.last_frame_curation_scaling = curation_scaling
        self.last_frame_curation_source_center_point = curation_source_center_point
        self.last_frame_curation_target_center_point = curation_target_center_point
        return curated_search_image

    def get_bounding_box(self, bbox_normalized_cxcywh):
        return self.bounding_box_post_processor(bbox_normalized_cxcywh, self.last_frame_image_size, self.last_frame_curation_scaling, self.last_frame_curation_source_center_point, self.last_frame_curation_target_center_point)
