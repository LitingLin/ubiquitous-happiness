from data.tracking.processor.siamfc_curation import prepare_SiamFC_curation, do_SiamFC_curation
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


def build_evaluation_transform():
    return transforms.Normalize(
            mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
            std=torch.tensor(IMAGENET_DEFAULT_STD))


class TransTEvaluationDataProcessor:
    def __init__(self, template_area_factor, search_area_factor, template_size, search_size,
                 interpolation_mode,
                 device, preprocessing_on_device, bounding_box_post_processor):
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor
        self.template_size = template_size
        self.search_size = search_size
        self.interpolation_mode = interpolation_mode
        self.device = device
        self.preprocessing_on_device = preprocessing_on_device
        self.transform = build_evaluation_transform()
        self.bounding_box_post_processor = bounding_box_post_processor

    def initialize(self, image, bbox):
        curation_parameter, _ = prepare_SiamFC_curation(bbox, self.template_area_factor, self.template_size)
        if self.preprocessing_on_device:
            image = image.to(self.device, non_blocking=True)
            curation_parameter_device = curation_parameter.to(self.device, non_blocking=True)
        else:
            curation_parameter_device = curation_parameter
        image = image.float() / 255.

        curated_template_image, self.image_mean = do_SiamFC_curation(image, self.template_size, curation_parameter_device, self.interpolation_mode)
        curated_template_image = self.transform(curated_template_image)

        if not self.preprocessing_on_device:
            curated_template_image = curated_template_image.to(self.device, non_blocking=True)
        return curated_template_image.unsqueeze(0)

    def track(self, image, last_frame_bbox):
        curation_parameter, _ = prepare_SiamFC_curation(last_frame_bbox, self.search_area_factor, self.search_size)
        if self.preprocessing_on_device:
            image = image.to(self.device, non_blocking=True)
            curation_parameter_device = curation_parameter.to(self.device, non_blocking=True)
        else:
            curation_parameter_device = curation_parameter

        image = image.float() / 255.

        curated_search_image, _ = do_SiamFC_curation(image, self.search_size, curation_parameter_device, self.interpolation_mode, self.image_mean)
        curated_search_image = self.transform(curated_search_image)

        if not self.preprocessing_on_device:
            curated_search_image = curated_search_image.to(self.device, non_blocking=True)

        c, h, w = image.shape
        self.last_frame_image_size = (w, h)
        self.last_frame_curation_parameter = curation_parameter
        return curated_search_image.unsqueeze(0)

    def get_bounding_box(self, bbox_normalized):
        return self.bounding_box_post_processor(bbox_normalized, self.last_frame_image_size, self.last_frame_curation_parameter)
