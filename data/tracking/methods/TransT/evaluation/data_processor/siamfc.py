from data.tracking.processor.siamfc_curation import prepare_SiamFC_curation, do_SiamFC_curation
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


def build_evaluation_transform():
    return transforms.Normalize(
            mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
            std=torch.tensor(IMAGENET_DEFAULT_STD))


class SiamFCEvaluationDataProcessor:
    def __init__(self, template_area_factor, search_area_factor, template_size, search_size,
                 scale_num, scale_step,
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
        self.scale_factors = scale_step ** torch.linspace(-(scale_num // 2), scale_num // 2, scale_num)

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
        return curated_template_image

    def track(self, image, last_frame_bbox):
        c, h, w = image.shape
        scale_steps = len(self.scale_factors)

        curation_parameter, _ = prepare_SiamFC_curation(last_frame_bbox, self.search_area_factor, self.search_size)
        curation_parameter.unsqueeze(0).repeat(scale_steps, 1, 1)
        curation_parameter[:, 0, :] *= self.scale_factors.view(scale_steps, 1)

        if self.preprocessing_on_device:
            image = image.to(self.device, non_blocking=True)
            curation_parameter_device = curation_parameter.to(self.device, non_blocking=True)
        else:
            curation_parameter_device = curation_parameter

        image = image.float() / 255.

        curated_search_image = torch.empty([scale_steps, c, *self.search_size], dtype=image.dtype, device=image.device)

        for i_scale_level in range(scale_steps):
            _, _ = do_SiamFC_curation(image, self.search_size, curation_parameter_device, self.interpolation_mode, self.image_mean, curated_search_image[i_scale_level, ...])
        curated_search_image = self.transform(curated_search_image)

        if not self.preprocessing_on_device:
            curated_search_image = curated_search_image.to(self.device, non_blocking=True)

        self.last_frame_image_size = (w, h)
        self.last_frame_curation_parameter = curation_parameter
        return curated_search_image

    def get_bounding_box(self, bbox_normalized_cxcywh):
        return self.bounding_box_post_processor(bbox_normalized_cxcywh, self.last_frame_image_size, self.last_frame_curation_parameter)
