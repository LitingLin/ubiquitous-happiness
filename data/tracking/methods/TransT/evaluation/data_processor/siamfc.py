from data.tracking.processor.siamfc_curation import prepare_SiamFC_curation, do_SiamFC_curation
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from data.operator.point.scale_and_translate import xy_point_scale_and_translate
from data.operator.bbox.spatial.cxcywh2xyxy import bbox_cxcywh2xyxy


def build_evaluation_transform():
    return transforms.Normalize(
            mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
            std=torch.tensor(IMAGENET_DEFAULT_STD))


class SiamFCEvaluationDataProcessor:
    def __init__(self, template_area_factor, search_area_factor, template_size, search_size,
                 scale_num, scale_step, scale_lr,
                 interpolation_mode,
                 device, preprocessing_on_device):
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor
        self.template_size = template_size
        self.search_size = search_size
        self.interpolation_mode = interpolation_mode
        self.device = device
        self.preprocessing_on_device = preprocessing_on_device
        self.transform = build_evaluation_transform()
        self.scale_factors = scale_step ** torch.linspace(-(scale_num // 2), scale_num // 2, scale_num)
        self.scale_lr = scale_lr

    def initialize(self, image, bbox):
        self.object_wh = bbox[2].item(), bbox[3].item()
        curation_parameter, _ = prepare_SiamFC_curation(bbox, self.template_area_factor, self.template_size)
        if self.preprocessing_on_device:
            image = image.to(self.device, non_blocking=True)
            curation_parameter_device = curation_parameter.to(self.device, non_blocking=True)
        else:
            curation_parameter_device = curation_parameter
        image = image.float() / 255.

        curated_template_image, self.image_mean = do_SiamFC_curation(image, self.template_size, curation_parameter_device, self.interpolation_mode)
        curated_template_image = curated_template_image.unsqueeze(0)
        curated_template_image = self.transform(curated_template_image)

        if not self.preprocessing_on_device:
            curated_template_image = curated_template_image.to(self.device, non_blocking=True)
        return curated_template_image

    def track(self, image, last_frame_bbox):
        c, h, w = image.shape
        scale_steps = len(self.scale_factors)

        curation_parameter, _ = prepare_SiamFC_curation(last_frame_bbox, self.search_area_factor, self.search_size)
        curation_parameter = curation_parameter.unsqueeze(0).repeat(scale_steps, 1, 1)
        curation_parameter[:, 0, :] *= self.scale_factors.view(scale_steps, 1)

        if self.preprocessing_on_device:
            image = image.to(self.device, non_blocking=True)
            curation_parameter_device = curation_parameter.to(self.device, non_blocking=True)
        else:
            curation_parameter_device = curation_parameter

        image = image.float() / 255.

        curated_search_image = torch.empty([scale_steps, c, *self.search_size], dtype=image.dtype, device=image.device)

        for i_scale_level in range(scale_steps):
            _, _ = do_SiamFC_curation(image, self.search_size, curation_parameter_device[i_scale_level], self.interpolation_mode, self.image_mean, curated_search_image[i_scale_level, ...])
        curated_search_image = self.transform(curated_search_image)

        if not self.preprocessing_on_device:
            curated_search_image = curated_search_image.to(self.device, non_blocking=True)

        self.last_frame_curation_parameter = curation_parameter
        return curated_search_image

    def get_bounding_box(self, response_peak_position):
        scale_index, response_y, response_x = response_peak_position

        curation_parameter = self.last_frame_curation_parameter[scale_index]
        scaling, input_center, output_center = curation_parameter
        object_center = xy_point_scale_and_translate((response_x, response_y), 1 / scaling, output_center, input_center)
        scale = ((1 - self.scale_lr) * 1.0 + self.scale_lr * self.scale_factors[scale_index]).item()
        self.object_wh = self.object_wh[0] * scale, self.object_wh[1] * scale

        bounding_box = object_center[0].item(), object_center[1].item(), self.object_wh[0], self.object_wh[1]
        bounding_box = bbox_cxcywh2xyxy(bounding_box)

        return bounding_box, bounding_box
