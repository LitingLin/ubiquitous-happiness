import torch
from data.tracking.processor.siamfc_curation import prepare_SiamFC_curation_with_position_augmentation, do_SiamFC_curation
from data.operator.bbox.spatial.vectorized.torch.utility.aligned.image import bbox_restrict_in_image_boundary_
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


def build_TransT_image_augmentation_transformer(color_jitter=0.4):
    # color jitter is enabled when not using AA
    if isinstance(color_jitter, (list, tuple)):
        # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
        # or 4 if also augmenting hue
        assert len(color_jitter) in (3, 4)
    else:
        # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
        color_jitter = (float(color_jitter),) * 3
    transform_list = [
        transforms.ColorJitter(*color_jitter),
        transforms.Normalize(
            mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
            std=torch.tensor(IMAGENET_DEFAULT_STD))
    ]
    return transforms.Compose(transform_list)


def TransT_training_data_preprocessing_pipeline(image, bbox, area_factor, output_size, scaling_jitter_factor,
                                                translation_jitter_factor, transform=None):
    curation_parameter, bbox = \
        prepare_SiamFC_curation_with_position_augmentation(bbox, area_factor, output_size,
                                                           scaling_jitter_factor, translation_jitter_factor)
    bbox_restrict_in_image_boundary_(bbox, output_size)

    if transform is not None:
        curation_parameter = curation_parameter.to(image.device)
        image = do_SiamFC_curation(image, output_size, curation_parameter)
        image = transform(image)
        curation_parameter = None

    return image, bbox, curation_parameter


def _transt_data_pre_processing_train_pipeline(image, gray_scale_transformer):
    if gray_scale_transformer is not None:
        image = gray_scale_transformer(image)
    return image.float() / 255.


def TransT_training_image_preprocessing(z_image, x_image, gray_scale_transformer, gray_scale_probability, rng_engine):
    if rng_engine.random() > gray_scale_probability:
        gray_scale_transformer = None
    if id(x_image) != id(z_image):
        z_image = _transt_data_pre_processing_train_pipeline(z_image, gray_scale_transformer)
        x_image = _transt_data_pre_processing_train_pipeline(x_image, gray_scale_transformer)
    else:
        x_image = z_image = _transt_data_pre_processing_train_pipeline(z_image, gray_scale_transformer)
    return z_image, x_image
