from data.operator.image_and_bbox.half_pixel_center.torch_scale_and_translate import torch_scale_and_translate



def siamrpn_data_augmentation_pipeline(image, bbox):
    torch_scale_and_translate(image, bbox)