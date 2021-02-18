from data.siamfc.curation import siamfc_z_curation_with_bbox
import numpy as np


class SiamFCLikeCurationExemplarProcessor:
    def __init__(self, context, exemplar_size):
        self.context = context
        self.exemplar_size = exemplar_size

    def __call__(self, image, bbox):
        return siamfc_z_curation_with_bbox(image, bbox, self.context, self.exemplar_size)


def generate_mask_from_bbox(img, bbox):
    h, w = img.shape[0: 2]
    mask = np.ones((h, w), dtype=np.bool)
    mask[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]] = False
    return mask


class SiamFCLikeCurationExemplar_MaskGenerating_Processor:
    def __init__(self, context, exemplar_size):
        self.context = context
        self.exemplar_size = exemplar_size

    def __call__(self, image, bbox):
        image, bbox = siamfc_z_curation_with_bbox(image, bbox, self.context, self.exemplar_size)
        return image, generate_mask_from_bbox(image, bbox)
