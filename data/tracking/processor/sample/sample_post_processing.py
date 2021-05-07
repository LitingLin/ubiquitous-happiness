from data.operator.bbox.validity import bbox_is_valid
import numpy as np


def _generate_random_bounding_box(image_size, rng_engine):
    bounding_box = rng_engine.rand(4)
    x1 = np.minimum(bounding_box[0], bounding_box[2]) * image_size[0]
    x2 = np.maximum(bounding_box[0], bounding_box[2]) * image_size[0]
    y1 = np.minimum(bounding_box[1], bounding_box[1]) * image_size[1]
    y2 = np.maximum(bounding_box[1], bounding_box[3]) * image_size[1]
    return np.stack((x1, y1, x2, y2))


class SiameseSamplePairPostProcessor:
    def __init__(self, negative_sample_random_bounding_box_generating_ratio, rng_engine):
        self.negative_sample_random_bounding_box_generating_ratio = negative_sample_random_bounding_box_generating_ratio
        self.rng_engine = rng_engine

    def __call__(self, data, is_positive):
        '''
            Assuming invalid bounding box means occlusion or out of view, not the sparse annotation case
            positive sample allow single image
        '''
        assert data is not None and 0 < len(data) < 3
        if is_positive:
            bounding_box_validity_count = len([_ for _, _, _, validity in data if validity])
            if bounding_box_validity_count != len(data):
                is_positive = False

        for index in range(len(data)):
            image_path, image_size, bounding_box, bounding_box_validity = data[index]
            if not is_positive:
                if bounding_box is None or not bbox_is_valid(bounding_box) or (
                        self.negative_sample_random_bounding_box_generating_ratio > 0 and self.rng_engine.rand() < self.negative_sample_random_bounding_box_generating_ratio):
                    bounding_box = _generate_random_bounding_box(image_size, self.rng_engine)
            data[index] = (image_path, bounding_box)
        return data, is_positive


class TripletSamplePairPostProcessor:
    def __init__(self, negative_sample_random_bounding_box_generating_ratio, rng_engine):
        self.negative_sample_random_bounding_box_generating_ratio = negative_sample_random_bounding_box_generating_ratio
        self.rng_engine = rng_engine

    def __call__(self, data, is_positive):
        assert data is not None and 0 < len(data) < 4
        if is_positive:
            bounding_box_validity_count = len([_ for _, _, _, validity in data if validity])
            if len(data) < 3:
                if bounding_box_validity_count != len(data):
                    is_positive = False
            else:
                if not (data[0][3] and data[2][3]):
                    is_positive = False

        for index in range(len(data)):
            image_path, image_size, bounding_box, bounding_box_validity = data[index]
            if bounding_box is None or not bbox_is_valid(bounding_box):
                bounding_box = _generate_random_bounding_box(image_size, self.rng_engine)
            if not is_positive and self.negative_sample_random_bounding_box_generating_ratio > 0 and self.rng_engine.rand() < self.negative_sample_random_bounding_box_generating_ratio:
                bounding_box = _generate_random_bounding_box(image_size, self.rng_engine)
            data[index] = (image_path, bounding_box, bounding_box_validity)
        return data, is_positive
