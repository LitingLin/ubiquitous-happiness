from ._common import _BaseFilter


class DataCleaner_BoundingBox(_BaseFilter):
    def __call__(self, bounding_box, image_size):
        assert len(bounding_box) == 4
        assert len(image_size) == 2
        bounding_box = [bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]]
        if bounding_box[0] < 0:
            bounding_box[2] += bounding_box[0]
            bounding_box[0] = 0
        if bounding_box[1] < 0:
            bounding_box[3] += bounding_box[1]
            bounding_box[1] = 0
        if bounding_box[0] >= image_size[0]:
            return None
        if bounding_box[1] >= image_size[1]:
            return None
        if bounding_box[2] <= 0:
            return None
        if bounding_box[3] <= 0:
            return None
        if bounding_box[0] + bounding_box[2] > image_size[0]:
            bounding_box[2] = image_size[0] - bounding_box[0]
        if bounding_box[1] + bounding_box[3] > image_size[1]:
            bounding_box[3] = image_size[1] - bounding_box[1]
        return bounding_box
