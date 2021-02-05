import cv2


class ImageSizeLimiter:
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, image, bounding_box):
        h, w = image.shape[0:2]
        ratio = min([self.max_size / h, self.max_size / w, 1])
        if ratio == 1:
            return image, bounding_box
        else:
            dst_size = (int(round(ratio * w)), int(round(ratio * h)))
            bounding_box = [int(round(v * ratio)) for v in bounding_box]
            image = cv2.resize(image, dst_size)
            return image, bounding_box

    def _get_scaling_ratio(self, size):
        w, h = size
        return min([self.max_size / h, self.max_size / w, 1])

    def do_image(self, image):
        h, w = image.shape[0:2]
        ratio = self._get_scaling_ratio((w, h))
        if ratio == 1:
            return image

        dst_size = (int(round(ratio * w)), int(round(ratio * h)))
        image = cv2.resize(image, dst_size)
        return image

    def do_bbox(self, bounding_box, image_size):
        ratio = self._get_scaling_ratio(image_size)
        return [round(v * ratio) for v in bounding_box]
