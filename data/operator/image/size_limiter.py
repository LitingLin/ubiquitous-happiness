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
