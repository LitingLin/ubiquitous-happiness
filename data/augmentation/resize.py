import cv2


class ImageResizer:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, bbox):
        h, w = image.shape[0:2]
        image = cv2.resize(image, (self.size, self.size))
        x_ratio = w / self.size
        y_ratio = h / self.size
        bbox = [bbox[0] / x_ratio, bbox[1] / y_ratio, bbox[2] / x_ratio, bbox[3] / y_ratio]
        return image, bbox
