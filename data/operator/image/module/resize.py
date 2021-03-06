import cv2


class ImageResizer:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, bbox):
        h, w = image.shape[0:2]
        return self.do_image(image), self.do_bbox(bbox, (w, h))

    def do_image(self, image):
        image = cv2.resize(image, (self.size, self.size))
        return image

    def do_bbox(self, bbox, image_size):
        return self.do_bbox_(bbox.copy(), image_size)

    def do_bbox_(self, bbox, image_size):
        w, h = image_size
        x_ratio = w / self.size
        y_ratio = h / self.size

        bbox[0] /= x_ratio
        bbox[1] /= y_ratio
        bbox[2] /= x_ratio
        bbox[3] /= y_ratio

        return bbox

    def reverse_do_bbox_(self, bbox, origin_image_size):
        w, h = origin_image_size

        x_ratio = w / self.size
        y_ratio = h / self.size

        bbox[0] *= x_ratio
        bbox[1] *= y_ratio
        bbox[2] *= x_ratio
        bbox[3] *= y_ratio

        return bbox
