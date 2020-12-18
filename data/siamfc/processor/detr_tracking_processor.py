from data.siamfc.curation import siamfc_z_curation
from data.operator.image.size_limiter import ImageSizeLimiter
from torchvision.transforms import ToTensor


class DETRTrackingProcessor:
    def __init__(self, z_context, z_size, x_size_limit):
        self.z_context = z_context
        self.z_size = z_size
        self.x_size_limiter = ImageSizeLimiter(x_size_limit)
        self.to_tensor = ToTensor()

    def __call__(self, z_image, z_bbox, x_image, x_bbox, _):
        z = siamfc_z_curation(z_image, z_bbox, self.z_context, self.z_size)
        x_image, x_bbox = self.x_size_limiter(x_image, x_bbox)

        x_bbox = [x_bbox[0] + x_bbox[2] / 2,
                               x_bbox[1] + x_bbox[3] / 2,
                               x_bbox[2], x_bbox[3]]

        x_h, x_w = x_image.shape[0:2]
        x_bbox = [x_bbox[0] / x_w, x_bbox[1] / x_h,
                               x_bbox[2] / x_w, x_bbox[3] / x_h]

        return self.to_tensor(z), self.to_tensor(x_image), x_bbox
