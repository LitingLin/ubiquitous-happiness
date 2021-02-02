from data.operator.image.numpy_pytorch_interop import image_numpy_to_torch


class SiamFCPostDataCombiner:
    def __init__(self, siamfc_data_processor, siamfc_label_generator):
        self.siamfc_data_processor = siamfc_data_processor
        self.siamfc_label_generator = siamfc_label_generator

    def __call__(self, z, z_bbox, x, x_bbox, is_positive):
        z, x = self.siamfc_data_processor(z, z_bbox, x, x_bbox)
        z = image_numpy_to_torch(z)
        x = image_numpy_to_torch(x)
        label = self.siamfc_label_generator(is_positive)
        return (z, x), label
