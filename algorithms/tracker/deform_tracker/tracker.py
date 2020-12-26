class DeformableTracker:
    def __init__(self, network, device, data_processor):
        network.to(device)
        self.network = network
        self.device = device
        self.data_processor = data_processor

    def initialize(self, image, bbox):
        z = self.data_processor.get_z(image, bbox)
        z = z.unsqueeze(0)
        self.z = z.to(self.device)

    def track(self, image):
        h, w = image.shape[0:2]
        x = self.data_processor.get_x_image(image)
        x = x.unsqueeze(0)
        x = x.to(self.device)
        x_bbox = self.network(self.z, x)
        x_bbox = self.data_processor.reverse_x_bbox(x_bbox[0], (w, h))
        return x_bbox
