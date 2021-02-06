import torch
from data.detr_tracking_variants.siam_encoder.processor.mask_generator import generate_mask_from_bbox


class DETRTracker:
    def __init__(self, network, device, data_processor):
        network.to(device)
        network.eval()
        self.network = network
        self.device = device
        self.data_processor = data_processor

    def initialize(self, image, bbox):
        z, z_bbox = self.data_processor.get_z(image, bbox)
        z_mask = generate_mask_from_bbox(z, z_bbox)
        z = z.unsqueeze(0)
        z_mask = z_mask.unsqueeze(0)
        z = z.to(self.device)
        z_mask = z_mask.to(self.device)
        with torch.no_grad():
            self.z_feat, self.z_feat_mask, self.z_feat_pos = self.network.inference_template(z, z_mask)

    def track(self, image):
        h, w = image.shape[0:2]
        x = self.data_processor.get_x_image(image)
        x = x.unsqueeze(0)
        x = x.to(self.device)
        with torch.no_grad():
            x_bbox_predicted = self.network.inference_instance(self.z_feat, self.z_feat_mask, self.z_feat_pos, x)
        x_bbox_predicted = x_bbox_predicted.cpu()
        x_bbox = self.data_processor.reverse_x_bbox(x_bbox_predicted[0], (w, h))
        return x_bbox
