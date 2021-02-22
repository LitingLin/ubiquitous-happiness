import torch


class DETRTracker:
    def __init__(self, name, network, device, data_processor):
        self.name = name
        network.to(device)
        network.eval()
        self.network = network
        self.device = device
        self.data_processor = data_processor

    def get_name(self):
        return self.name

    def initialize(self, image, bbox):
        z, z_mask = self.data_processor.do_init(image, bbox)
        z = z.unsqueeze(0)
        z_mask = z_mask.unsqueeze(0)
        z = z.to(self.device)
        z_mask = z_mask.to(self.device)
        with torch.no_grad():
            self.z_feat, self.z_feat_mask, self.z_feat_pos = self.network.inference_template(z, z_mask)

    def track(self, image):
        h, w = image.shape[0:2]
        x = self.data_processor.do_track(image)
        x = x.unsqueeze(0)
        x = x.to(self.device)
        with torch.no_grad():
            x_bbox_predicted = self.network.inference_instance(self.z_feat, self.z_feat_mask, self.z_feat_pos, x)
        x_bbox_predicted = x_bbox_predicted.cpu()
        x_bbox = self.data_processor.do_result(x_bbox_predicted[0], (w, h))
        return x_bbox
