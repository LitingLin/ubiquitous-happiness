class TransTTracker(object):
    def __init__(self, model, device, data_processor, network_post_processor):
        model = model.to(device)
        model.eval()
        self.net = model
        self.device = device
        self.data_processor = data_processor
        self.network_post_processor = network_post_processor

    def initialize(self, image, bbox):
        self.last_frame_object_bbox = bbox
        curated_template_image = self.data_processor.initialize(image, bbox)
        curated_template_image = curated_template_image.unsqueeze()
        self.z = self.net.template(curated_template_image)

    def track(self, image):
        curated_search_image = self.data_processor.track(image, self.last_frame_object_bbox)
        curated_search_image = curated_search_image.unsqueeze()
        net_output = self.net.track(self.z, curated_search_image)
        bounding_box = self.network_post_processor(net_output)
        bounding_box_predicted, self.last_frame_object_bbox = self.data_processor.get_bounding_box(bounding_box)
        return bounding_box_predicted
