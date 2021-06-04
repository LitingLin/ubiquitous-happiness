class TransTTracker(object):
    def __init__(self, model, device, data_preprocessor, label_postprocessor, bounding_box_postprocessor):
        model = model.to(device)
        model.eval()
        self.net = model
        self.device = device
        self.data_preprocessor = data_preprocessor
        self.label_postprocessor = label_postprocessor
        self.bounding_box_postprocessor = bounding_box_postprocessor

    def initialize(self, image, bbox):
        self.last_frame_object_bbox = bbox
        curated_template_image = self.data_preprocessor.initialize(image, bbox)
        self.z = self.net.template(curated_template_image)

    def track(self, image):
        c, h, w = image.shape
        curated_search_image, curation_scaling, curation_source_center_point, curation_target_center_point = self.data_preprocessor.track(image, self.last_frame_object_bbox)
        net_output = self.net.track(self.z, curated_search_image)
        bounding_box = self.label_postprocessor(net_output)
        bounding_box_predicted, self.last_frame_object_bbox = self.bounding_box_postprocessor(bounding_box, (w, h), curation_scaling, curation_source_center_point, curation_target_center_point)
