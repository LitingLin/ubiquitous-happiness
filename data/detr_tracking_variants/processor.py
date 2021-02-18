class DETRTrackingProcessor:
    def __init__(self, z_processor, x_processor, output_organizer):
        self.z_processor = z_processor
        self.x_processor = x_processor
        self.output_organizer = output_organizer

    def __call__(self, z, z_bbox, x, x_bbox, _):
        z_output = self.z_processor(z, z_bbox)
        x_output = self.x_processor(x, x_bbox)
        return self.output_organizer(z_output, x_output)
