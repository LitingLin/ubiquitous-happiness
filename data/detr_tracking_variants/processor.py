class DETRTrackingProcessor:
    def __init__(self, z_processor, x_processor, z_post_processor, x_post_processor, output_organizer):
        self.z_processor = z_processor
        self.x_processor = x_processor
        self.z_post_processor = z_post_processor
        self.x_post_processor = x_post_processor
        self.output_organizer = output_organizer

    def __call__(self, z, z_bbox, x, x_bbox, _):
        z_output = self.z_processor(z, z_bbox)
        z_output = self.z_post_processor(z_output)
        x_output = self.x_processor(x, x_bbox)
        x_output = self.x_post_processor(x_output)
        return self.output_organizer(z_output, x_output)


class SimpleOrganizer:
    def __call__(self, z_output, x_output):
        return (*z_output, *x_output)


class DETRTrackingEvaluationProcessor:
    def __init__(self, z_processor, x_processor, init_post_processor, track_post_processor, result_processor):
        self.z_processor = z_processor
        self.x_processor = x_processor
        self.init_post_processor = init_post_processor
        self.track_post_processor = track_post_processor
        self.result_processor = result_processor

    def do_init(self, z, z_bbox):
        z_output = self.z_processor(z, z_bbox)
        return self.init_post_processor(z_output)

    def do_track(self, x):
        x_output = self.x_processor(x)
        return self.track_post_processor(x_output)

    def do_result(self, result, image_size):
        return self.result_processor(result, image_size)
