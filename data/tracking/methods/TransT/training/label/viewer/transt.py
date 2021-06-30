from Miscellaneous.Viewer.qt5_viewer import Qt5Viewer
from ._common import imagenet_denormalize
import math


class TransTDataPreprocessingVisualizer:
    def __init__(self, network_config: dict, train_config: dict, visualize_target):
        batch_size = train_config['data']['sampler'][visualize_target]['batch_size']
        self.n_vertical_canvas = math.floor(math.sqrt(batch_size))
        self.n_horizontal_canvas = math.ceil(batch_size / self.n_vertical_canvas)
        self.imagenet_normalized = network_config['data']['imagenet_normalization']

    def on_create(self):
        viewer = Qt5Viewer(n_vertical_canvas=self.n_vertical_canvas, n_horizontal_canvas=self.n_horizontal_canvas)
        self.process_label = viewer.get_control_region().new_label()
        self.data_label = viewer.get_control_region().new_label()
        self.viewer = viewer
        return viewer

    def on_data(self, data):
        (z_image_batch, x_image_batch), \
        (num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices_batch,
         target_class_label_vector_batch, target_bounding_box_label_matrix_batch), \
        is_positives, context = data
        assert context is None
        z_qimages = []
        x_qimages = []

        if self.imagenet_normalized:
            imagenet_denormalize(z_image_batch, True)
            imagenet_denormalize(x_image_batch, True)

        z_image_batch = z_image_batch.permute(0, 2, 3, 1)  # N, C, H, W => N, H, W, C
        x_image_batch = x_image_batch.permute(0, 2, 3, 1)  # N, C, H, W => N, H, W, C

        for z_image, x_image in zip(z_image_batch, x_image_batch):
            z_image



    def on_draw(self, data):
        pass
