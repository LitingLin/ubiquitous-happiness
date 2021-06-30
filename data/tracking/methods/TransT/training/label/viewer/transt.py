from Miscellaneous.Viewer.qt5_viewer import Qt5Viewer, QPixmap
from ._common import imagenet_denormalize, tensor_list_to_cpu
import math
from Miscellaneous.qt_numpy_interop import numpy_rgb888_to_qimage


class TransTDataPreprocessingVisualizer:
    def __init__(self, network_config: dict, train_config: dict, visualize_target, overwrite_batch_size = None):
        if overwrite_batch_size is None:
            batch_size = train_config['data']['sampler'][visualize_target]['batch_size']
        else:
            batch_size = overwrite_batch_size
        self.n_vertical_subplots = math.floor(math.sqrt(batch_size))
        self.n_horizontal_subplots = math.ceil(batch_size / self.n_vertical_subplots)
        self.imagenet_normalized = network_config['data']['imagenet_normalization']

    def on_create(self):
        viewer = Qt5Viewer(n_vertical_subplots=self.n_vertical_subplots, n_horizontal_subplots=self.n_horizontal_subplots)
        for i_row in range(self.n_vertical_subplots):
            for i_col in range(self.n_horizontal_subplots):
                subplot = viewer.get_subplot(i_row, i_col)
                subplot.create_canvas()
                subplot.create_canvas()
                subplot.create_canvas()
                subplot.create_canvas()
                subplot.create_label()
        self.process_label = viewer.get_control_region().new_label()
        self.data_label = viewer.get_control_region().new_label()
        self.viewer = viewer
        return viewer




    def on_data(self, data):
        (z_image_batch, x_image_batch), \
        target, \
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
            z_qimages.append(QPixmap(numpy_rgb888_to_qimage(z_image.cpu().numpy())))
            x_qimages.append(QPixmap(numpy_rgb888_to_qimage(x_image.cpu().numpy())))

        return (z_qimages, x_qimages), tensor_list_to_cpu(target), is_positives.cpu()

    def on_draw(self, data):
        (z_qimages, x_qimages), \
        (num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices_batch,
         target_class_label_vector_batch, target_bounding_box_label_matrix_batch), \
        is_positives = data

