import torch
import torch.nn.functional as F

from miscellanies.Viewer.qt5_viewer import Qt5Viewer, QPixmap, QPen, QColor
from ._common import imagenet_denormalize, tensor_list_to_cpu
import math
from miscellanies.qt_numpy_interop import numpy_rgb888_to_qimage
from data.operator.bbox.spatial.utility.aligned.normalize_v2 import bbox_denormalize
from data.operator.bbox.spatial.cxcywh2xyxy import bbox_cxcywh2xyxy
from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
from data.operator.bbox.transform.pixel_coordinate_system.mapping import bbox_pixel_coordinate_system_aligned_to_half_pixel_offset


def _bbox_format_convert(bbox):
    return bbox_xyxy2xywh(bbox_pixel_coordinate_system_aligned_to_half_pixel_offset(bbox))


class TransTDataPreprocessingVisualizer:
    def __init__(self, network_config: dict, train_config: dict, visualization_target):
        self.batch_size = train_config['data']['sampler'][visualization_target]['batch_size']
        self.label_size = network_config['head']['parameters']['input_size']
        self.n_vertical_subplots = math.floor(math.sqrt(self.batch_size))
        self.n_horizontal_subplots = math.ceil(self.batch_size / self.n_vertical_subplots)
        self.imagenet_normalized = network_config['data']['imagenet_normalization']
        self.bbox_pen = QPen(QColor(255, 0, 0, int(255 * 0.5)))

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
        self.viewer = viewer
        return viewer

    def on_data(self, data):
        (z_image_batch, x_image_batch), \
        (num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices_batch,
         target_class_label_vector_batch, target_bounding_box_label_matrix_batch), \
        miscellanies, context = data

        assert context is None

        z_qimages = []
        x_qimages = []

        N, x_C, x_H, x_W = x_image_batch.shape

        recovered_bbox = []

        if self.imagenet_normalized:
            imagenet_denormalize(z_image_batch, True)
            imagenet_denormalize(x_image_batch, True)

        assert N == self.batch_size
        if target_feat_map_indices_batch_id_vector is not None:
            label = torch.ones([N, x_C, *self.label_size], dtype=torch.float, device=target_feat_map_indices_batch_id_vector.device)
            label = label.view(N, x_C, -1)
            label[target_feat_map_indices_batch_id_vector, :, target_feat_map_indices_batch] = 0.5
            label = label.view(N, x_C, *self.label_size)
            label = F.interpolate(label, (x_H, x_W))
            x_image_batch = x_image_batch * label

            assert torch.all(target_class_label_vector_batch[target_feat_map_indices_batch_id_vector, target_feat_map_indices_batch] == 0)

            min_index = 0
            for batch in range(N):
                indices = torch.nonzero(target_feat_map_indices_batch_id_vector == batch).flatten()
                if len(indices) == 0:
                    recovered_bbox.append(None)
                    continue
                assert indices[0] == min_index
                min_index += len(indices)
                bounding_boxes = target_bounding_box_label_matrix_batch[indices, :]
                assert (bounding_boxes - bounding_boxes[0]).sum() == 0
                bounding_box = bounding_boxes[0]
                bounding_box = bounding_box.tolist()
                bounding_box = bbox_cxcywh2xyxy(bounding_box)
                bounding_box = bbox_denormalize(bounding_box, (x_W, x_H))
                recovered_bbox.append(bounding_box)
        else:
            recovered_bbox = [None] * N

        z_image_batch = z_image_batch.permute(0, 2, 3, 1)  # N, C, H, W => N, H, W, C
        x_image_batch = x_image_batch.permute(0, 2, 3, 1)  # N, C, H, W => N, H, W, C

        z_image_batch = z_image_batch.round_().clip_(0, 255).to(torch.uint8)
        x_image_batch = x_image_batch.round_().clip_(0, 255).to(torch.uint8)

        for z_image, x_image in zip(z_image_batch, x_image_batch):
            z_qimages.append(QPixmap(numpy_rgb888_to_qimage(z_image.cpu().numpy())))
            x_qimages.append(QPixmap(numpy_rgb888_to_qimage(x_image.cpu().numpy())))

        z_origin_qimages = []
        x_origin_qimages = []
        z_origin_bboxes = []
        x_origin_bboxes = []
        is_positive_samples = []
        for miscellany in miscellanies:
            z_origin = miscellany['z']  # C, H, W
            x_origin = miscellany['x']  # C, H, W
            z_origin_bbox = miscellany['z_bbox']
            x_origin_bbox = miscellany['x_bbox']
            is_positive_sample = miscellany['is_positive_sample']
            z_origin = z_origin.permute(1, 2, 0)  # C, H, W => H, W, C
            x_origin = x_origin.permute(1, 2, 0)  # C, H, W => H, W, C
            z_origin_qimages.append(QPixmap(numpy_rgb888_to_qimage(z_origin.cpu().numpy())))
            x_origin_qimages.append(QPixmap(numpy_rgb888_to_qimage(x_origin.cpu().numpy())))
            z_origin_bboxes.append(z_origin_bbox.tolist())
            x_origin_bboxes.append(x_origin_bbox.tolist())
            is_positive_samples.append(is_positive_sample)

        return (z_qimages, x_qimages, recovered_bbox), (z_origin_qimages, z_origin_bboxes, x_origin_qimages, x_origin_bboxes, is_positive_samples)

    def on_draw(self, data):
        (z_curated_qimages, x_curated_qimages, recovered_x_curated_bbox), (z_batch, z_bboxes, x_batch, x_bboxes, is_positive_samples) = data
        for index, (z, z_bbox, x, x_bbox, z_curated, x_curated, x_curated_bbox, is_positive_sample) in enumerate(zip(z_batch, z_bboxes, x_batch, x_bboxes, z_curated_qimages, x_curated_qimages, recovered_x_curated_bbox, is_positive_samples)):
            i_row = index // self.n_vertical_subplots
            i_col = index % self.n_horizontal_subplots
            subplot = self.viewer.get_subplot(i_row, i_col)

            canvas = subplot.get_canvas(0)
            canvas.set_background(z)
            with canvas.get_painter() as painter:
                painter.set_pen(self.bbox_pen)
                painter.draw_rect(_bbox_format_convert(z_bbox))
            canvas.update()

            canvas = subplot.get_canvas(1)
            canvas.set_background(x)
            with canvas.get_painter() as painter:
                painter.set_pen(self.bbox_pen)
                painter.draw_rect(_bbox_format_convert(x_bbox))
            canvas.update()

            canvas = subplot.get_canvas(2)
            canvas.set_background(z_curated)
            canvas.update()

            canvas = subplot.get_canvas(3)
            canvas.set_background(x_curated)
            if x_curated_bbox is not None:
                with canvas.get_painter() as painter:
                    painter.set_pen(self.bbox_pen)
                    painter.draw_rect(_bbox_format_convert(x_curated_bbox))
            canvas.update()

            label = subplot.get_informative_widget(0)
            label.setText('positive' if is_positive_sample else 'negative')
