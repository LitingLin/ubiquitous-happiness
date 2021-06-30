from data.operator.bbox.spatial.scale_and_translate import bbox_scale_and_translate
from data.operator.bbox.aligned.validity import is_bbox_validity
from data.operator.bbox.intersection import bbox_compute_intersection
import tensorflow as tf
from miscellanies.qt_numpy_interop import numpy_rgb888_to_qimage
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont, QPixmap
from PyQt5.QtCore import QRectF, Qt
import numpy as np
from data.operator.image_and_bbox.align_corner.torch_scale_and_translate import torch_scale_and_translate_align_corners
import torch
from data.operator.image.tf.dtype import torch_image_round_to_uint8


def _do_render(background_image: tf.Tensor, rendering_contexts, target_size, scale=(1., 1.), translation_source_center=(0, 0), translation_target_center=(0, 0), with_qpixmap=True):
    background_image = torch.tensor(background_image.numpy())
    canvas, _ = torch_scale_and_translate_align_corners(background_image, target_size, scale,
                                                        translation_source_center, translation_target_center)
    canvas = torch_image_round_to_uint8(canvas)
    canvas = canvas.squeeze(0)
    canvas = canvas.numpy()
    #canvas = tf_image_scale_and_translate_align_corners(background_image, target_size, scale,
    #                                                    translation_source_center, translation_target_center)
    #canvas = tf_image_round_to_uint8(canvas)
    #canvas = tf.squeeze(canvas, 0)
    canvas = numpy_rgb888_to_qimage(canvas)
    if with_qpixmap:
        canvas = QPixmap.fromImage(canvas)
    painter = QPainter(canvas)
    for rendering_context in rendering_contexts:
        rendering_command = rendering_context[0]
        if rendering_command == 'rectangle':
            bounding_box = rendering_context[1]
            pen = rendering_context[2]
            bbox = bbox_scale_and_translate(bounding_box, scale, translation_source_center, translation_target_center)
            if not is_bbox_validity(bbox_compute_intersection(bbox, (0, 0, target_size[0] - 1, target_size[1] - 1))):
                continue
            painter.setPen(pen)
            painter.drawRect(QRectF(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
        elif rendering_command == 'rectangle_with_label':
            bounding_box = rendering_context[1]
            label_text = rendering_context[2]
            bounding_box_pen = rendering_context[3]
            text_pen = rendering_context[4]
            font = rendering_context[5]

            bbox = bbox_scale_and_translate(bounding_box, scale, translation_source_center, translation_target_center)
            if not is_bbox_validity(bbox_compute_intersection(bbox, (0, 0, target_size[0] - 1, target_size[1] - 1))):
                continue

            painter.setPen(bounding_box_pen)
            painter.drawRect(QRectF(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
            painter.setFont(font)
            font_metrics = painter.fontMetrics()

            rendered_text_size = font_metrics.boundingRect(label_text)
            label_rect = QRectF(bbox[0], bbox[1] - rendered_text_size.height(), rendered_text_size.width(), rendered_text_size.height())
            painter.fillRect(label_rect, bounding_box_pen.brush())
            painter.setPen(text_pen)
            painter.drawText(label_rect, Qt.AlignVCenter, label_text)
        else:
            raise NotImplementedError

    painter.end()
    return canvas


class SimplePainter:
    def __init__(self, canvas):
        self.canvas = canvas
        self.rendering_contexts = []

    def draw_bounding_box(self, bounding_box, color=QColor(255, 0, 0, 127), line_style=Qt.SolidLine, line_width=1):
        self.rendering_contexts.append(('rectangle', bounding_box, QPen(QBrush(color), line_width, line_style)))

    def draw_bounding_box_with_label(self, bounding_box, label_text,
                                     bounding_box_color=QColor(255, 0, 0, 127), bounding_box_boundary_line_style=Qt.SolidLine, bounding_box_boundary_line_width=1,
                                     label_text_font_family=None, label_text_font_size=-1, label_text_color=QColor(0, 255, 255, 127)):
        font = QFont()
        if label_text_font_family is not None:
            font.setFamily(label_text_font_family)
        if label_text_font_size != -1:
            if isinstance(label_text_font_size, int):
                font.setPixelSize(label_text_font_size)
            else:
                font.setPixelSize(label_text_font_size)
        else:
            font.setPixelSize(48)
        self.rendering_contexts.append(('rectangle_with_label', bounding_box, label_text,
                                        QPen(QBrush(bounding_box_color), bounding_box_boundary_line_width, bounding_box_boundary_line_style),
                                        QPen(label_text_color), font))

    def render(self, target_size, scale=(1., 1.), translation_source_center=(0, 0), translation_target_center=(0, 0), with_qpixmap=False):
        return _do_render(self.canvas, self.rendering_contexts, target_size, scale, translation_source_center, translation_target_center, with_qpixmap)

    def get_canvas_size(self):
        return self.canvas.shape[2], self.canvas.shape[1]

    @staticmethod
    def create_from_tf_image(image: tf.Tensor):
        if tf.rank(image) == 2:
            image = tf.expand_dims(image, 0)
            image = tf.expand_dims(image, -1)
            image = tf.image.grayscale_to_rgb(image)
        elif image.ndim == 3:
            image = tf.expand_dims(image, 0)
            if image.shape[-1] == 1:
                image = tf.image.grayscale_to_rgb(image)
            elif image.shape[-1] != 3:
                raise NotImplementedError
        elif image.ndim == 4:
            if image.shape[-1] == 1:
                image = tf.image.grayscale_to_rgb(image)
            elif image.shape[-1] != 3:
                raise NotImplementedError
        return SimplePainter(image)

    @staticmethod
    def create_from_image(image: np.ndarray):
        return SimplePainter.create_from_tf_image(tf.constant(image))

    @staticmethod
    def create_empty(width, height):
        canvas = tf.zeros([1, height, width, 3])
        return SimplePainter(canvas)
