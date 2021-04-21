import cv2
from tqdm import tqdm
import os
import numpy as np


_colors = [(255, 0, 0),
 (0, 255, 0),
 (0, 0, 255),
 (0, 0, 0),
 (255, 0, 255),
 (0, 255, 255),
 (128, 128, 128),
 (136, 0, 21),
 (255, 127, 39),
 (0, 162, 232),
 (0, 128, 0),
 (255, 128, 51),
 (26, 102, 0),
 (153, 76, 230),
 (102, 178, 26),
 (51, 26, 178),
 (178, 153, 51)]


class Composer:
    def __init__(self, size, tracker_names, colors, opacity, thickness=2):
        assert len(tracker_names) == len(colors)
        self.background = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self.colors = colors
        self.opacity = opacity
        self.thickness = thickness
        if size[1] > 160:
            font_height = int(round(size[1] / 40))
        else:
            font_height = 8

        font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, font_height, 1)
        offset = int(round(min(size[0] / 10, size[1] / 10)))
        font_height = int(font_height * 1.25)
        text_width = 0
        font_sizes = []
        for tracker_name in tracker_names:
            size, baseline = cv2.getTextSize(tracker_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            font_sizes.append((size, baseline))
            text_width = max(size[0], text_width)
        sample_color_width = 6 * font_height
        bounding_box_pad = int(0.025 * (text_width + sample_color_width))
        text_width += bounding_box_pad
        bounding_box_width = text_width + sample_color_width + 2 * bounding_box_pad
        bounding_box_height = len(tracker_names) * font_height + 2 * bounding_box_pad
        bounding_box_x = offset - bounding_box_pad
        bounding_box_y = offset - bounding_box_pad

        cv2.rectangle(self.background, (bounding_box_x, bounding_box_y), (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height), (255, 255, 255), thickness=-1)
        offset_y = offset
        for index, ((font_size, font_baseline), tracker_name) in enumerate(zip(font_sizes, tracker_names)):
            cv2.putText(self.background, tracker_name, (offset, offset_y + font_size[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (1, 1, 1))
            cv2.rectangle(self.background, (offset + text_width, offset_y + int(font_height * 0.2)), (offset + text_width + sample_color_width, offset_y + font_height - int(font_height * 0.2)), self.colors[index], -1)
            offset_y += font_height

    def _resize(self, image):
        h, w, c = image.shape
        assert c == self.background.shape[2]
        if h != self.background.shape[0] or w != self.background.shape[1]:
            image = cv2.resize(image, (w, h))
        else:
            image = image
        return image

    def render(self, image, bounding_boxes):
        image = self._resize(image)
        canvas = self.background.copy()
        assert len(bounding_boxes) == len(self.colors)
        for index, bounding_box in enumerate(bounding_boxes):
            cv2.rectangle(canvas, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), self.colors[index], self.thickness)
        mask = canvas > 0
        # mask = np.tile(mask.sum(axis=2)[:, :, np.newaxis], (1, 1, 3)).astype(np.bool)
        mask = mask.sum(axis=2).astype(np.bool)
        image[mask] = np.around(image[mask] * (1 - self.opacity) + canvas[mask] * self.opacity).clip(0, 255).astype(np.uint8)
        return image


def get_standard_bounding_box_rasterizer():
    from data.operator.bbox.transform.compile import compile_bbox_transform

    from data.types.bounding_box_format import BoundingBoxFormat
    from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
    from data.types.pixel_coordinate_system import PixelCoordinateSystem
    from data.types.pixel_definition import PixelDefinition

    return compile_bbox_transform(BoundingBoxFormat.XYXY, BoundingBoxFormat.XYXY, PixelCoordinateSystem.Aligned, PixelCoordinateSystem.Aligned, BoundingBoxCoordinateSystem.Spatial, BoundingBoxCoordinateSystem.Rasterized, PixelDefinition.Point)


def generate_sequence_video(tracker_name, sequence, predicted_bboxes, bounding_box_rasterizer, output_file_path):
    if sequence.has_fps():
        fps = sequence.get_fps()
    else:
        fps = 30
    frame = sequence[0]
    size = frame.get_image_size()
    tmp_file_path = os.path.join(os.path.dirname(output_file_path), '~' + os.path.basename(output_file_path))
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)
    writer = cv2.VideoWriter(tmp_file_path, cv2.VideoWriter_fourcc('X', '2', '6', '4'), fps, size, True)
    assert predicted_bboxes.shape[0] == len(sequence)

    composer = Composer(size, ('GT', tracker_name), _colors[:2], 0.8)

    for frame, predicted_bbox in tqdm(zip(sequence, predicted_bboxes), desc=f'Rendering {sequence.get_name()}', total=len(sequence)):
        image = cv2.imread(frame.get_image_path(), cv2.IMREAD_COLOR)
        bounding_boxes = []
        if frame.has_bounding_box():
            if frame.has_bounding_box_validity_flag() and not frame.get_bounding_box_validity_flag():
                bounding_boxes.append((-1, -1, -1, -1))
            else:
                bounding_box = frame.get_bounding_box().tolist()
                bounding_boxes.append(bounding_box_rasterizer(bounding_box))
        else:
            bounding_boxes.append((-1, -1, -1, -1))
        bounding_boxes.append(bounding_box_rasterizer(predicted_bbox.tolist()))
        image = composer.render(image, bounding_boxes)
        writer.write(image)
    writer.release()
    os.rename(tmp_file_path, output_file_path)


def visualize_tracking_results(tracker_names, sequence, predicted_bboxes, bounding_box_rasterizer, output_file_path):
    if sequence.has_fps():
        fps = sequence.get_fps()
    else:
        fps = 30
    frame = sequence[0]
    size = frame.get_image_size()
    tmp_file_path = os.path.join(os.path.dirname(output_file_path), '~' + os.path.basename(output_file_path))
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)
    writer = cv2.VideoWriter(tmp_file_path, cv2.VideoWriter_fourcc('X', '2', '6', '4'), fps, size, True)
    # assert predicted_bboxes.shape[0] == len(sequence)

    composer = Composer(size, ['GT', *tracker_names], _colors[:len(tracker_names) + 1], 0.8)

    for index_of_frame, frame in tqdm(enumerate(sequence), desc=f'Rendering {sequence.get_name()}', total=len(sequence)):
        image = cv2.imread(frame.get_image_path(), cv2.IMREAD_COLOR)
        bounding_boxes = []
        if frame.has_bounding_box():
            if frame.has_bounding_box_validity_flag() and not frame.get_bounding_box_validity_flag():
                bounding_boxes.append((-1, -1, -1, -1))
            else:
                bounding_box = frame.get_bounding_box().tolist()
                bounding_boxes.append(bounding_box_rasterizer(bounding_box))
        else:
            bounding_boxes.append((-1, -1, -1, -1))
        for tracker_predicted_bboxes in predicted_bboxes:
            bounding_boxes.append(bounding_box_rasterizer(tracker_predicted_bboxes[index_of_frame].tolist()))
        image = composer.render(image, bounding_boxes)
        writer.write(image)
    writer.release()
    os.rename(tmp_file_path, output_file_path)
