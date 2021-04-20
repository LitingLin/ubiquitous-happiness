import cv2
from tqdm import tqdm
import os


def get_standard_bounding_box_rasterizer():
    from data.operator.bbox.transform.compile import compile_bbox_transform

    from data.types.bounding_box_format import BoundingBoxFormat
    from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
    from data.types.pixel_coordinate_system import PixelCoordinateSystem
    from data.types.pixel_definition import PixelDefinition

    return compile_bbox_transform(BoundingBoxFormat.XYXY, BoundingBoxFormat.XYXY, PixelCoordinateSystem.Aligned, PixelCoordinateSystem.Aligned, BoundingBoxCoordinateSystem.Spatial, BoundingBoxCoordinateSystem.Rasterized, PixelDefinition.Point)


def generate_sequence_video(sequence, predicted_bboxes, bounding_box_rasterizer, output_file_path):
    if sequence.has_fps():
        fps = sequence.get_fps()
    else:
        fps = 30
    frame = sequence[0]
    size = frame.get_image_size()
    tmp_file_path = output_file_path + '.tmp'
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)
    writer = cv2.VideoWriter(tmp_file_path, cv2.VideoWriter_fourcc('X', '2', '6', '4'), fps, size, True)
    assert predicted_bboxes.shape[0] == len(sequence)
    for frame, predicted_bbox in tqdm(zip(sequence, predicted_bboxes), desc=f'Rendering {sequence.get_name()}', total=len(sequence)):
        image = cv2.imread(frame.get_image_path(), cv2.IMREAD_COLOR)
        if frame.get_image_size() != size:
            image = cv2.resize(image, size)
        if frame.has_bounding_box():
            if frame.has_bounding_box_validity_flag() and not frame.get_bounding_box_validity_flag():
                pass
            else:
                bounding_box = frame.get_bounding_box().tolist()
                bounding_box = bounding_box_rasterizer(bounding_box)
                cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 255, 0), 2)
        predicted_bbox = bounding_box_rasterizer(predicted_bbox.tolist())
        cv2.rectangle(image, (predicted_bbox[0], predicted_bbox[1]), (predicted_bbox[2], predicted_bbox[3]), (0, 0, 255), 2)
        writer.write(image)
    writer.release()
    os.rename(tmp_file_path, output_file_path)
