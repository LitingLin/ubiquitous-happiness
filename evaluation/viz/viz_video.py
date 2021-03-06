from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped
import numpy as np
import cv2


def _round_to_int(bounding_box):
    if isinstance(bounding_box, np.ndarray):
        bounding_box = bounding_box.tolist()
    bounding_box = [int(round(v)) for v in bounding_box]
    return bounding_box


def generate_video_on_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, tracked_file: str, output_file_path: str):
    if sequence.has_fps():
        fps = sequence.get_fps()
    else:
        fps = 30
    frame = sequence[0]
    size = frame.get_image_size()
    writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('X', '2', '6', '4'), fps, size, True)
    predicted_bboxes = np.loadtxt(tracked_file, delimiter=',')
    for frame, predicted_bbox in zip(sequence, predicted_bboxes):
        assert frame.get_image_size() == size
        image = cv2.imread(frame.get_image_path(), cv2.IMREAD_COLOR)
        if frame.has_bounding_box():
            if frame.has_bounding_box_validity_flag() and not frame.get_bounding_box_validity_flag():
                pass
            else:
                bounding_box = frame.get_bounding_box()
                bounding_box = _round_to_int(bounding_box)
                cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 0), 2)
        predicted_bbox = _round_to_int(predicted_bbox)
        cv2.rectangle(image, (predicted_bbox[0], predicted_bbox[1]), (predicted_bbox[0] + predicted_bbox[2], predicted_bbox[1] + predicted_bbox[3]), (0, 0, 255), 2)
        writer.write(image)
    writer.release()


if __name__ == '__main__':
    from Dataset.SOT.Seed.OTB100 import OTB100_Seed
    from Dataset.SOT.Seed.GOT10k import GOT10k_Seed
    from Dataset.SOT.Seed.LaSOT import LaSOT_Seed
    from Dataset.Type.data_split import DataSplit
    from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
    from Dataset.Filter.DataCleaning.Integrity import DataCleaning_Integrity
    from Dataset.Filter.DataCleaning.BoundingBox import DataCleaning_BoundingBox

    standard_filters = [DataCleaning_BoundingBox(update_validity=True, remove_non_validity_objects=True, remove_empty_annotation_objects=True), DataCleaning_Integrity()]

    dataset = SingleObjectTrackingDatasetFactory([OTB100_Seed()]).construct(filters=standard_filters)[0]
    for sequence in dataset:
        if sequence.get_name() == 'David2':
            generate_video_on_sequence(sequence, "C:\\git\\David2.txt", 'C:\\git\\David2.mp4')

            break

    #
    # dataset = SingleObjectTrackingDatasetFactory([GOT10k_Seed(data_split=DataSplit.Validation)]).construct(filters=standard_filters)[0]
    # for sequence in dataset:
    #     if sequence.get_name() == 'GOT-10k_Val_000005':
    #         generate_video_on_sequence(sequence, "C:\\git\\GOT-10k_Val_000005_001.txt", 'C:\\git\\GOT-10k_Val_000005_001.mp4')
    #
    #         break

    #
    # dataset = SingleObjectTrackingDatasetFactory([LaSOT_Seed(data_split=DataSplit.Validation)]).construct(filters=standard_filters)[0]
    # for sequence in dataset:
    #     if sequence.get_name() == 'leopard-1':
    #         generate_video_on_sequence(sequence, "C:\\git\\leopard-1.txt", 'C:\\git\\leopard-1.mp4')
    #
    #         break
