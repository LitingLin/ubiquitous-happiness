import random
import numpy as np
from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDataset_MemoryMapped
from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDataset_MemoryMapped


def _sample_visible_ids(visible, num_ids=1, min_id=None, max_id=None):
    """ Samples num_ids frames between min_id and max_id for which target is visible

    args:
        visible - 1d Tensor indicating whether target is visible for each frame
        num_ids - number of frames to be samples
        min_id - Minimum allowed frame number
        max_id - Maximum allowed frame number

    returns:
        list - List of sampled frame numbers. None if not sufficient visible frames could be found.
    """
    if num_ids == 0:
        return []
    if min_id is None or min_id < 0:
        min_id = 0
    if max_id is None or max_id > len(visible):
        max_id = len(visible)

    valid_ids = [i for i in range(min_id, max_id) if visible[i]]

    # No visible ids
    if len(valid_ids) == 0:
        return None

    return random.choices(valid_ids, k=num_ids)


def _adjust_range_of_frame(range_of_frame, fps):
    range_of_frame = int(round(fps / 30 * range_of_frame))
    return range_of_frame


class DetectionDatasetSiamFCSampler:
    def __init__(self, dataset: DetectionDataset_MemoryMapped):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_positive_pair(self, index):
        image = self.dataset[index]
        visible = image.get_all_bounding_box_validity_flag()
        if visible is None:
            visible = np.ones([len(image)], dtype=np.uint8)
        z_index = _sample_visible_ids(visible)
        if z_index is None:
            return None

        z_index = z_index[0]
        object_ = image[z_index]
        return image.get_image_path(), object_.get_bounding_box()

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, len(self.dataset))
        return self.get_positive_pair(index)


class SOTDatasetSiamFCSampler:
    def __init__(self, dataset: SingleObjectTrackingDataset_MemoryMapped, frame_range: int):
        self.dataset = dataset
        self.frame_range = frame_range

    def __len__(self):
        return len(self.dataset)

    def get_positive_pair(self, index: int):
        sequence = self.dataset[index]
        frame_range = self.frame_range

        if sequence.has_fps():
            frame_range = _adjust_range_of_frame(frame_range, sequence.get_fps())

        visible = sequence.get_all_bounding_box_validity_flag()
        if visible is None:
            visible = np.ones([len(sequence)], dtype=np.uint8)
        z_index = _sample_visible_ids(visible)
        if z_index is None:
            return None

        z_index = z_index[0]
        z_frame = sequence[z_index]

        x_index = _sample_visible_ids(visible, 1, z_index - frame_range, z_index + frame_range)
        if x_index is None or x_index[0] == z_index:
            return z_frame.get_image_path(), z_frame.get_bounding_box()
        else:
            x_frame = sequence[x_index[0]]
            return z_frame.get_image_path(), z_frame.get_bounding_box(), x_frame.get_image_path(), x_frame.get_bounding_box()

    def get_random_target(self, index: int = -1):
        if index == -1:
            index = np.random.randint(0, len(self.dataset))
        sequence = self.dataset[index]

        visible = sequence.get_all_bounding_box_validity_flag()
        if visible is None:
            visible = np.ones([len(sequence)], dtype=np.uint8)
        z_index = _sample_visible_ids(visible)

        if z_index is None:
            return None

        z_index = z_index[0]
        z_frame = sequence[z_index]
        return z_frame.get_image_path(), z_frame.get_bounding_box()


class MOTDatasetSiamFCSampler:
    def __init__(self, dataset: MultipleObjectTrackingDataset_MemoryMapped, frame_range: int):
        self.dataset = dataset
        self.frame_range = frame_range

    def __len__(self):
        return len(self.dataset)

    def get_positive_pair(self, index):
        sequence = self.dataset[index]
        frame_range = self.frame_range
        if sequence.has_fps():
            frame_range = _adjust_range_of_frame(frame_range, sequence.get_fps())
        index_of_track = random.randint(0, sequence.get_number_of_objects() - 1)
        track = sequence.get_object(index_of_track)
        length_of_sequence = len(sequence)
        visible = np.zeros(length_of_sequence, dtype=np.uint8)
        if track.get_all_bounding_box_validity_flag() is not None:
            ind = track.get_all_frame_index()[track.get_all_bounding_box_validity_flag()]
            visible[ind] = 1
        else:
            visible[track.get_all_frame_index()] = 1
        object_id = track.get_id()
        z_index = _sample_visible_ids(visible)
        if z_index is None:
            return None
        z_index = z_index[0]

        z_frame = sequence[z_index]
        z_object = z_frame.get_object_by_id(object_id)

        x_index = _sample_visible_ids(visible, 1, z_index - frame_range, z_index + frame_range)
        if x_index is None or x_index[0] == z_index:
            return z_frame.get_image_path(), z_object.get_bounding_box()
        else:
            x_frame = sequence[x_index[0]]
            x_object = x_frame.get_object_by_id(object_id)
            return z_frame.get_image_path(), z_object.get_bounding_box(), x_frame.get_image_path(), x_object.get_bounding_box()

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, len(self.dataset))
        sequence = self.dataset[index]
        index_of_track = random.randint(0, sequence.get_number_of_objects() - 1)
        track = sequence.get_object(index_of_track)
        length_of_sequence = len(sequence)
        visible = np.zeros(length_of_sequence, dtype=np.uint8)
        if track.get_all_bounding_box_validity_flag() is not None:
            ind = track.get_all_frame_index()[track.get_all_bounding_box_validity_flag()]
            visible[ind] = 1
        else:
            visible[track.get_all_frame_index()] = 1
        object_id = track.get_id()
        z_index = _sample_visible_ids(visible)
        if z_index is None:
            return None
        z_index = z_index[0]

        z_frame = sequence[z_index]
        z_object = z_frame.get_object_by_id(object_id)
        return z_frame.get_image_path(), z_object.get_bounding_box()
