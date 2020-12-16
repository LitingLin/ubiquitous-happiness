import random
import numpy as np


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

# may return None
def _sampling_on_dataset(dataset, sub_sequence_index, range_of_frame: int):
    dataset_class_name: str = dataset.__class__.__name__
    if dataset_class_name.startswith('Single'):
        sequence = dataset[sub_sequence_index]
        if dataset.hasAttibuteFPS():
            fps = sequence.getFPS()
            range_of_frame = _adjust_range_of_frame(range_of_frame, fps)
        bounding_boxes = sequence.getAllBoundingBox()
        visible = (bounding_boxes[:, 2] > 0) & (bounding_boxes[:, 3] > 0)
        z_index = _sample_visible_ids(visible)
        if z_index is None:
            return None

        z_index = z_index[0]
        z_frame = sequence[z_index]

        x_index = _sample_visible_ids(visible, 1, z_index - range_of_frame, z_index + range_of_frame)
        if x_index is None or x_index[0] == z_index:
            return z_frame.getImagePath(), z_frame.getBoundingBox()
        else:
            x_frame = sequence[x_index[0]]
            return (z_frame.getImagePath(), z_frame.getBoundingBox()), (x_frame.getImagePath(), x_frame.getBoundingBox())
    elif dataset_class_name.startswith('Multiple'):
        sequence = dataset[sub_sequence_index]
        if dataset.hasAttibuteFPS():
            fps = sequence.getFPS()
            range_of_frame = _adjust_range_of_frame(range_of_frame, fps)

        index_of_track = random.randint(0, sequence.getNumberOfTracks() - 1)
        track = sequence.getTrackView(index_of_track)
        length_of_sequence = len(sequence)
        visible = np.zeros(length_of_sequence, dtype=np.uint8)
        visible[track.getFrameIndices()] = 1
        object_id = track.getObjectId()

        z_index = _sample_visible_ids(visible)
        if z_index is None:
            return None
        z_index = z_index[0]

        z_object = sequence[z_index].getObjectById(object_id)

        x_index = _sample_visible_ids(visible, 1, z_index - range_of_frame, z_index + range_of_frame)
        if x_index is None or x_index[0] == z_index:
            return z_object.getImagePath(), z_object.getBoundingBox()
        else:
            x_object = sequence[x_index[0]].getObjectById(object_id)
            return (z_object.getImagePath(), z_object.getBoundingBox()),\
                   (x_object.getImagePath(), x_object.getBoundingBox())
    elif dataset_class_name.startswith('Detection'):
        image = dataset[sub_sequence_index]
        number_of_objects = len(image)
        visible = np.zeros(number_of_objects, dtype=np.uint8)
        visible[image.getAllAttributeIsPresent()] = 1
        index_of_object = _sample_visible_ids(visible)
        if index_of_object is None:
            return None
        object_ = image[index_of_object[0]]
        return object_.getImagePath(), object_.getBoundingBox()
    else:
        raise Exception('Unknown dataset')


class DetectionDatasetSiamFCSampler:
    def __init__(self, dataset: 'DetectionDataset_MemoryMapped'):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_positive_pair(self, index):
        image = self.dataset[index]
        bounding_boxes = image.getAllBoundingBox()
        visible = (bounding_boxes[:, 2] > 0) & (bounding_boxes[:, 3] > 0)
        if image.hasAttributeIsPresent():
            visible[~(image.getAllAttributeIsPresent())] = 0

        z_index = _sample_visible_ids(visible)
        if z_index is None:
            return None

        z_index = z_index[0]
        object_ = image[z_index]
        return image.getImagePath(), object_.getBoundingBox()

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, len(self.dataset))
        return self.get_positive_pair(index)


class SOTDatasetSiamFCSampler:
    def __init__(self, dataset: 'SingleObjectTrackingDataset_MemoryMapped', frame_range: int):
        self.dataset = dataset
        self.frame_range = frame_range

    def __len__(self):
        return len(self.dataset)

    def get_positive_pair(self, index: int):
        sequence = self.dataset[index]
        frame_range = self.frame_range
        bounding_boxes = sequence.getAllBoundingBox()
        visible = (bounding_boxes[:, 2] > 0) & (bounding_boxes[:, 3] > 0)
        z_index = _sample_visible_ids(visible)
        if z_index is None:
            return None

        z_index = z_index[0]
        z_frame = sequence[z_index]

        x_index = _sample_visible_ids(visible, 1, z_index - frame_range, z_index + frame_range)
        if x_index is None or x_index[0] == z_index:
            return z_frame.getImagePath(), z_frame.getBoundingBox()
        else:
            x_frame = sequence[x_index[0]]
            return z_frame.getImagePath(), z_frame.getBoundingBox(), x_frame.getImagePath(), x_frame.getBoundingBox()

    def get_random_target(self, index: int = -1):
        if index == -1:
            index = np.random.randint(0, len(self.dataset))
        sequence = self.dataset[index]

        bounding_boxes = sequence.getAllBoundingBox()
        visible = (bounding_boxes[:, 2] > 0) & (bounding_boxes[:, 3] > 0)
        z_index = _sample_visible_ids(visible)

        if z_index is None:
            return None

        z_index = z_index[0]
        z_frame = sequence[z_index]
        return z_frame.getImagePath(), z_frame.getBoundingBox()


class MOTDatasetSiamFCSampler:
    def __init__(self, dataset: 'MultipleObjectTrackingDataset', frame_range: int):
        self.dataset = dataset
        self.frame_range = frame_range

    def __len__(self):
        return len(self.dataset)

    def get_positive_pair(self, index):
        sequence = self.dataset[index]
        frame_range = self.frame_range

        index_of_track = random.randint(0, sequence.getNumberOfTracks() - 1)
        track = sequence.getTrackView(index_of_track)
        length_of_sequence = len(sequence)
        visible = np.zeros(length_of_sequence, dtype=np.uint8)
        for track_index, object_ in zip(track.getFrameIndices(), track):
            if object_.hasAttributeIsPresent():
                if object_.getAttributeIsPresent():
                    visible[track_index] = 1
            else:
                visible[track_index] = 1
        object_id = track.getObjectId()
        z_index = _sample_visible_ids(visible)
        if z_index is None:
            return None
        z_index = z_index[0]

        z_object = sequence[z_index].getObjectById(object_id)

        x_index = _sample_visible_ids(visible, 1, z_index - frame_range, z_index + frame_range)
        if x_index is None or x_index[0] == z_index:
            return z_object.getImagePath(), z_object.getBoundingBox()
        else:
            x_object = sequence[x_index[0]].getObjectById(object_id)
            return z_object.getImagePath(), z_object.getBoundingBox(), x_object.getImagePath(), x_object.getBoundingBox()

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, len(self.dataset))
        sequence = self.dataset[index]
        index_of_track = random.randint(0, sequence.getNumberOfTracks() - 1)
        track = sequence.getTrackView(index_of_track)
        length_of_sequence = len(sequence)
        visible = np.zeros(length_of_sequence, dtype=np.uint8)
        for track_index, object_ in zip(track.getFrameIndices(), track):
            if object_.hasAttributeIsPresent():
                if object_.getAttributeIsPresent():
                    visible[track_index] = 1
            else:
                visible[track_index] = 1
        object_id = track.getObjectId()
        z_index = _sample_visible_ids(visible)
        if z_index is None:
            return None
        z_index = z_index[0]

        z_object = sequence[z_index].getObjectById(object_id)
        return z_object.getImagePath(), z_object.getBoundingBox()
