from Dataset.Type.bounding_box_format import BoundingBoxFormat
from Dataset.Base.Common.ops import set_bounding_box_
from Dataset.Base.Common.constructor import BaseDatasetConstructorGenerator, set_path_, BaseDatasetSequenceConstructorGenerator, BaseDatasetSequenceConstructor, BaseVideoDatasetConstructor


class MultipleObjectTrackingDatasetSequenceFrameObjectConstructor:
    def __init__(self, object_: dict):
        self.object_ = object_

    def set_bounding_box(self, bounding_box, bounding_box_format: BoundingBoxFormat = BoundingBoxFormat.XYWH,
                         validity=None):
        set_bounding_box_(self.object_, bounding_box, bounding_box_format, validity)

    def set_attribute(self, name: str, value):
        self.object_[name] = value

    def merge_attributes(self, attributes):
        for key, value in attributes.items():
            self.object_[key] = value


class MultipleObjectTrackingDatasetSequenceFrameObjectConstructorGenerator:
    def __init__(self, object_: dict):
        self.object_ = object_

    def __enter__(self):
        return MultipleObjectTrackingDatasetSequenceFrameObjectConstructor(self.object_)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MultipleObjectTrackingDatasetSequenceFrameConstructor:
    def __init__(self, frame: dict, root_path):
        self.frame = frame
        self.root_path = root_path

    def new_object(self, object_id):
        if 'objects' not in self.frame:
            self.frame['objects'] = []
        object_ = {'id': object_id}
        self.frame['objects'].append(object_)
        return MultipleObjectTrackingDatasetSequenceFrameObjectConstructorGenerator(object_)

    def set_path(self, path, image_size=None):
        set_path_(self.frame, path, self.root_path, image_size)

    def set_attribute(self, name: str, value):
        self.frame[name] = value

    def merge_attributes(self, attributes):
        for key, value in attributes.items():
            self.frame[key] = value


class MultipleObjectTrackingDatasetSequenceFrameConstructorGenerator:
    def __init__(self, frame: dict, root_path: str):
        self.frame = frame
        self.root_path = root_path

    def __enter__(self):
        return MultipleObjectTrackingDatasetSequenceFrameConstructor(self.frame, self.root_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MultipleObjectTrackingDatasetSequenceObjectConstructor:
    def __init__(self, object_: dict):
        self.object_ = object_

    def set_category_id(self, category_id: int):
        self.object_['category_id'] = category_id

    def set_attribute(self, name: str, value):
        self.object_[name] = value

    def merge_attributes(self, attributes):
        for key, value in attributes.items():
            self.object_[key] = value


class MultipleObjectTrackingDatasetSequenceObjectConstructorGenerator:
    def __init__(self, object_: dict):
        self.object_ = object_

    def __enter__(self):
        return MultipleObjectTrackingDatasetSequenceObjectConstructor(self.object_)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MultipleObjectTrackingDatasetSequenceConstructor(BaseDatasetSequenceConstructor):
    def __init__(self, sequence: dict, root_path: str, pbar):
        super(MultipleObjectTrackingDatasetSequenceConstructor, self).__init__(sequence, root_path, pbar)

    def new_object(self, object_id):
        if 'objects' not in self.sequence:
            self.sequence['objects'] = []
        object_ = {'id': object_id}
        self.sequence['objects'].append(object_)
        return MultipleObjectTrackingDatasetSequenceObjectConstructorGenerator(object_)

    def new_frame(self):
        if 'frames' not in self.sequence:
            self.sequence['frames'] = []
        frame = {}
        self.sequence['frames'].append(frame)
        return MultipleObjectTrackingDatasetSequenceFrameConstructorGenerator(frame, self.root_path)

    def open_frame(self, index: int):
        frame = self.sequence['frames'][index]
        return MultipleObjectTrackingDatasetSequenceFrameConstructorGenerator(frame, self.root_path)


class MultipleObjectTrackingDatasetSequenceConstructorGenerator(BaseDatasetSequenceConstructorGenerator):
    def __init__(self, sequence, root_path, pbar):
        super(MultipleObjectTrackingDatasetSequenceConstructorGenerator, self).__init__(sequence, pbar)
        self.root_path = root_path

    def __enter__(self):
        return MultipleObjectTrackingDatasetSequenceConstructor(self.sequence, self.root_path, self.pbar)


class MultipleObjectTrackingDatasetConstructor(BaseVideoDatasetConstructor):
    def __init__(self, dataset: dict, root_path: str, version: int, pbar):
        super(MultipleObjectTrackingDatasetConstructor, self).__init__(dataset, root_path, version, pbar)
        if 'sequences' not in dataset:
            dataset['sequences'] = []

    def new_sequence(self):
        sequence = {}
        self.dataset['sequences'].append(sequence)
        return MultipleObjectTrackingDatasetSequenceConstructorGenerator(sequence, self.root_path, self.pbar)


class MultipleObjectTrackingDatasetConstructorGenerator(BaseDatasetConstructorGenerator):
    def __init__(self, dataset: dict, root_path: str, version: int):
        super(MultipleObjectTrackingDatasetConstructorGenerator, self).__init__(dataset, root_path, version, MultipleObjectTrackingDatasetConstructor)
