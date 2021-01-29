from Dataset.Type.bounding_box_format import BoundingBoxFormat
from Dataset.Base.Common.ops import set_bounding_box_
from Dataset.Base.Common.constructor import BaseDatasetConstructorGenerator, set_path_, BaseDatasetSequenceConstructorGenerator, BaseDatasetSequenceConstructor, BaseVideoDatasetConstructor


__all__ = ['SingleObjectTrackingDatasetConstructor', 'SingleObjectTrackingDatasetConstructorGenerator']

'''
{
    version: '',
    name: '',
    split: '',
    category_id_name_map: '',
    sequences: [
            {
                name: '',
                path: '',
                objects: [
                    {
                        'category_id': 11,
                        'id': 0
                    }
                ],
                frames: [
                    {
                        path: '',
                        objects: [
                            {
                                'id': 0
                                bbox: [],
                                bbox_type: '',
                            }
                        ]

                    }
                ]
            },
            {}
        ]
}
'''


class SingleObjectTrackingDatasetFrameConstructor:
    def __init__(self, frame, root_path):
        self.frame = frame
        self.root_path = root_path
        self.object = None

    def _try_allocate_object(self):
        if self.object is None:
            self.object = {'id': 0}
            self.frame['objects'] = [self.object]

    def set_path(self, path: str, image_size=None):
        set_path_(self.frame, path, self.root_path, image_size)

    def set_bounding_box(self, bounding_box, bounding_box_format: BoundingBoxFormat=BoundingBoxFormat.XYWH, validity=None):
        self._try_allocate_object()
        set_bounding_box_(self.object, bounding_box, bounding_box_format, validity)

    def set_object_attribute(self, name, value):
        self._try_allocate_object()
        self.object[name] = value

    def set_frame_attribute(self, name, value):
        self.frame[name] = value


class SingleObjectTrackingDatasetFrameConstructorGenerator:
    def __init__(self, frame, root_path):
        self.frame = frame
        self.root_path = root_path

    def __enter__(self):
        return SingleObjectTrackingDatasetFrameConstructor(self.frame, self.root_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class SingleObjectTrackingDatasetSequenceConstructor(BaseDatasetSequenceConstructor):
    def __init__(self, sequence: dict, root_path: str, pbar):
        super(SingleObjectTrackingDatasetSequenceConstructor, self).__init__(sequence, root_path, pbar)
        self.object_ = None
        if 'objects' in self.sequence:
            self.object_ = self.sequence['objects'][0]

    def new_frame(self):
        frame = {}
        if 'frames' not in self.sequence:
            self.sequence['frames'] = []
        self.sequence['frames'].append(frame)
        return SingleObjectTrackingDatasetFrameConstructorGenerator(frame, self.root_path)

    def open_frame(self, index: int):
        return SingleObjectTrackingDatasetFrameConstructorGenerator(self.sequence['frames'][index], self.root_path)

    def set_object_attribute(self, name: str, value):
        if self.object_ is None:
            self.object_ = {name: value, 'id': 0}
            self.sequence['objects'] = [self.object_]
        else:
            self.object_[name] = value


class SingleObjectTrackingDatasetSequenceConstructorGenerator(BaseDatasetSequenceConstructorGenerator):
    def __init__(self, sequence, root_path, category_id, pbar):
        super(SingleObjectTrackingDatasetSequenceConstructorGenerator, self).__init__(sequence, pbar)

        self.root_path = root_path
        if category_id is not None:
            self.sequence['objects'] = [{'id': 0, 'category_id': category_id}]

    def __enter__(self):
        return SingleObjectTrackingDatasetSequenceConstructor(self.sequence, self.root_path, self.pbar)


class SingleObjectTrackingDatasetConstructor(BaseVideoDatasetConstructor):
    def __init__(self, dataset: dict, root_path: str, version: int, pbar):
        super(SingleObjectTrackingDatasetConstructor, self).__init__(dataset, root_path, version, pbar)
        if 'sequences' not in dataset:
            dataset['sequences'] = []

    def new_sequence(self, category_id=None):
        if category_id is not None:
            assert category_id in self.dataset['category_id_name_map']
        sequence = {}
        self.dataset['sequences'].append(sequence)
        return SingleObjectTrackingDatasetSequenceConstructorGenerator(sequence, self.root_path, category_id, self.pbar)


class SingleObjectTrackingDatasetConstructorGenerator(BaseDatasetConstructorGenerator):
    def __init__(self, dataset: dict, root_path: str, version: int):
        super(SingleObjectTrackingDatasetConstructorGenerator, self).__init__(dataset, root_path, version, SingleObjectTrackingDatasetConstructor)
