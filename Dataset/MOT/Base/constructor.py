from .dataset import MultipleObjectTrackingDataset
from .sequence import MultipleObjectTrackingDatasetSequence
from .frame import MultipleObjectTrackingDatasetFrame
from .object import MultipleObjectTrackingDataset_SequenceObjectAttribute, MultipleObjectTrackingDataset_FrameObjectAttribute
from typing import List, Dict, Any, Union
from ._impl import _set_image_path, _get_or_allocate_category_id
from Dataset.CacheService.constructor import DatasetConstructor_CacheService_Base


class MultipleObjectTrackingDatasetConstructor(DatasetConstructor_CacheService_Base):
    dataset: MultipleObjectTrackingDataset
    sequence: MultipleObjectTrackingDatasetSequence

    def __init__(self, dataset: MultipleObjectTrackingDataset):
        super(MultipleObjectTrackingDatasetConstructor, self).__init__(dataset)
        self.sequence = None
        self.has_fps_attribute = None

    def setDatasetAttribute(self, name: str, value: Any):
        self.dataset.attributes[name] = value

    def beginInitializingSequence(self):
        if self.sequence is not None:
            raise Exception("Already being initializing state")

        self.sequence = MultipleObjectTrackingDatasetSequence()

    def setSequenceName(self, name: str):
        self.sequence.name = name

    def updateSequenceAttributes(self, attributes: Dict):
        self.sequence.attributes.update(attributes)

    def setSequenceAttribute(self, name: str, value: Any):
        self.sequence.attributes[name] = value

    def setSequenceFPS(self, fps: Union[int, float]):
        self.sequence.fps = fps

    def addFrame(self, image_path: str, additional_attributes: Dict = None):
        assert isinstance(image_path, str)
        if additional_attributes is not None:
            assert isinstance(additional_attributes, dict)
        frame = MultipleObjectTrackingDatasetFrame()
        frame.size, frame.image_path = _set_image_path(self.root_path, image_path)
        frame.objects = {}
        frame.attributes = additional_attributes
        self.sequence.frames.append(frame)
        return len(self.sequence.frames) - 1

    def addObject(self, object_id: int, category_name: str, additional_attributes: Dict = None):
        assert isinstance(object_id, int)
        assert isinstance(category_name, str)
        if additional_attributes is not None:
            assert isinstance(additional_attributes, dict)
        if object_id in self.sequence.object_id_attributes_mapper:
            raise Exception("Object id {} already exists".format(object_id))

        category_id = _get_or_allocate_category_id(category_name, self.dataset.category_names, self.dataset.category_name_id_mapper)

        object_information = MultipleObjectTrackingDataset_SequenceObjectAttribute()
        object_information.category_id = category_id
        object_information.frame_indices = []
        object_information.attributes = additional_attributes
        self.sequence.object_id_attributes_mapper[object_id] = object_information
        self.sequence.object_ids.append(object_id)

    def setObjectCategoryName(self, object_id: int, category_name: str):
        assert isinstance(object_id, int)
        assert isinstance(category_name, str)
        if category_name not in self.dataset.category_name_id_mapper:
            category_id = len(self.dataset.category_names)
            self.dataset.category_names.append(category_name)
            self.dataset.category_name_id_mapper[category_name] = category_id
        else:
            category_id = self.dataset.category_name_id_mapper[category_name]
        self.sequence.object_id_attributes_mapper[object_id].category_id = category_id

    def addRecord(self, index_of_frame: int, object_id: int, bounding_box: List, is_present=None, additional_attributes: Dict = None):
        if is_present is not None:
            assert isinstance(is_present, bool)
        if additional_attributes is not None:
            assert isinstance(additional_attributes, dict)
        if object_id not in self.sequence.object_id_attributes_mapper:
            raise Exception("Object id {} not exists".format(object_id))

        current_object_information = self.sequence.object_id_attributes_mapper[object_id]
        if index_of_frame in current_object_information.frame_indices:
            raise Exception("Object id {} already exists in frame {}".format(object_id, index_of_frame))

        if 'has_present_attr' in self.sequence.attributes:
            assert is_present is not None
        if is_present is not None:
            if 'has_present_attr' not in self.sequence.attributes:
                self.sequence.attributes['has_present_attr'] = True
            else:
                assert self.sequence.attributes['has_present_attr']
        assert ~(bounding_box is None and is_present is None)

        frame_object_information = MultipleObjectTrackingDataset_FrameObjectAttribute()
        frame_object_information.bounding_box = bounding_box
        frame_object_information.attributes = additional_attributes
        frame_object_information.is_present = is_present

        frame = self.sequence.frames[index_of_frame]
        if object_id in frame.objects:
            raise Exception("Object id {} already exists in frame {}".format(object_id, index_of_frame))
        frame.objects[object_id] = frame_object_information
        current_object_information.frame_indices.append(index_of_frame)

    def endInitializingSequence(self):
        if self.sequence.fps is None:
            if self.has_fps_attribute is None:
                self.has_fps_attribute = False
            elif self.has_fps_attribute is not False:
                raise Exception('fps sequence attribute is empty')
        else:
            if self.has_fps_attribute is None:
                self.has_fps_attribute = True
            elif self.has_fps_attribute is not True:
                raise Exception('fps sequence attribute cannot only be specified to several sequences')


        if self.sequence is None:
            raise Exception("Call beginInitializingSequence() first")
        if len(self.sequence.object_id_attributes_mapper) == 0 or len(self.sequence.frames) == 0:
            raise Exception("Empty sequence")
        for object_information in self.sequence.object_id_attributes_mapper.values():
            object_information.frame_indices.sort()
        if 'has_present_attr' not in self.sequence.attributes:
            self.sequence.attributes['has_present_attr'] = False
        self.dataset.sequences.append(self.sequence)
        self.sequence = None

    def performStatistic(self):
        has_present_attribute_count = 0

        for sequence in self.dataset:
            if sequence.getAttribute('has_present_attr'):
                has_present_attribute_count += 1
        if has_present_attribute_count == len(self.dataset):
            self.dataset.attributes['has_present_attr'] = True
        elif has_present_attribute_count == 0:
            self.dataset.attributes['has_present_attr'] = False
        else:
            self.dataset.attributes['has_present_attr'] = has_present_attribute_count

        self.dataset.attributes['has_fps_attr'] = self.has_fps_attribute
