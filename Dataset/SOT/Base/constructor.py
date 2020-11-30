from __future__ import annotations
from typing import List, Dict, Union
from Dataset.SOT.Base.dataset import SingleObjectTrackingDataset
from Dataset.SOT.Base.sequence import SingleObjectTrackingDatasetSequence
from Dataset.SOT.Base.frame import SingleObjectTrackingDatasetFrame
from Dataset.CacheService.constructor import DatasetConstructor_CacheService_Base
from ._impl import _set_image_path, _get_or_allocate_category_id


class SingleObjectTrackingDatasetConstructor(DatasetConstructor_CacheService_Base):
    dataset: SingleObjectTrackingDataset
    sequence: SingleObjectTrackingDatasetSequence

    def __init__(self, dataset: SingleObjectTrackingDataset):
        super(SingleObjectTrackingDatasetConstructor, self).__init__(dataset)
        self.sequence = None
        self.has_category_attribute = None
        self.has_fps_attribute = None

    def __del__(self):
        if self.sequence is not None:
            raise Exception("Call endInitializeSequence() before destroying the Constructor")

    def beginInitializingSequence(self):
        if self.sequence is not None:
            raise Exception("Calling function beginInitializeSequence() twice")

        self.sequence = SingleObjectTrackingDatasetSequence()

    def setSequenceName(self, name: str):
        if self.sequence.name is not None:
            raise Exception("Calling setSequenceName() twice before endInitializeSequence()")
        self.sequence.name = name

    def setSequenceFPS(self, fps: Union[float, int]):
        self.sequence.fps = fps

    def setSequenceAttribute(self, name: str, value):
        self.sequence.attributes[name] = value

    def setSequenceObjectCategory(self, category: str):
        if self.sequence.category_id is not None:
            raise Exception("Calling setSequenceClassLabel() twice before endInitializeSequence()")

        self.sequence.category_id = _get_or_allocate_category_id(category, self.dataset.category_names, self.dataset.category_name_id_mapper)

    def addFrame(self, path: str):
        frame = SingleObjectTrackingDatasetFrame()
        frame.size, frame.image_path = _set_image_path(self.root_path, path)
        self.sequence.frames.append(frame)
        return len(self.sequence.frames) - 1

    def setFrameAttributes(self, index_of_frame: int, bounding_box: List = None, present: bool = None,
                           additional_attributes: Dict = None):
        if bounding_box is not None:
            assert isinstance(bounding_box, list) or isinstance(bounding_box, tuple)
        if present is not None:
            assert isinstance(present, bool)
        if additional_attributes is not None:
            assert isinstance(additional_attributes, dict)

        if 'has_present_attr' in self.sequence.attributes:
            assert present is not None
        if present is not None:
            if 'has_present_attr' not in self.sequence.attributes:
                self.sequence.attributes['has_present_attr'] = True
            else:
                assert self.sequence.attributes['has_present_attr']
        assert ~(bounding_box is None and present is None)
        frame = self.sequence.frames[index_of_frame]
        frame.bounding_box = bounding_box
        frame.is_present = present
        frame.attributes = additional_attributes

    def endInitializingSequence(self):
        if self.sequence.category_id is None:
            if self.has_category_attribute is None:
                self.has_category_attribute = False
            elif self.has_category_attribute is not False:
                raise Exception("category_name is empty")
        else:
            if self.has_category_attribute is None:
                self.has_category_attribute = True
            elif self.has_category_attribute is not True:
                raise Exception("category_name cannot only be specified to several sequences")

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

        if len(self.sequence.frames) == 0:
            raise Exception("Empty sequence")
        if self.sequence.name is None:
            raise Exception("name is empty")

        if 'has_present_attr' not in self.sequence.attributes:
            self.sequence.attributes['has_present_attr'] = False
        self.dataset.sequences.append(self.sequence)
        self.sequence = None

    def setDatasetAttribute(self, name, value):
        self.dataset.attributes[name] = value

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
        self.dataset.attributes['has_object_category_attr'] = self.has_category_attribute
