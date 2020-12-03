from Dataset.Filter.DataCleaner_BoundingBox import DataCleaner_BoundingBox
from Dataset.Filter.DataCleaner_Integrity import DataCleaner_Integrity
from Dataset.Filter.DataCleaner_NoAbsentObjects import DataCleaner_NoAbsentObjects
from Dataset.Filter.Selector import Selector
import copy

from Dataset.SOT.Base.dataset import SingleObjectTrackingDataset


def apply_filters(dataset: SingleObjectTrackingDataset, filters: list, make_cache:bool=True):
    assert isinstance(dataset, SingleObjectTrackingDataset)

    new_filters = copy.copy(dataset.filters)
    new_filters.extend(filters)

    new_dataset = copy.deepcopy(dataset)
    constructor = new_dataset.getConstructor()

    modifier = new_dataset.getModifier()

    for filter_ in filters:
        if isinstance(filter_, DataCleaner_BoundingBox):
            for sequence in modifier:
                for frame in sequence:
                    bounding_box = frame.getBoundingBox()
                    if bounding_box is not None:
                        frame.setBoundingBox(filter_(frame.getBoundingBox(), frame.getImageSize()))

        elif isinstance(filter_, DataCleaner_NoAbsentObjects):
            for sequence in modifier:
                for frame in sequence:
                    is_present = frame.getAttributeIsPresent()
                    if is_present is not None:
                        if not is_present:
                            frame.delete()

        elif isinstance(filter_, DataCleaner_Integrity):
            if filter_.no_zero_size_image:
                for sequence in modifier:
                    for frame in sequence:
                        size = frame.getImageSize()
                        if size[0] == 0 or size[1] == 0:
                            frame.delete()

            if filter_.no_zero_annotations:
                for sequence in modifier:
                    for frame in sequence:
                        if frame.getBoundingBox() is None:
                            frame.delete()

                for sequence in modifier:
                    if len(sequence) == 0:
                        sequence.delete()
        elif isinstance(filter_, Selector):
            modifier.applyIndicesFilter(filter_(len(modifier)))
        else:
            raise Exception

    new_dataset.filters = new_filters
    if make_cache:
        constructor.makeCache()
    return new_dataset
