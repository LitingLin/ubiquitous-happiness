from Dataset.Detection.Base.dataset import DetectionDataset

import copy

from Dataset.Filter.DataCleaner_BoundingBox import DataCleaner_BoundingBox
from Dataset.Filter.DataCleaner_Integrity import DataCleaner_Integrity
from Dataset.Filter.DataCleaner_NoAbsentObjects import DataCleaner_NoAbsentObjects
from Dataset.Filter.SortByImageRatio import SortByImageRatio


def apply_filters(dataset: DetectionDataset, filters, make_cache = True):
    new_filters = copy.copy(dataset.filters)
    new_filters.extend(filters)

    new_dataset = copy.deepcopy(dataset)
    constructor = new_dataset.getConstructor()
    modifier = new_dataset.getModifier()
    for filter in filters:
        if isinstance(filter, DataCleaner_BoundingBox):
            for image in modifier:
                for object_ in image:
                    object_.setBoundingBox(filter(object_.getBoundingBox(), image.getImageSize()))
        elif isinstance(filter, DataCleaner_Integrity):
            if filter.no_zero_size_image:
                for image in modifier:
                    size = image.getImageSize()
                    if size[0] == 0 or size[1] == 0:
                        image.delete()
            if filter.no_zero_annotations:
                modifier.removeZeroAnnotationObjects()
                modifier.removeZeroAnnoationImages()
        elif isinstance(filter, DataCleaner_NoAbsentObjects):
            modifier.removeAbsentObjects()
        elif isinstance(filter, SortByImageRatio):
            modifier.sortByImageRatio()
        else:
            raise ValueError('Unsupported')

    new_dataset.filters = new_filters
    if make_cache:
        constructor.makeCache()
    return new_dataset
