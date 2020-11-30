from Dataset.MOT.Base.dataset import MultipleObjectTrackingDataset

import copy

from Dataset.Filter.DataCleaner_BoundingBox import DataCleaner_BoundingBox
from Dataset.Filter.DataCleaner_Integrity import DataCleaner_Integrity
from Dataset.Filter.DataCleaner_NoAbsentObjects import DataCleaner_NoAbsentObjects


def apply_filters(dataset: MultipleObjectTrackingDataset, filters, make_cache=True):
    new_filters = copy.copy(dataset.filters)
    new_filters.extend(filters)

    new_dataset = copy.deepcopy(dataset)
    constructor = new_dataset.getConstructor()
    modifier = new_dataset.getModifier()

    for filter_ in filters:
        if isinstance(filter_, DataCleaner_BoundingBox):
            for sequence in modifier:
                for track in sequence.iterateByObjectTrack():
                    for frame_object in track:
                        frame_object.setBoundingBox(filter_(frame_object.getBoundingBox(), frame_object.getImageSize()))
        elif isinstance(filter_, DataCleaner_NoAbsentObjects):
            pass
        elif isinstance(filter_, DataCleaner_Integrity):
            if filter_.no_zero_size_image:
                for sequence in modifier:
                    for frame in sequence.iterateByFrame():
                        size = frame.getImageSize()
                        if size[0] == 0 or size[1] == 0:
                            frame.delete()
            if filter_.no_zero_annotations:
                for sequence in modifier:
                    for frame in sequence.iterateByFrame():
                        for object_ in frame:
                            if object_.getBoundingBox() is None:
                                object_.delete()

                    for frame in sequence.iterateByFrame():
                        if len(frame) == 0:
                            frame.delete()
                    for track in sequence.iterateByObjectTrack():
                        if len(track) == 0:
                            track.delete()
                    if len(sequence) == 0:
                        sequence.delete()

    new_dataset.filters = new_filters
    if make_cache:
        constructor.makeCache()
    return new_dataset
