from Dataset.Filter.DataCleaning.ObjectCategory import DataCleaning_ObjectCategory
from Dataset.Filter.Selector import Selector
from Dataset.Filter.SortByImageRatio import SortByImageRatio
from Dataset.Filter.DataCleaning.Integrity import DataCleaning_Integrity
from Dataset.Filter.DataCleaning.BoundingBox import DataCleaning_BoundingBox
from .tweak_tool import VideoDatasetTweakTool
from Dataset.Type.bounding_box_format import BoundingBoxFormat

__all__ = ['apply_filters_on_video_dataset_']


def apply_filters_on_video_dataset_(dataset: dict, filters: list):
    if len(filters) == 0:
        return dataset

    if 'filters' not in dataset:
        dataset['filters'] = []

    filters_backup = dataset['filters']

    dataset_tweak_tool = VideoDatasetTweakTool(dataset)

    for filter_ in filters:
        if isinstance(filter_, Selector):
            dataset_tweak_tool.apply_index_filter(filter_(len(dataset['sequences'])))
        elif isinstance(filter_, SortByImageRatio):
            raise NotImplementedError
        elif isinstance(filter_, DataCleaning_BoundingBox):
            if filter_.fit_in_image_size:
                dataset_tweak_tool.bounding_box_fit_in_image_size()
            if filter_.format is not None:
                dataset_tweak_tool.bounding_box_convert_format(BoundingBoxFormat[filter_.format])
            if filter_.update_validity:
                dataset_tweak_tool.bounding_box_update_validity()
            if filter_.remove_non_validity_objects:
                dataset_tweak_tool.bounding_box_remove_non_validity_objects()
            if filter_.remove_empty_annotation_objects:
                dataset_tweak_tool.bounding_box_remove_empty_annotation_objects()
        elif isinstance(filter_, DataCleaning_Integrity):
            if filter_.remove_zero_annotation:
                dataset_tweak_tool.remove_empty_annotation()
            if filter_.remove_invalid_image:
                dataset_tweak_tool.remove_invalid_image()
        elif isinstance(filter_, DataCleaning_ObjectCategory):
            if filter_.category_ids_to_remove is not None:
                dataset_tweak_tool.remove_category_ids(filter_.category_ids_to_remove)
            if filter_.make_category_id_sequential:
                dataset_tweak_tool.make_category_id_sequential()

        filters_backup.append(filter_.serialize())
    dataset['filters'] = filters_backup
