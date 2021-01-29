from Dataset.Base.Image.manipulator import ImageDatasetManipulator
from Dataset.Type.bounding_box_format import BoundingBoxFormat
import numpy as np
import copy


class ImageDatasetTweakTool:
    def __init__(self, dataset: dict):
        self.manipulator = ImageDatasetManipulator(dataset)

    def apply_index_filter(self, indices):
        self.manipulator.apply_index_filter(indices)

    def sort_by_image_size_ratio(self):
        image_sizes = []
        for image in self.manipulator:
            image_sizes.append(image.get_image_size())
        image_sizes = np.array(image_sizes)
        ratio = image_sizes[:, 1] / image_sizes[:, 0]
        indices = ratio.argsort()
        self.manipulator.apply_index_filter(indices)

    def bounding_box_convert_format(self, target_format: BoundingBoxFormat):
        for image in self.manipulator:
            for object_ in image:
                if object_.has_bounding_box():
                    object_.convert_bounding_box_format(target_format)

    def bounding_box_fit_in_image_size(self, exclude_non_validity=True):
        for image in self.manipulator:
            for object_ in image:
                if object_.has_bounding_box():
                    _, _, bounding_box_validity = object_.get_bounding_box()
                    if exclude_non_validity:
                        if bounding_box_validity is False:
                            continue
                    object_.bounding_box_fit_in_image_size()

    def bounding_box_update_validity(self, skip_if_mark_non_validity=True):
        for image in self.manipulator:
            for object_ in image:
                if object_.has_bounding_box():
                    object_.bounding_box_update_validity(skip_if_mark_non_validity)

    def bounding_box_remove_non_validity_objects(self):
        for image in self.manipulator:
            for object_ in image:
                if object_.has_bounding_box():
                    _, _, validity = object_.get_bounding_box()
                    if validity is False:
                        object_.delete()

    def bounding_box_remove_empty_annotation_objects(self):
        for image in self.manipulator:
            for object_ in image:
                if not object_.has_bounding_box():
                    object_.delete()

    def remove_empty_annotation(self):
        for image in self.manipulator:
            if len(image) == 0:
                image.delete()

    def remove_invalid_image(self):
        for image in self.manipulator:
            w, h = image.get_image_size()
            if w == 0 or h == 0:
                image.delete()

    def remove_category_ids(self, category_ids: list):
        for image in self.manipulator:
            for object_ in image:
                if object_.has_category_id():
                    if object_.get_category_id() in category_ids:
                        object_.delete()

        category_id_name_map: dict = copy.copy(self.manipulator.get_category_id_name_map())
        for category_id in category_ids:
            category_id_name_map.pop(category_id)
        self.manipulator.set_category_id_name_map(category_id_name_map)

    def make_category_id_sequential(self):
        category_id_name_map = self.manipulator.get_category_id_name_map()
        new_category_ids = list(range(len(category_id_name_map)))
        old_new_category_id_map = {o: n for n, o in zip(new_category_ids, category_id_name_map.keys())}
        for image in self.manipulator:
            for object_ in image:
                if object_.has_category_id():
                    if object_.get_category_id() in old_new_category_id_map:
                        object_.set_category_id(old_new_category_id_map[object_.get_category_id()])
        new_category_id_name_map = {n: category_id_name_map[o] for n, o in
                                    zip(new_category_ids, category_id_name_map.keys())}
        self.manipulator.set_category_id_name_map(new_category_id_name_map)
