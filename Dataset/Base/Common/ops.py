from Dataset.Type.bounding_box_format import BoundingBoxFormat
import Dataset.Base.Common.Operator.bounding_box


def get_bounding_box(object_: dict):
    bounding_box = object_['bounding_box']
    bounding_box_validity = None
    if 'validity' in bounding_box:
        bounding_box_validity = bounding_box['validity']
    return bounding_box['value'], BoundingBoxFormat[bounding_box['format']], bounding_box_validity


def get_bounding_box_format(object_: dict):
    bounding_box = object_['bounding_box']
    return BoundingBoxFormat[bounding_box['format']]


def set_bounding_box_(object_: dict, bounding_box, bounding_box_format: BoundingBoxFormat, validity: bool = None):
    if validity is not None:
        assert isinstance(validity, bool)
    bounding_box = {
        'value': list(bounding_box),
        'format': bounding_box_format.name
    }
    if validity is not None:
        bounding_box['validity'] = validity
    object_['bounding_box'] = bounding_box


def bounding_box_convert_format_(object_: dict, target_format: BoundingBoxFormat, strict = True):
    bounding_box = object_['bounding_box']
    bounding_box['value'] = Dataset.Base.Common.Operator.bounding_box.convert_bounding_box_format(bounding_box['value'], BoundingBoxFormat[bounding_box['format']], target_format, strict)
    bounding_box['format'] = target_format


def bounding_box_fit_in_image_size_(object_: dict, image: dict):
    bounding_box = object_['bounding_box']
    bounding_box['value'] = Dataset.Base.Common.Operator.bounding_box.fit_in_image_boundary(bounding_box['value'], bounding_box['format'], image['size'])


def get_bounding_box_in_format(object_: dict, target_bounding_box_format: BoundingBoxFormat, strict = True):
    value, format, validity = get_bounding_box(object_)
    return Dataset.Base.Common.Operator.bounding_box.convert_bounding_box_format(value, format, target_bounding_box_format, strict), validity


def bounding_box_update_validity_(object_: dict, image: dict, skip_if_mark_non_validity=True):
    bounding_box = object_['bounding_box']
    if 'validity' in bounding_box and skip_if_mark_non_validity and not bounding_box['validity']:
        return
    image_size = image['size']
    bounding_box_value = bounding_box['value']
    bounding_box_format = BoundingBoxFormat[bounding_box['format']]
    bounding_box['validity'] = Dataset.Base.Common.Operator.bounding_box.check_bounding_box_validity_by_intersection_over_image(bounding_box_value, bounding_box_format, image_size)
