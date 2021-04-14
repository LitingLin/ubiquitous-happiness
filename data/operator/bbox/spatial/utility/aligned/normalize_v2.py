from data.operator.bbox.spatial.utility.aligned.image import get_image_bounding_box


def bbox_normalize(bbox, image_size):
    image_bounding_box = get_image_bounding_box(image_size)
    image_bounding_box_wh = (image_bounding_box[2] - image_bounding_box[0],
                             image_bounding_box[3] - image_bounding_box[1])
    return tuple(
        v / image_bounding_box_wh[0] if i % 2 == 0 else v / image_bounding_box_wh[1] for i, v in enumerate(bbox))


def bbox_denormalize(bbox, image_size):
    image_bounding_box = get_image_bounding_box(image_size)
    image_bounding_box_wh = (image_bounding_box[2] - image_bounding_box[0],
                             image_bounding_box[3] - image_bounding_box[1])
    return tuple(
        v * image_bounding_box_wh[0] if i % 2 == 0 else v * image_bounding_box_wh[1] for i, v in enumerate(bbox))
