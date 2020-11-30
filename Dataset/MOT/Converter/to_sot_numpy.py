import numpy as np


def to_sot_numpy(dataset, sot_dataset):
    sot_dataset.category_name_id_mapper = dataset.category_name_id_mapper
    sot_dataset.category_names = dataset.category_names

    image_paths = []
    bounding_boxes = []
    sequence_attributes_indices = []
    sequence_category_ids = []

    for sequence in dataset:
        for track in sequence.getTrackIterator():
            attribute_index = len(image_paths)
            category_id = track.getObject().getCategoryId()
            sequence_attributes_indices.append(attribute_index)
            sequence_category_ids.append(category_id)
            for frame in track:
                image_path = frame.frame.image_path
                is_present = frame.getAttributeIsPresent()
                if is_present is False:
                    bounding_box = [-1, -1, -1, -1]
                else:
                    bounding_box = frame.getBoundingBox()
                    if bounding_box[0] + bounding_box[2] <= 0 or bounding_box[1] + bounding_box[3] <= 0:
                        bounding_box = [-1, -1, -1, -1]
                image_paths.append(image_path)
                bounding_boxes.append(bounding_box)

    sot_dataset.image_paths = image_paths
    sot_dataset.bounding_boxes = np.array(bounding_boxes)
    sot_dataset.sequence_attributes_indices = np.array(sequence_attributes_indices)
    sot_dataset.sequence_category_ids = np.array(sequence_category_ids)
