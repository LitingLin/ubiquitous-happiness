from Dataset.Detection.VehicleTask.base import BaseDataset
import pickle


def serialize(dataset: BaseDataset, path: str):
    with open(path, 'wb') as fid:
        pickle.dump(
            (dataset.root_dir, dataset.bounding_boxes, dataset.image_paths,
             dataset.classes, dataset.classIndexer, dataset.classNamesMapper),
            fid)


def deserialize(path: str, root_path: str = None):
    with open(path, 'rb') as fid:
        objects = pickle.load(fid)

    dataset = BaseDataset(objects[0], objects[5])
    dataset.bounding_boxes = objects[1]
    dataset.image_paths = objects[2]
    dataset.classes = objects[3]
    dataset.classIndexer = objects[4]

    if root_path is not None:
        dataset.root_dir = root_path

    return dataset
