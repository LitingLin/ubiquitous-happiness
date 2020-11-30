from Dataset.MOT.Base.constructor import MultipleObjectTrackingDatasetConstructor
from Dataset.DataSplit import DataSplit
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os


def construct_ILSVRC_VID(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    data_type = seed.data_split
    class_id_name_mapper = {
        'n02691156': 'airplane',
        'n02419796': 'antelope',
        'n02131653': 'bear',
        'n02834778': 'bicycle',
        'n01503061': 'bird',
        'n02924116': 'bus',
        'n02958343': 'car',
        'n02402425': 'cattle',
        'n02084071': 'dog',
        'n02121808': 'domestic cat',
        'n02503517': 'elephant',
        'n02118333': 'fox',
        'n02510455': 'giant panda',
        'n02342885': 'hamster',
        'n02374451': 'horse',
        'n02129165': 'lion',
        'n01674464': 'lizard',
        'n02484322': 'monkey',
        'n03790512': 'motorcycle',
        'n02324045': 'rabbit',
        'n02509815': 'red panda',
        'n02411705': 'sheep',
        'n01726692': 'snake',
        'n02355227': 'squirrel',
        'n02129604': 'tiger',
        'n04468005': 'train',
        'n01662784': 'turtle',
        'n04530566': 'watercraft',
        'n02062744': 'whale',
        'n02391049': 'zebra'
    }

    def _parse_sequences(image_path: str, annotation_path: str, constructor: MultipleObjectTrackingDatasetConstructor, data_type: DataSplit):
        sequences = os.listdir(image_path)
        sequences.sort()

        for sequence in tqdm(sequences):
            constructor.beginInitializingSequence()
            constructor.setSequenceAttribute('type', data_type.name)
            constructor.setSequenceName(sequence)
            sequence_image_path = os.path.join(image_path, sequence)
            sequence_annotation_path = os.path.join(annotation_path, sequence)

            images = os.listdir(sequence_image_path)
            annotations = os.listdir(sequence_annotation_path)

            images = [image for image in images if image.endswith('.JPEG')]
            annotations = [annotation for annotation in annotations if annotation.endswith('.xml')]
            images.sort()
            annotations.sort()

            for image in images:
                constructor.addFrame(os.path.join(sequence_image_path, image))


            object_ids = {}

            for index_of_frame in range(len(images)):
                annotation_file_path = os.path.join(sequence_annotation_path, annotations[index_of_frame])

                tree = ET.parse(annotation_file_path)

                root = tree.getroot()

                for object_child in root.findall('object'):
                    object_id = None
                    bounding_box = None
                    object_name = None
                    occluded = None
                    generated = None
                    for attribute in object_child:#type: ET.Element
                        if attribute.tag == 'trackid':
                            assert object_id is None
                            object_id = int(attribute.text)
                        elif attribute.tag == 'name':
                            assert object_name is None
                            object_name = attribute.text
                        elif attribute.tag == 'bndbox':
                            assert bounding_box is None
                            xmin = None
                            xmax = None
                            ymin = None
                            ymax = None
                            for bounding_box_element in attribute:#type: ET.Element
                                if bounding_box_element.tag == 'xmax':
                                    assert xmax is None
                                    xmax = int(bounding_box_element.text)
                                elif bounding_box_element.tag == 'xmin':
                                    assert xmin is None
                                    xmin = int(bounding_box_element.text)
                                elif bounding_box_element.tag == 'ymax':
                                    assert ymax is None
                                    ymax = int(bounding_box_element.text)
                                elif bounding_box_element.tag == 'ymin':
                                    assert ymin is None
                                    ymin = int(bounding_box_element.text)
                                else:
                                    raise Exception
                            bounding_box = [xmin, ymin, xmax - xmin, ymax - ymin]
                        elif attribute.tag == 'occluded':
                            occluded = int(attribute.text)
                        elif attribute.tag == 'generated':
                            generated = int(attribute.text)
                        else:
                            raise Exception
                    assert object_id is not None
                    assert bounding_box is not None
                    assert object_name is not None
                    assert occluded is not None
                    assert generated is not None

                    if object_id not in object_ids:
                        object_ids[object_id] = object_name
                        constructor.addObject(object_id, class_id_name_mapper[object_name], {'WordNet ID': object_name})
                    else:
                        assert object_ids[object_id] == object_name

                    constructor.addRecord(index_of_frame, object_id, bounding_box, additional_attributes= {'occluded': occluded, 'generated': generated})
            constructor.endInitializingSequence()

    image_path = os.path.join(root_path, 'Data', 'VID')
    annotation_path = os.path.join(root_path, 'Annotations', 'VID')

    if data_type & DataSplit.Training:
        train_image_path = os.path.join(image_path, 'train')
        train_annotation_path = os.path.join(annotation_path, 'train')
        image_paths = [os.path.join(train_image_path, data_folder_name) for data_folder_name in os.listdir(train_image_path)]
        annotation_paths = [os.path.join(train_annotation_path, data_folder_name) for data_folder_name in os.listdir(train_annotation_path)]
        image_paths.sort()
        annotation_paths.sort()
        for train_image_path, train_annotation_path in zip(image_paths, annotation_paths):
            _parse_sequences(train_image_path, train_annotation_path, constructor, DataSplit.Training)
    if data_type & DataSplit.Validation:
        _parse_sequences(os.path.join(image_path, 'val'), os.path.join(annotation_path, 'val'), constructor, DataSplit.Validation)
