from Dataset.MOT.Base.dataset import MultipleObjectTrackingDataset
import os
import xml.etree.ElementTree as ET

# deprecated
def construct_UCF101THUMOSDataset(root_path: str, thumos_annotation_path: str):
    xml_namespace = {'default': 'http://lamp.cfar.umd.edu/viper#',
                     'data': 'http://lamp.cfar.umd.edu/viperdata#'}

    dataset = MultipleObjectTrackingDataset()
    constructor = dataset.getConstructor()
    constructor.setDatasetName('UCF-101-THUMOS')
    constructor.setRootPath(root_path)

    class_labels = os.listdir(thumos_annotation_path)
    class_labels.sort()

    for class_label in class_labels:
        current_sequences_path = os.path.join(root_path, class_label)
        current_annotation_sequences_path = os.path.join(thumos_annotation_path, class_label)

        sequences = os.listdir(current_annotation_sequences_path)
        sequences = [sequence[:-5] for sequence in sequences if sequence.endswith('.xgtf')]
        sequences.sort()

        for sequence in sequences:
            sequence_bounding_box_annotation_file_path = os.path.join(current_annotation_sequences_path, "{}.xgtf".format(sequence))
            sequence_images_path = os.path.join(current_sequences_path, sequence)

            try:
                tree = ET.parse(sequence_bounding_box_annotation_file_path)
            except ET.ParseError as e:
                print('Error: failed to parse annotation for sequence {}'.format(sequence))
                continue

            constructor.beginInitializingSequence()

            images = os.listdir(sequence_images_path)
            images = [image for image in images if image.endswith('.jpg')]
            images.sort()
            for image in images:
                image_path = os.path.join(sequence_images_path, image)
                constructor.addFrame(image_path)

            root = tree.getroot()

            object_id = 0
            for data in root.findall('default:data', xml_namespace):
                for child in data:#type: ET.Element
                    for object_ in child.findall('default:object', xml_namespace):#type: ET.Element
                        object_name = object_.attrib['name']

                        bounding_boxes = {}
                        attributes = {}

                        for attribute in object_.findall('default:attribute', xml_namespace):
                            if attribute.attrib['name'] == 'Location':
                                for attribute_value in attribute:#type: ET.Element
                                    assert attribute_value.tag == '{{{}}}bbox'.format(xml_namespace['data'])
                                    frame_span = attribute_value.attrib['framespan']
                                    frame_span = frame_span.split(':')
                                    assert len(frame_span) == 2
                                    frame_span = [int(frame_span[0]), int(frame_span[1])]

                                    bounding_box = [int(attribute_value.attrib['x']), int(attribute_value.attrib['y']),
                                                    int(attribute_value.attrib['width']), int(attribute_value.attrib['height'])]

                                    for index in range(frame_span[0] - 1, frame_span[1]):
                                        bounding_boxes[index] = bounding_box
                            else:
                                attribute_name = attribute.attrib['name']
                                has_true_value = False

                                parsed_attribute = {}

                                for attribute_value in attribute:#type: ET.Element
                                    frame_span = attribute_value.attrib['framespan']
                                    frame_span = frame_span.split(':')
                                    assert len(frame_span) == 2
                                    frame_span = [int(frame_span[0]), int(frame_span[1])]
                                    value = attribute_value.attrib['value']
                                    if attribute_value.tag == '{{{}}}bvalue'.format(xml_namespace['data']):
                                        if value == 'true':
                                            value = True
                                            has_true_value = True
                                        elif value == 'false':
                                            value = False
                                        else:
                                            raise Exception('Unexpected value: {}'.format(value))
                                    elif attribute_value.tag == '{{{}}}fvalue'.format(xml_namespace['data']):
                                        value = float(value)
                                        has_true_value = True
                                    else:
                                        raise Exception('Unknown attribute value type')

                                    for index in range(frame_span[0] - 1, frame_span[1]):
                                        parsed_attribute[index] = value

                                if has_true_value:
                                    for index, value in parsed_attribute.items():
                                        if index not in attributes:
                                            attributes[index] = {}
                                        if attribute_name in attributes[index]:
                                            raise Exception("Duplicate attribute name")
                                        attributes[index][attribute_name] = value

                        constructor.addObject(object_id, object_name)

                        for index, bounding_box in bounding_boxes.items():
                            if index >= len(images):
                                print('warning: sequence {} length mismatch between video and object_id {} in annotation'.format(sequence, object_id))
                                break
                            if index in attributes:
                                additional_attributes = attributes[index]
                            else:
                                additional_attributes = {}
                            constructor.addRecord(index, object_id, bounding_box, additional_attributes=additional_attributes)

                        object_id += 1
            constructor.endInitializingSequence()

    return dataset
