import os
from Dataset.DataSplit import DataSplit


def construct_UAV123(constructor, seed):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path

    def _parse_configseqs_m(root_path: str):
        sequence_attrs = {}
        file_path = os.path.join(root_path, 'configSeqs.m')
        with open(file_path) as fid:
            content = fid.read()
        begin_index = content.find('seqUAV123={')
        def _find_next_object_by_pattern(string: str, begin_pattern: str, end_pattern: str, begin_index: int):
            begin_index = string.find(begin_pattern, begin_index)
            if begin_index == -1:
                return None, -1
            begin_index += len(begin_pattern)
            end_index = content.find(end_pattern, begin_index)
            object_ = content[begin_index:end_index]
            end_index += len(end_pattern)
            return object_, end_index
        while True:
            sequence_name, begin_index = _find_next_object_by_pattern(content, "'name','", "'", begin_index)
            if sequence_name is None:
                break

            sequence_path_name, begin_index = _find_next_object_by_pattern(content, "d:\\data_seq\\UAV123\\", '\\', begin_index)
            start_frame, begin_index = _find_next_object_by_pattern(content, "startFrame',", ',', begin_index)
            start_frame = int(start_frame) - 1
            end_frame, begin_index = _find_next_object_by_pattern(content, "endFrame',", ',', begin_index)
            end_frame = int(end_frame)
            sequence_attrs[sequence_name] = (sequence_path_name, start_frame, end_frame)
        return sequence_attrs

    annotation_path = os.path.join(root_path, 'anno', 'UAV123')
    seq_path = os.path.join(root_path, 'data_seq', 'UAV123')
    sequence_attrs = _parse_configseqs_m(root_path)
    for sequence_name, (sequence_path_name, start_frame, end_frame) in sequence_attrs.items():
        words = sequence_name.split('_')
        class_label = words[0]
        for ind in reversed(range(len(class_label))):
            if class_label[ind].isdigit():
                class_label = class_label[:-1]
            else:
                break
        sequence_images_path = os.path.join(seq_path, sequence_path_name)

        gt_file_path = os.path.join(annotation_path, '{}.txt'.format(sequence_name))

        bounding_boxes = []
        occlusions = []
        for line in open(gt_file_path):
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split(',')
            assert len(words) == 4
            if words[0] == 'NaN':
                bounding_boxes.append(None)
                occlusions.append(True)
            else:
                bounding_boxes.append([int(words[0]), int(words[1]), int(words[2]), int(words[3])])
                occlusions.append(False)
        constructor.beginInitializingSequence()
        constructor.setSequenceName(sequence_name)
        constructor.setSequenceObjectCategory(class_label)
        assert end_frame - start_frame == len(bounding_boxes)
        for index in range(len(bounding_boxes)):
            index_of_image = index + start_frame + 1
            image_path = os.path.join(sequence_images_path, '{:06}.jpg'.format(index_of_image))
            constructor.setFrameAttributes(constructor.addFrame(image_path), bounding_boxes[index], not occlusions[index])
        constructor.endInitializingSequence()
