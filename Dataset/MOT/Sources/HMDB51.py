from Dataset.MOT.Base.constructor import MultipleObjectTrackingDatasetConstructor
from Dataset.Utils.parse_THOTH_bb_file import parse_THOTH_bb_file
import os


# https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
# http://lear.inrialpes.fr/people/wang/improved_trajectories
def construct_HMDB51(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    annotation_path = seed.annotation_path
    actions = os.listdir(root_path)
    actions.sort()
    for action in actions:
        action_path = os.path.join(root_path, action)
        videos = os.listdir(action_path)
        videos.sort()
        for video in videos:
            video_path = os.path.join(action_path, video)
            images = os.listdir(video_path)
            images = [image for image in images if image.endswith('.jpg')]
            images.sort()
            annotation_file_path = os.path.join(annotation_path, video + '.bb')
            frame_annotations = parse_THOTH_bb_file(annotation_file_path)
            if len(frame_annotations) == 0:
                continue
            constructor.beginInitializingSequence()
            constructor.setSequenceName(video)
            constructor.setSequenceAttribute('action', action)
            for image in images:
                constructor.addFrame(os.path.join(video_path, image))

            registered_object_ids = {}
            for frame_index, object_annotations in frame_annotations.items():
                for object_id, (bounding_box, confidence_score) in object_annotations.items():
                    if object_id not in registered_object_ids:
                        constructor.addObject(object_id, 'person')
                        registered_object_ids[object_id] = None
                    constructor.addRecord(frame_index, object_id, bounding_box, None, {'confidence': confidence_score})
            constructor.endInitializingSequence()
