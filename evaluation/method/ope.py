import numpy as np


def run_one_pass_evaluation_on_sequence(sequence, tracker):
    sequence_iter = iter(sequence)
    initial_frame, initial_bbox = next(sequence_iter)
    tracker.initialize(initial_frame, initial_bbox)
    groundtruth_bboxes = [initial_bbox]
    predicted_bboxes = [initial_bbox]
    while True:
        try:
            frame, bbox = next(sequence_iter)
        except StopIteration:
            break

        predicted_bbox = tracker.track(frame)

        groundtruth_bboxes.append(bbox)
        predicted_bboxes.append(predicted_bbox)

    return np.array(groundtruth_bboxes), np.array(predicted_bboxes)
