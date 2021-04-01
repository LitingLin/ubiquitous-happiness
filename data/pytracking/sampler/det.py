from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDataset_MemoryMapped
import random
import numpy as np
from ._sample_on_valid_frames import sample_on_valid_ids


def det_dataset_sampler(dataset: DetectionDataset_MemoryMapped, max_gap, num_search_frames, num_template_frames, frame_sample_mode):
    # Sample a sequence with enough visible frames
    while True:
        # Sample a sequence
        seq_id = random.randint(0, len(dataset) - 1)
        sequence = dataset[seq_id]
        track_id = random.randint(0, sequence.get_number_of_objects() - 1)
        track = sequence.get_object(track_id)

        valid_frame_ids = np.zeros(sequence.get_number_of_frames(), dtype=np.bool)

        if sequence.has_bounding_box_validity_flag():
            ind = track.get_all_frame_index()[track.get_all_bounding_box_validity_flag()]
            valid_frame_ids[ind] = True
        else:
            valid_frame_ids[track.get_all_frame_index()] = True

        if len(valid_frame_ids) >= 20 and sum(valid_frame_ids) > 2 * (num_search_frames + num_template_frames):
            break

    template_frame_ids, search_frame_ids = sample_on_valid_ids(valid_frame_ids, frame_sample_mode, max_gap, num_template_frames, num_search_frames)
    return sequence, track, template_frame_ids, search_frame_ids
