from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
import random
import numpy as np
from ._sample_on_valid_frames import sample_on_valid_ids


def sot_dataset_sampler(dataset: SingleObjectTrackingDataset_MemoryMapped, max_gap, num_search_frames,
                        num_template_frames, frame_sample_mode):
    # Sample a sequence with enough visible frames
    while True:
        # Sample a sequence
        seq_id = random.randint(0, len(dataset) - 1)
        sequence = dataset[seq_id]
        if sequence.has_bounding_box_validity_flag():
            valid_frame_ids = sequence.get_all_bounding_box_validity_flag()
        else:
            valid_frame_ids = np.ones(len(sequence), dtype=np.bool)

        if len(valid_frame_ids) >= 20 and sum(valid_frame_ids) > 2 * (num_search_frames + num_template_frames):
            break

    template_frame_ids, search_frame_ids = sample_on_valid_ids(valid_frame_ids, frame_sample_mode, max_gap, num_template_frames, num_search_frames)
    return sequence, template_frame_ids, search_frame_ids
