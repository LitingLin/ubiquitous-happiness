import random


def sample_visible_ids(visible, num_ids=1, min_id=None, max_id=None):
    """ Samples num_ids frames between min_id and max_id for which target is visible

    args:
        visible - 1d Tensor indicating whether target is visible for each frame
        num_ids - number of frames to be samples
        min_id - Minimum allowed frame number
        max_id - Maximum allowed frame number

    returns:
        list - List of sampled frame numbers. None if not sufficient visible frames could be found.
    """
    if num_ids == 0:
        return []
    if min_id is None or min_id < 0:
        min_id = 0
    if max_id is None or max_id > len(visible):
        max_id = len(visible)

    valid_ids = [i for i in range(min_id, max_id) if visible[i]]

    # No visible ids
    if len(valid_ids) == 0:
        return None

    return random.choices(valid_ids, k=num_ids)


def sample_on_valid_ids(valid_frame_ids, frame_sample_mode, max_gap, num_template_frames, num_search_frames):
    template_frame_ids = None
    search_frame_ids = None
    gap_increase = 0
    if frame_sample_mode == 'interval':
        # Sample frame numbers within interval defined by the first frame
        while search_frame_ids is None:
            base_frame_id = sample_visible_ids(valid_frame_ids, num_ids=1)
            extra_template_frame_ids = sample_visible_ids(valid_frame_ids, num_ids=num_template_frames - 1,
                                                                min_id=base_frame_id[
                                                                           0] - max_gap - gap_increase,
                                                                max_id=base_frame_id[
                                                                           0] + max_gap + gap_increase)
            if extra_template_frame_ids is None:
                gap_increase += 5
                continue
            template_frame_ids = base_frame_id + extra_template_frame_ids
            search_frame_ids = sample_visible_ids(valid_frame_ids, num_ids=num_search_frames,
                                                        min_id=template_frame_ids[0] - max_gap - gap_increase,
                                                        max_id=template_frame_ids[0] + max_gap + gap_increase)
            gap_increase += 5  # Increase gap until a frame is found

    elif frame_sample_mode == 'causal':
        # Sample search and template frames in a causal manner, i.e. search_frame_ids > template_frame_ids
        while search_frame_ids is None:
            base_frame_id = sample_visible_ids(valid_frame_ids, num_ids=1, min_id=num_template_frames - 1,
                                                     max_id=len(valid_frame_ids) - num_search_frames)
            prev_frame_ids = sample_visible_ids(valid_frame_ids, num_ids=num_template_frames - 1,
                                                      min_id=base_frame_id[0] - max_gap - gap_increase,
                                                      max_id=base_frame_id[0])
            if prev_frame_ids is None:
                gap_increase += 5
                continue
            template_frame_ids = base_frame_id + prev_frame_ids
            search_frame_ids = sample_visible_ids(valid_frame_ids, min_id=template_frame_ids[0] + 1,
                                                        max_id=template_frame_ids[0] + max_gap + gap_increase,
                                                        num_ids=num_search_frames)
            # Increase gap until a frame is found
            gap_increase += 5

    return template_frame_ids, search_frame_ids
