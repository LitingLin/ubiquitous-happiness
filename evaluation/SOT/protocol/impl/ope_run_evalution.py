import os
from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped
from Dataset.Base.Common.constructor import DatasetProcessBar
import shutil
import time
import pickle
import numpy as np
from Miscellaneous.simple_prefetcher import SimplePrefetcher
import torchvision.io


def get_sequence_result_path(result_path, sequence, run_time=None):
    if run_time is not None:
        sequence_name = f'{sequence.get_name()}-{run_time}'
    else:
        sequence_name = sequence.get_name()

    return os.path.join(result_path, sequence_name), sequence_name


def run_one_pass_evaluation_on_sequence(tracker, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, result_path, run_time, process_bar: DatasetProcessBar):
    result_path, sequence_name = get_sequence_result_path(result_path, sequence, run_time)
    if os.path.exists(result_path):
        process_bar.set_sequence_name(f'{sequence_name}: evaluated')
        process_bar.update()
        return

    tmp_path = result_path + '-tmp'
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    os.mkdir(tmp_path)

    predicted_bboxes = []
    times = []

    class _Sequence_Data_Getter:
        def __init__(self, sequence):
            self.sequence = sequence

        def __getitem__(self, index: int):
            frame = self.sequence[index]
            return torchvision.io.read_image(frame.get_image_path(), torchvision.io.image.ImageReadMode.RGB), frame.get_bounding_box()

        def __len__(self):
            return len(self.sequence)

    sequence_data_getter = _Sequence_Data_Getter(sequence)
    sequence_data_getter = SimplePrefetcher(sequence_data_getter)

    for index_of_frame, (image, bounding_box) in enumerate(sequence_data_getter):
        begin_time = time.perf_counter()
        if index_of_frame == 0:
            tracker.initialize(image, bounding_box)
            predicted_bboxes.append(bounding_box)
        else:
            predicted_bbox = tracker.track(image)
            predicted_bboxes.append(predicted_bbox)
        times.append(time.perf_counter() - begin_time)
    predicted_bboxes = np.array(predicted_bboxes)
    times = np.array(times)
    with open(os.path.join(tmp_path, 'bounding_box.p'), 'wb') as f:
        pickle.dump(predicted_bboxes, f)
    np.savetxt(os.path.join(tmp_path, 'bounding_box.txt'), predicted_bboxes, fmt='%.3f', delimiter=',')
    with open(os.path.join(tmp_path, 'time.p'), 'wb') as f:
        pickle.dump(times, f)
    np.savetxt(os.path.join(tmp_path, 'time.txt'), times, fmt='%.3f', delimiter=',')

    os.rename(tmp_path, result_path)
    process_bar.set_sequence_name(f'{sequence_name}: FPS {1.0 / times.mean()}')

    process_bar.update()


def check_no_sequence_name_conflict(datasets):
    names = set()
    for dataset in datasets:
        for sequence in dataset:
            assert sequence.get_name() not in names
            names.add(sequence.get_name())


def run_one_pass_evaluation_on_dataset(dataset, tracker, result_path, run_times=None):
    process_bar = DatasetProcessBar()
    process_bar.set_dataset_name(dataset.get_name())
    total_sequences = len(dataset)
    if run_times is not None:
        total_sequences *= run_times
    process_bar.set_total(total_sequences)
    for sequence in dataset:
        if run_times is not None:
            for run_time in range(run_times):
                run_one_pass_evaluation_on_sequence(tracker, sequence, result_path, run_time, process_bar)
        else:
            run_one_pass_evaluation_on_sequence(tracker, sequence, result_path, None, process_bar)
    process_bar.close()


def prepare_result_path(output_path, datasets, tracker_name):
    output_path = os.path.join(output_path, 'ope', tracker_name, 'result')
    os.makedirs(output_path, exist_ok=True)

    check_no_sequence_name_conflict(datasets)
    return output_path
