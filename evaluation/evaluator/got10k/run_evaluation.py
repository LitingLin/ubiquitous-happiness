from native_extension import ImageDecoder
from .utils.viz import show_frame
import numpy as np
import time
from Dataset.Config.path import DatasetPath
from .experiments.otb import ExperimentOTB
from .experiments.got10k import ExperimentGOT10k
from .experiments.lasot import ExperimentLaSOT


class Tracker:
    def __init__(self, tracker, name, is_deterministic=True):
        self.tracker = tracker
        self.name = name
        self.is_deterministic = is_deterministic
        self.decoder = ImageDecoder()

    def track(self, image_files, initial_box, visualize=False):
        initial_frame_path = image_files[0]
        initial_frame = self.decoder.decode(initial_frame_path)
        pred_boxes = np.zeros((len(image_files), 4), dtype=np.float)
        pred_boxes[0] = initial_box

        times = np.zeros(len(image_files), dtype=np.float)

        start_time = time.time()
        self.tracker.initialize(initial_frame, initial_box)
        times[0] = time.time() - start_time

        if visualize:
            show_frame(initial_frame, initial_box)
        for index, frame_path in enumerate(image_files[1:]):
            frame = self.decoder.decode(frame_path)

            start_time = time.time()
            pred_box = self.tracker.track(frame)
            times[index + 1] = time.time() - start_time

            if visualize:
                show_frame(frame, pred_box)
            pred_boxes[index + 1] = pred_box

        return pred_boxes, times


def run_evaluation_on_tracker(tracker, name, is_deterministic, result_path: str, report_path: str, visualize=False):
    tracker = Tracker(tracker, name, is_deterministic)

    experiments = [
        ExperimentOTB(DatasetPath.OTB100_PATH, version=2013, result_dir=result_path, report_dir=report_path),
        ExperimentOTB(DatasetPath.OTB100_PATH, version='tb100', result_dir=result_path, report_dir=report_path),
        ExperimentGOT10k(DatasetPath.GOT10k_PATH, subset='val', result_dir=result_path, report_dir=report_path),
        ExperimentLaSOT(DatasetPath.LaSOT_PATH, subset='test', result_dir=result_path, report_dir=report_path)
    ]

    for e in experiments:
        e.run(tracker, visualize=visualize)
        e.report([tracker.name])



