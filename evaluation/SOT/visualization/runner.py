from Viewer.viewer import SimpleViewer
from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped
from evaluation.SOT.util.simple_sequence_prefetcher import get_simple_sequence_data_prefetcher
from tqdm import tqdm
from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh


class SequenceRunner:
    def __init__(self, tracker, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped):
        self.tracker = tracker
        self.sequence = sequence
        assert len(sequence) > 1

    def __iter__(self):
        self.index = 0
        self.sequence_data_iter = iter(get_simple_sequence_data_prefetcher(self.sequence))
        return self

    def __next__(self):
        image, groundtruth_bounding_box = next(self.sequence_data_iter)
        if self.index == 0:
            self.tracker.initialize(image, groundtruth_bounding_box)
            predicted_bounding_box = groundtruth_bounding_box.tolist()
        else:
            predicted_bounding_box, _ = self.tracker.track(image)
        self.index += 1
        return (image.permute(1, 2, 0), bbox_xyxy2xywh(groundtruth_bounding_box.tolist()), bbox_xyxy2xywh(predicted_bounding_box))


class VisibleTrackerRunner:
    def __init__(self, tracker):
        self.tracker = tracker
        self.viewer = SimpleViewer()
        # self.viewer.switch_backend()

    def run(self, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped):
        assert isinstance(sequence, SingleObjectTrackingDatasetSequence_MemoryMapped)

        with tqdm(total=len(sequence)) as process_bar:
            process_bar.set_description_str(sequence.get_name())
            runner = SequenceRunner(self.tracker, sequence)

            for index_of_frame, (image, groundtruth_bounding_box, predicted_bounding_box) in enumerate(runner):
                self.viewer.clear()
                self.viewer.drawImage(image)
                self.viewer.drawBoundingBox(groundtruth_bounding_box)
                if index_of_frame != 0:
                    self.viewer.drawBoundingBox(predicted_bounding_box, (0, 1, 0))
                self.viewer.update()
                # self.viewer.waitKey()
                self.viewer.pause(0.0001)
                process_bar.update()

        self.viewer.waitKey()


def visualize_tracking_procedure_on_datasets(tracker, datasets):
    visualizer = VisibleTrackerRunner(tracker)
    for dataset in datasets:
        for sequence in dataset:
            visualizer.run(sequence)


def visualize_tracking_procedure_on_standard_datasets(tracker):
    from evaluation.SOT.runner import get_standard_evaluation_datasets
    visualize_tracking_procedure_on_datasets(tracker, get_standard_evaluation_datasets())
