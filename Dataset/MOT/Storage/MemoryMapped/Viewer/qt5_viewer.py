from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDataset_MemoryMapped, \
    MultipleObjectTrackingDatasetSequence_MemoryMapped, MultipleObjectTrackingDatasetFrame_MemoryMapped
from Miscellaneous.Viewer.qt5_viewer import Qt5Viewer
from Dataset.Base.Common.Viewer.qt5_viewer import draw_object
from PyQt5.QtGui import QPixmap, QColor
from Miscellaneous.simple_prefetcher import SimplePrefetcher
import random

__all__ = ['MOTDatasetQt5Viewer']


class _DatasetSequenceImageLoader:
    def __init__(self, sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index: int):
        frame = self.sequence.get_frame(index)
        pixmap = QPixmap()
        assert pixmap.load(frame.get_image_path())
        return pixmap, frame


class MOTDatasetQt5Viewer:
    def __init__(self, dataset: MultipleObjectTrackingDataset_MemoryMapped):
        self.dataset = dataset
        self.viewer = Qt5Viewer()

        if dataset.has_category_id_name_map():
            self.category_id_color_map = {}
            for category_id in self.dataset.get_category_id_name_map().keys():
                color = [random.randint(0, 255) for _ in range(3)]
                self.category_id_color_map[category_id] = QColor(color[0], color[1], color[2], int(0.5 * 255))
        else:
            self.category_id_color_map = None

        sequence_names = []
        for sequence in self.dataset:
            sequence_names.append(sequence.get_name())

        self.viewer.get_content_region().new_list(sequence_names, self._sequenceSelectedCallback)
        self.timer = self.viewer.new_timer()
        self.timer.set_callback(self._timerTimeoutCallback)

    def _sequenceSelectedCallback(self, index: int):
        if index < 0:
            return
        self.sequence = SimplePrefetcher(_DatasetSequenceImageLoader(self.dataset[index]))
        self._stopTimer()
        self._startTimer()

    def _startTimer(self):
        self.sequence_iter = iter(self.sequence)
        self.timer.start()

    def _timerTimeoutCallback(self):
        try:
            image, frame = next(self.sequence_iter)
        except StopIteration:
            self._stopTimer()
            return
        frame: MultipleObjectTrackingDatasetFrame_MemoryMapped = frame
        canvas = self.viewer.get_canvas()
        canvas.set_background(image)
        with canvas.get_painter() as painter:
            for object_ in frame:
                draw_object(painter, object_, object_, object_, object_, self.category_id_color_map, self.dataset, self.dataset)
        canvas.update()

    def _stopTimer(self):
        self.timer.stop()

    def run(self):
        return self.viewer.run_event_loop()
