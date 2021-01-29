from Dataset.Base.Video.dataset import VideoDataset, VideoDatasetSequence, VideoDatasetFrame
from Dataset.Base.Common.Viewer.qt5_viewer import draw_object
from Miscellaneous.Viewer.qt5_viewer import Qt5Viewer
from PyQt5.QtGui import QPixmap, QColor
from Miscellaneous.simple_prefetcher import SimplePrefetcher
import random

__all__ = ['VideoDatasetViewer']


class _DatasetSequenceImageLoader:
    def __init__(self, sequence: VideoDatasetSequence):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index: int):
        frame = self.sequence[index]
        pixmap = QPixmap()
        assert pixmap.load(frame.get_image_path())

        return pixmap, frame, self.sequence


class VideoDatasetViewer:
    def __init__(self, dataset: VideoDataset):
        self.dataset = dataset
        self.viewer = Qt5Viewer()

        if self.dataset.has_category_id_name_map():
            self.category_id_color_map = {}
            for category_id in self.dataset.get_category_id_name_map().keys():
                color = [random.randint(0, 255) for _ in range(3)]
                self.category_id_color_map[category_id] = QColor(color[0], color[1], color[2], int(0.5 * 255))
        else:
            self.category_id_color_map = None

        sequence_names = []
        for sequence in self.dataset:
            sequence_names.append(sequence.get_name())

        self.viewer.addList(sequence_names, self._sequenceSelectedCallback)
        self.viewer.setTimerCallback(self._timerTimeoutCallback)

    def _sequenceSelectedCallback(self, index: int):
        self.sequence = SimplePrefetcher(_DatasetSequenceImageLoader(self.dataset[index]))
        self._stopTimer()
        self._startTimer()

    def _startTimer(self):
        self.sequence_iter = iter(self.sequence)
        self.viewer.startTimer()

    def _timerTimeoutCallback(self):
        try:
            image, frame, sequence = next(self.sequence_iter)
        except StopIteration:
            self._stopTimer()
            return
        frame: VideoDatasetFrame = frame
        sequence: VideoDatasetSequence = sequence
        painter = self.viewer.getPainter(image)

        with painter:
            for object_ in frame:
                object_category_id_accessor = object_
                if object_.has_id():
                    id_ = object_.get_id()
                    for sequence_object in sequence.get_object_iterator():
                        if sequence_object.get_id() == id_:
                            object_category_id_accessor = (object_, sequence_object)
                            break

                draw_object(painter, object_, object_category_id_accessor, object_, None, self.category_id_color_map, self.dataset)
        painter.update()

    def _stopTimer(self):
        self.viewer.stopTimer()

    def run(self):
        return self.viewer.runEventLoop()
