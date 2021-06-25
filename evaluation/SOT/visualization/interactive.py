from Miscellaneous.Viewer.qt5_viewer import Qt5Viewer, QPen, QColor, Qt
from ._sequence_runner import SequenceRunner
import numpy as np
from Miscellaneous.qt_numpy_interop import numpy_rgb888_to_qimage
from data.operator.bbox.validity import bbox_is_valid


def _get_bbox_format_transformer():
    from data.operator.bbox.transform.compile import compile_bbox_transform
    from data.types.bounding_box_format import BoundingBoxFormat
    from data.types.pixel_coordinate_system import PixelCoordinateSystem
    from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
    from data.types.pixel_definition import PixelDefinition

    bbox_transformer = compile_bbox_transform(BoundingBoxFormat.XYXY, BoundingBoxFormat.XYWH,
                                              PixelCoordinateSystem.Aligned, PixelCoordinateSystem.Aligned,
                                              BoundingBoxCoordinateSystem.Spatial,
                                              BoundingBoxCoordinateSystem.Rasterized,
                                              PixelDefinition.Point)
    return bbox_transformer


class InteractiveDatasetsRunner:
    def __init__(self, tracker, datasets):
        self.viewer = Qt5Viewer()
        self.tracker = tracker

        total_number_of_sequences = 0
        for dataset in datasets:
            total_number_of_sequences += len(dataset)
        index_of_datasets = np.empty(total_number_of_sequences, dtype=np.uint32)
        index_of_sequences = np.empty(total_number_of_sequences, dtype=np.uint32)
        sequence_names = []

        _index = 0
        for index_of_dataset, dataset in enumerate(datasets):
            for index_of_sequence, sequence in enumerate(dataset):
                index_of_datasets[_index] = index_of_dataset
                index_of_sequences[_index] = index_of_sequence
                sequence_names.append(sequence.get_name())

        self.viewer.addList(sequence_names, self._sequence_selected_callback)
        self.datasets = datasets
        self.index_of_datasets = index_of_datasets
        self.index_of_sequences = index_of_sequences
        self.viewer.setTimerInterval(int(round(1000 / 60)))
        self.viewer.setTimerCallback(self._sequence_selected_callback)
        groundtruth_color = QColor(255, 0, 0, int(255 * 0.5))
        self.groundtruth_pen = QPen(groundtruth_color)
        self.groundtruth_invalid_pen = QPen(groundtruth_color)
        self.groundtruth_invalid_pen.setStyle(Qt.DashDotDotLine)
        predicted_color = QColor(0, 255, 0, int(255 * 0.5))
        self.predicted_pen = QPen(predicted_color)
        self.bbox_transformer = _get_bbox_format_transformer()

    def run(self):
        return self.viewer.runEventLoop()

    def _sequence_selected_callback(self, index):
        index_of_dataset = self.index_of_datasets[index]
        index_of_sequence = self.index_of_sequences[index]
        dataset = self.datasets[index_of_dataset]
        sequence = dataset[index_of_sequence]

        if hasattr(self, 'sequence_runner_iter'):
            del self.sequence_runner_iter
        self.sequence_runner_iter = iter(SequenceRunner(self.tracker, sequence))
        self.viewer.startTimer()

    def _timer_callback(self):
        try:
            index_of_frame, image, groundtruth_bbox, groundtruth_bbox_validity_flag, predicted_bbox = \
                next(self.sequence_runner_iter)
        except StopIteration:
            self.viewer.stopTimer()
            return

        qimage = numpy_rgb888_to_qimage(image)
        with self.viewer.getPainter(qimage) as painter:
            if groundtruth_bbox_validity_flag is False:
                if bbox_is_valid(groundtruth_bbox):
                    painter.setPen(self.groundtruth_invalid_pen)
                    painter.drawRect(self.bbox_transformer(groundtruth_bbox))
            else:
                painter.setPen(self.groundtruth_pen)
                painter.drawRect(self.bbox_transformer(groundtruth_bbox))
            if index_of_frame != 0:
                if bbox_is_valid(predicted_bbox):
                    painter.setPen(self.predicted_pen)
                    painter.drawRect(self.bbox_transformer(predicted_bbox))
            painter.update()


def visualize_tracker_on_standard_datasets(tracker):
    from evaluation.SOT.runner import get_standard_evaluation_datasets
    viewer = InteractiveDatasetsRunner(tracker, get_standard_evaluation_datasets())
    return viewer.run()
