from Dataset.CLS.Storage.MemoryMapped.dataset import ImageClassificationDataset_MemoryMapped
from Dataset.Base.Common.Viewer.qt5_viewer import draw_object
from Miscellaneous.Viewer.qt5_viewer import Qt5Viewer
from PyQt5.QtGui import QPixmap, QColor
import random


class DetectionDatasetQt5Viewer:
    def __init__(self, dataset: ImageClassificationDataset_MemoryMapped):
        self.dataset = dataset
        self.viewer = Qt5Viewer()
        image_names = []
        for index in range(len(self.dataset)):
            image_names.append(str(index))

        self.viewer.addList(image_names, self._imageSelectedCallback)
        self.label = self.viewer.addLabel('')

    def _imageSelectedCallback(self, index: int):
        image = self.dataset[index]
        pixmap = QPixmap()
        assert pixmap.load(image.get_image_path())
        painter = self.viewer.getPainter(pixmap)
        painter.update()
        self.label.setText(self.dataset.get_category_id_name_map()[image.get_category_id()])

    def run(self):
        return self.viewer.runEventLoop()
