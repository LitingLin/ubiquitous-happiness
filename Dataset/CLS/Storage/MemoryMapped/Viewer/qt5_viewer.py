from Dataset.CLS.Storage.MemoryMapped.dataset import ImageClassificationDataset_MemoryMapped
from Miscellaneous.Viewer.qt5_viewer import Qt5Viewer
from PyQt5.QtGui import QPixmap


class CLSDatasetQt5Viewer:
    def __init__(self, dataset: ImageClassificationDataset_MemoryMapped):
        self.dataset = dataset
        self.viewer = Qt5Viewer()
        image_names = []
        for index in range(len(self.dataset)):
            image_names.append(str(index))

        self.viewer.get_content_region().new_list(image_names, self._imageSelectedCallback)
        self.label = self.viewer.get_control_region().new_label()

    def _imageSelectedCallback(self, index: int):
        if index < 0:
            return
        image = self.dataset[index]
        pixmap = QPixmap()
        assert pixmap.load(image.get_image_path())
        canvas = self.viewer.get_canvas()
        canvas.set_background(pixmap)
        canvas.update()
        self.label.setText(self.dataset.get_category_id_name_map()[image.get_category_id()])

    def run(self):
        return self.viewer.run_event_loop()
