from PyQt5.QtWidgets import QApplication, QDialog, QSlider, QLabel, QPushButton, QListWidget, \
    QListWidgetItem, QAbstractItemView, QVBoxLayout, QHBoxLayout, QFileDialog, QSizePolicy, QLineEdit, QMessageBox, QWidget, QCheckBox
from PyQt5.QtGui import QPainter, QPixmap, QPolygonF, QIntValidator, QResizeEvent, QFontMetrics, QColor, QBrush
from PyQt5.QtCore import pyqtSlot, QObject, Qt, QPointF, QSettings, QUrl, QRectF
import typing
import random

import sys
import Dataset.Detection
import Dataset.DataSplit

def constructFileDialog(parent: QWidget, settings: QSettings):
    fileDialog = QFileDialog(parent)
    if settings.contains('QFileDialogState'):
        fileDialog.setDirectoryUrl(QUrl.fromEncoded(settings.value('QFileDialogState')))
    return fileDialog

def saveFileDialogState(fileDialog: QFileDialog, settings: QSettings):
    settings.setValue('QFileDialogState', fileDialog.directoryUrl().toEncoded())

def selectSingleFile(parent: QWidget, settings: QSettings):
    fileDialog = constructFileDialog(parent, settings)
    fileDialog.setFileMode(QFileDialog.ExistingFile)
    fileName = None
    if fileDialog.exec_():
        fileName = fileDialog.selectedFiles()[0]
    saveFileDialogState(fileDialog, settings)
    return fileName

def selectSingleDirectory(parent: QWidget, settings: QSettings):
    fileDialog = constructFileDialog(parent, settings)
    fileDialog.setFileMode(QFileDialog.Directory)
    fileDialog.setOption(QFileDialog.ShowDirsOnly, True)
    dirName = None
    if fileDialog.exec_():
        dirName = fileDialog.selectedFiles()[0]
    saveFileDialogState(fileDialog, settings)
    return dirName

def browseForFileSavingPath(parent: QWidget, settings: QSettings):
    fileDialog = constructFileDialog(parent, settings)
    fileDialog.setFileMode(QFileDialog.AnyFile)
    fileDialog.setAcceptMode(QFileDialog.AcceptSave)
    fileName = None
    if fileDialog.exec_():
        fileName = fileDialog.selectedFiles()[0]
    saveFileDialogState(fileDialog, settings)
    return fileName

def messageBoxError(title, text):
    messageBox = QMessageBox(QMessageBox.Critical, title, text, QMessageBox.Ok)
    messageBox.exec_()

def messageBoxOk(title, text):
    messageBox = QMessageBox(QMessageBox.Information, title, text, QMessageBox.Ok)
    messageBox.exec_()

class Viewer(QObject):
    class QLabel_QPixmapAutoFitToLabelSize(QLabel):
        def resizeEvent(self, qResizeEvent: QResizeEvent):
            super().resizeEvent(qResizeEvent)
            Viewer.fitQPixmapToLabel(self)

    iterator: Dataset.Detection.BaseDataset

    def __init__(self):
        super(Viewer, self).__init__()

        app = QApplication(sys.argv)

        window = QDialog()
        window.setWindowState(Qt.WindowMaximized)
        window.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

        settings = QSettings('viewer.ini', QSettings.IniFormat, app)

        base_layout = QVBoxLayout()
        layout1 = QHBoxLayout()
        layout2 = QHBoxLayout()

        viewerLayout = QVBoxLayout()

        imageLabelWidget = Viewer.QLabel_QPixmapAutoFitToLabelSize()
        imageLabelWidget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        imageLabelWidget.setMinimumSize(1, 1)

        viewerLayout.addWidget(imageLabelWidget)
        layout1.addLayout(viewerLayout)

        listLayout = QVBoxLayout()
        clearClassListButtonWidget = QPushButton()
        clearClassListButtonWidget.setText("Clear")
        listLayout.addWidget(clearClassListButtonWidget)

        classListWidget = QListWidget()
        classListWidget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        classListWidget.setSelectionMode(QListWidget.SingleSelection)
        listLayout.addWidget(classListWidget)

        drawSelectedClassOnlyCheckBoxWidget = QCheckBox()
        drawSelectedClassOnlyCheckBoxWidget.setText('Draw selected class only')
        drawSelectedClassOnlyCheckBoxWidget.setChecked(False)
        self.drawSelectedClassOnly = False
        listLayout.addWidget(drawSelectedClassOnlyCheckBoxWidget)

        layout1.addLayout(listLayout)

        frameSelectionSliderWidget = QSlider(Qt.Horizontal)
        frameSelectionSliderWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        frameSelectionSliderWidget.setMinimum(0)
        indexOfFrameWidget = QLineEdit()
        indexOfFrameWidget.setText('0')
        indexOfFrameWidget.setValidator(QIntValidator(0, 0))
        indexOfFrameWidget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        frameIndexSepratorWidget = QLabel()
        frameIndexSepratorWidget.setText('/')
        numberOfFrameWidget = QLabel()
        numberOfFrameWidget.setText('0')
        layout2.addWidget(frameSelectionSliderWidget)
        layout2.addWidget(indexOfFrameWidget)
        layout2.addWidget(frameIndexSepratorWidget)
        layout2.addWidget(numberOfFrameWidget)

        openDatasetButton = QPushButton()
        serializeDatasetButton = QPushButton()
        openDatasetButton.setText('Load DataSet...')
        serializeDatasetButton.setText('Serialize To...')
        layout2.addWidget(openDatasetButton)
        layout2.addWidget(serializeDatasetButton)

        openDatasetButton.clicked.connect(self.openDataSet)
        serializeDatasetButton.clicked.connect(self.onDatasetSerializationButtonClicked)
        frameSelectionSliderWidget.valueChanged.connect(self.onSliderValueChanged)
        indexOfFrameWidget.textChanged.connect(self.indexOfFrameTextEditValueChanged)

        classListWidget.itemSelectionChanged.connect(self.onClassListWidgetSelectionChanged)

        clearClassListButtonWidget.clicked.connect(self.onClearClassListButtonClicked)
        drawSelectedClassOnlyCheckBoxWidget.stateChanged.connect(self.drawSelectedClassOnlyCheckBoxStateChanged)

        base_layout.addLayout(layout1)
        base_layout.addLayout(layout2)
        window.setLayout(base_layout)

        window.setWindowTitle('Dataset Viewer')

        self.indexOfFrame = 0
        self.numberOfFrame = 0

        self.viewerLayout = viewerLayout
        self.app = app
        self.window = window
        self.imageLabelWidget = imageLabelWidget
        self.classListWidget = classListWidget
        self.frameSelectionSliderWidget = frameSelectionSliderWidget
        self.indexOfFrameWidget = indexOfFrameWidget
        self.numberOfFrameWidget = numberOfFrameWidget
        self.openDataSetButton = openDatasetButton
        self.serializeDatasetButton = serializeDatasetButton
        self.settings = settings

        window.show()

    @staticmethod
    def fitQPixmapToLabel(label: QLabel):
        if not hasattr(label, 'sourceImage'):
            return
        p = label.sourceImage
        width = label.width()
        height = label.height()
        label.setPixmap(p.scaled(width, height, Qt.KeepAspectRatio))

    @pyqtSlot(int)
    def drawSelectedClassOnlyCheckBoxStateChanged(self, state: int):
        if state == Qt.Unchecked:
            self.drawSelectedClassOnly = False
        elif state == Qt.Checked:
            self.drawSelectedClassOnly = True

        if self.iterator:
            self.openFrame(self.indexOfFrame)


    @pyqtSlot()
    def onClearClassListButtonClicked(self):
        self.classListWidget.clearSelection()

    @pyqtSlot()
    def onClassListWidgetSelectionChanged(self):
        items = self.classListWidget.selectedItems()
        if len(items) == 0:
            self.iterator.clearClassFilter()
        else:
            item: QListWidgetItem = items[0]
            self.iterator.applyClassFilter(self.classNameToIndexMapper[item.text()])
        self.reloadDataset()

    @pyqtSlot()
    def onDatasetSerializationButtonClicked(self):
        path = browseForFileSavingPath(self.window, self.settings)
        if not path:
            return

        Dataset.Detection.serialize(self.iterator, path)
        messageBoxOk('Serialization Finished.', "Done")

    @pyqtSlot(int)
    def onSliderValueChanged(self, indexOfFrame):
        self.openFrame(indexOfFrame)
        self.indexOfFrameWidget.blockSignals(True)
        self.indexOfFrameWidget.setText(str(indexOfFrame + 1))
        self.indexOfFrameWidget.blockSignals(False)

    @pyqtSlot(str)
    def indexOfFrameTextEditValueChanged(self, value: str):
        indexOfFrame = int(value)
        indexOfFrame -= 1
        self.openFrame(indexOfFrame)
        self.frameSelectionSliderWidget.blockSignals(True)
        self.frameSelectionSliderWidget.setValue(indexOfFrame)
        self.frameSelectionSliderWidget.blockSignals(False)

    @pyqtSlot()
    def openDataSet(self):
        window = Viewer.DatasetSelectionWindow(self.settings)
        datasetIterator = window.getDatasetIterator()
        if datasetIterator is not None:
            self.loadDataset(datasetIterator)

    @staticmethod
    def isValidBoundingBox(boundingBox: typing.List):
        if len(boundingBox) == 4:
            if boundingBox[2] == 0 or boundingBox[3] == 0:
                return False
            return True
        elif len(boundingBox) == 8:
            return True
        return False

    def openFrame(self, index):
        image_path, bounding_boxes, class_indexes = self.iterator[index]
        image = QPixmap(image_path)
        painter = QPainter(image)
        fontMetrics = painter.fontMetrics()

        for bounding_box, class_index in zip(bounding_boxes, class_indexes):
            if self.drawSelectedClassOnly:
                if self.iterator.classFilter is not None:
                    if class_index != self.iterator.classFilter:
                        continue
            painter.setPen(self.classIndexColorMapper[class_index])

            bounding_box = [float(i) for i in bounding_box]
            polygon = QPolygonF((QPointF(bounding_box[0], bounding_box[1]),
                                 QPointF(bounding_box[0] + bounding_box[2], bounding_box[1]),
                                 QPointF(bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                                 QPointF(bounding_box[0], bounding_box[1] + bounding_box[3])))
            painter.drawConvexPolygon(polygon)

            class_name = self.iterator.getClassName(class_index)
            rendered_object_info_size = fontMetrics.tightBoundingRect(class_name)

            painter.fillRect(QRectF(bounding_box[0], bounding_box[1] - rendered_object_info_size.height(), rendered_object_info_size.width(), rendered_object_info_size.height()), QBrush(self.classIndexColorMapper[class_index]))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(QPointF(bounding_box[0], bounding_box[1]), class_name)

        painter.end()
        self.imageLabelWidget.sourceImage = image
        Viewer.fitQPixmapToLabel(self.imageLabelWidget)
        self.indexOfFrame = index

    def loadDataset(self, iterator: Dataset.Detection.BaseDataset):
        self.classListWidget.clear()

        classIndexes = iterator.getClasses()
        self.classNameToIndexMapper = {}
        for classIndex in classIndexes:
            self.classNameToIndexMapper[iterator.getClassName(classIndex)] = classIndex

        self.classIndexColorMapper = {}
        for classIndex in classIndexes:
            color = [random.randint(0, 255) for _ in range(3)]
            self.classIndexColorMapper[classIndex] = QColor(color[0], color[1], color[2])

        for className in self.classNameToIndexMapper.keys():
            QListWidgetItem(className, self.classListWidget)
        self.iterator = iterator
        self.reloadDataset()

    def reloadDataset(self):
        iterator = self.iterator
        self.numberOfFrame = len(iterator)

        self.frameSelectionSliderWidget.setMaximum(len(iterator) - 1)
        self.frameSelectionSliderWidget.blockSignals(True)
        self.frameSelectionSliderWidget.setValue(0)
        self.frameSelectionSliderWidget.blockSignals(False)
        self.numberOfFrameWidget.setText(str(len(iterator)))
        self.openFrame(0)
        self.indexOfFrameWidget.blockSignals(True)
        self.indexOfFrameWidget.setText('1')
        self.indexOfFrameWidget.blockSignals(False)
        self.indexOfFrameWidget.setValidator(QIntValidator(1, len(iterator)))

    def waitForExit(self):
        sys.exit(self.app.exec_())

    class DatasetSelectionWindow(QObject):
        datasetIterator: Dataset.Detection.BaseDataset

        def __init__(self, settings: QSettings):
            super(Viewer.DatasetSelectionWindow, self).__init__()

            window = QDialog()

            window.setWindowFlag(Qt.WindowType(Qt.Dialog | Qt.WindowCloseButtonHint))

            datasetList = QListWidget()
            QListWidgetItem("Cityscapes(PreferFine)", datasetList)
            QListWidgetItem("Cityscapes(FineOnly)", datasetList)
            QListWidgetItem("Cityscapes(CoarseOnly)", datasetList)
            QListWidgetItem("COCO", datasetList)
            QListWidgetItem("WebCamT", datasetList)
            QListWidgetItem("ILSVRC_DET", datasetList)
            QListWidgetItem("QingDao", datasetList)
            datasetList.setSelectionMode(QAbstractItemView.SingleSelection)
            datasetList.itemSelectionChanged.connect(self.datasetListSelectionChanged)

            okButton = QPushButton()
            okButton.setText("OK")
            okButton.setEnabled(False)
            okButton.clicked.connect(self.okButtonClicked)

            cancelButton = QPushButton()
            cancelButton.setText("Cancel")
            cancelButton.clicked.connect(self.cancelButtonClicked)

            browseSerializedDatasetButton = QPushButton()
            browseSerializedDatasetButton.setText("Serialized Dataset")
            browseSerializedDatasetButton.clicked.connect(self.browseSerializedDatasetButtonClicked)

            trainingCheckBox = QCheckBox('Training')
            trainingCheckBox.setCheckState(Qt.Checked)
            validationCheckBox = QCheckBox('Validation')
            validationCheckBox.setCheckState(Qt.Checked)
            testingCheckBox = QCheckBox('Testing')
            testingCheckBox.setCheckState(Qt.Unchecked)
            self.datasetType = Dataset.DataSplit.Training | Dataset.DataSplit.Validation
            trainingCheckBox.stateChanged.connect(self.trainingCheckBoxStateChanged)
            validationCheckBox.stateChanged.connect(self.validationCheckBoxStateChanged)
            testingCheckBox.stateChanged.connect(self.testingCheckBoxStateChanged)

            vboxLayout = QVBoxLayout()
            vboxLayout.addWidget(datasetList)

            hboxLayout = QHBoxLayout()
            hboxLayout.addWidget(trainingCheckBox)
            hboxLayout.addWidget(validationCheckBox)
            hboxLayout.addWidget(testingCheckBox)
            vboxLayout.addLayout(hboxLayout)

            hboxLayout = QHBoxLayout()
            hboxLayout.addWidget(okButton, alignment=Qt.AlignRight)
            hboxLayout.addWidget(cancelButton, alignment=Qt.AlignRight)
            hboxLayout.addWidget(browseSerializedDatasetButton, alignment=Qt.AlignRight)
            vboxLayout.addLayout(hboxLayout)

            window.setLayout(vboxLayout)

            self.okButton = okButton
            self.datasetList = datasetList
            self.window = window
            self.datasetIterator = None
            self.settings = settings

            window.show()
            window.exec_()

        @pyqtSlot(int)
        def trainingCheckBoxStateChanged(self, state: int):
            if state == Qt.Unchecked:
                self.datasetType &= ~Dataset.DataSplit.Training
            elif state == Qt.Checked:
                self.datasetType |= Dataset.DataSplit.Training

        @pyqtSlot(int)
        def validationCheckBoxStateChanged(self, state: int):
            if state == Qt.Unchecked:
                self.datasetType &= ~Dataset.DataSplit.Validation
            elif state == Qt.Checked:
                self.datasetType |= Dataset.DataSplit.Validation

        @pyqtSlot(int)
        def testingCheckBoxStateChanged(self, state: int):
            if state == Qt.Unchecked:
                self.datasetType &= ~Dataset.DataSplit.Testing
            elif state == Qt.Checked:
                self.datasetType |= Dataset.DataSplit.Testing

        @pyqtSlot()
        def browseSerializedDatasetButtonClicked(self):
            file = selectSingleFile(self.window, self.settings)
            if not file:
                return
            try:
                self.datasetIterator = Dataset.Detection.deserialize(file)
            except Exception as e:
                messageBoxError('Error', 'Failed to open dataset.\n{}'.format(e))
                return
            self.window.close()

        @pyqtSlot()
        def datasetListSelectionChanged(self):
            if len(self.datasetList.selectedItems()) == 0:
                self.okButton.setEnabled(False)
                return
            self.okButton.setEnabled(True)

        @pyqtSlot()
        def okButtonClicked(self):
            selectedItems = self.datasetList.selectedItems()
            selectedItem: QListWidgetItem = selectedItems[0]
            selectedDataset = selectedItem.text()
            path = selectSingleDirectory(self.window, self.settings)
            if not path:
                return

            if selectedDataset == 'Cityscapes(PreferFine)':
                self.datasetIterator = Dataset.Detection.CityscapesDataset(path, self.datasetType, Dataset.Detection.CityscapesDataset.AnnotationSource.PreferFine)
            elif selectedDataset == 'Cityscapes(FineOnly)':
                self.datasetIterator = Dataset.Detection.CityscapesDataset(path, self.datasetType, Dataset.Detection.CityscapesDataset.AnnotationSource.FineOnly)
            elif selectedDataset == 'Cityscapes(CoarseOnly)':
                self.datasetIterator = Dataset.Detection.CityscapesDataset(path, self.datasetType, Dataset.Detection.CityscapesDataset.AnnotationSource.CoarseOnly)
            elif selectedDataset == 'COCO':
                self.datasetIterator = Dataset.Detection.COCO2014Dataset(path, self.datasetType)
            elif selectedDataset == 'WebCamT':
                self.datasetIterator = Dataset.Detection.WebCamTDataset(path, self.datasetType)
            elif selectedDataset == 'ILSVRC_DET':
                self.datasetIterator = Dataset.Detection.ILSVRC_DET_Dataset(path, self.datasetType)
            elif selectedDataset == 'QingDao':
                self.datasetIterator = Dataset.Detection.QingDaoDataset(path)
            else:
                raise Exception
            self.window.close()

        @pyqtSlot()
        def cancelButtonClicked(self):
            self.window.close()

        def getDatasetIterator(self):
            return self.datasetIterator


from Utils.QtExceptionHook import UncaughtHook
if __name__ == '__main__':
    import sys

    debugging = False
    if 'pydevd' in sys.modules or 'pdb' in sys.modules:
        debugging = True

    if not debugging:
        qt_exception_hook = UncaughtHook()
    Viewer().waitForExit()
