from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QLabel, \
    QDoubleSpinBox, QSpacerItem
from PyQt5.QtCore import Qt, pyqtSlot, QObject
from PyQt5.QtGui import QResizeEvent


class _ScalableImageViewerContext(QObject):
    def __init__(self, label, x_offset_spinbox, y_offset_spinbox, scale_spinbox, reset_button, parent, relative=True):
        super(_ScalableImageViewerContext, self).__init__(parent)
        self.label = label
        self.x_offset_spinbox = x_offset_spinbox
        self.y_offset_spinbox = y_offset_spinbox
        self.scale_spinbox = scale_spinbox
        self.reset_button = reset_button
        x_offset_spinbox.valueChanged.connect(self._on_x_offset_changed)
        y_offset_spinbox.valueChanged.connect(self._on_y_offset_changed)
        scale_spinbox.valueChanged.connect(self._on_scale_changed)
        reset_button.clicked.connect(self._on_reset_button_clicked)
        label._image_viewer_context = self
        self.render = None

        self._default_scale = 1.
        self._default_x_offset = 0.
        self._default_y_offset = 0.

        self._scale = self._default_scale
        self._x_offset = self._default_x_offset
        self._y_offset = self._default_y_offset

        self.relative = relative

    def set_image(self, image):
        from Viewer.canvas.align_corner.simple_painter import SimplePainter
        self.render = SimplePainter.create_from_tf_image(image)
        self.update()

    def set_painter(self, painter):
        self.render = painter
        self.update()

    def update(self):
        if self.render is None:
            self.label.clear()
            return
        if self.relative:
            canvas_w, canvas_h = self.render.get_canvas_size()
            base_scale = min((self.label.width() - 1) / (canvas_w - 1), (self.label.height() - 1) / (canvas_h - 1))
            scale = self._scale * base_scale
            x_offset = self._x_offset * base_scale
            y_offset = self._y_offset * base_scale
        else:
            scale = self._scale
            x_offset = self._x_offset
            y_offset = self._y_offset
        image = self.render.render((self.label.width(), self.label.height()), (scale, scale), translation_target_center=(x_offset, y_offset), with_qpixmap=True)
        self.label.setPixmap(image)

    @pyqtSlot(float)
    def _on_scale_changed(self, scale):
        self._scale = scale
        self.update()

    @pyqtSlot(float)
    def _on_x_offset_changed(self, x_offset):
        self._x_offset = x_offset
        self.update()

    @pyqtSlot(float)
    def _on_y_offset_changed(self, y_offset):
        self._y_offset = y_offset
        self.update()

    @pyqtSlot(bool)
    def _on_reset_button_clicked(self, _):
        self.reset()

    def set_scale(self, value):
        self.scale_spinbox.setValue(value)

    def set_x_offset(self, value):
        self.x_offset_spinbox.setValue(value)

    def set_y_offset(self, value):
        self.y_offset_spinbox.setValue(value)

    def set(self, scale, x_offset, y_offset):
        self._scale = scale
        self._x_offset = x_offset
        self._y_offset = y_offset
        self.update()
        self.x_offset_spinbox.blockSignals(True)
        self.x_offset_spinbox.setValue(x_offset)
        self.x_offset_spinbox.blockSignals(False)
        self.y_offset_spinbox.blockSignals(True)
        self.y_offset_spinbox.setValue(y_offset)
        self.y_offset_spinbox.blockSignals(False)
        self.scale_spinbox.blockSignals(True)
        self.scale_spinbox.setValue(scale)
        self.scale_spinbox.blockSignals(False)
        self.update()

    def set_default(self, scale, x_offset, y_offset):
        self._default_scale = scale
        self._default_x_offset = x_offset
        self._default_y_offset = y_offset

    def set_default_scale(self, scale):
        self._default_scale = scale

    def set_default_x_offset(self, x_offset):
        self._default_x_offset = x_offset

    def set_default_y_offset(self, y_offset):
        self._default_y_offset = y_offset

    def reset(self):
        self.set(self._default_scale, self._default_x_offset, self._default_y_offset)


class _CanvasLabel(QLabel):
    def __init__(self, *args):
        super(_CanvasLabel, self).__init__(*args)
        self._image_viewer_context = None

    def resizeEvent(self, qResizeEvent: QResizeEvent):
        super().resizeEvent(qResizeEvent)
        if self._image_viewer_context is not None:
            self._image_viewer_context.update()


def construct_simple_image_viewer_on_qt_layout(layout):
    image_label = _CanvasLabel()
    image_label.setMinimumSize(1, 1)
    image_label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

    x_offset_spinbox = QDoubleSpinBox()
    y_offset_spinbox = QDoubleSpinBox()

    x_offset_spinbox.setValue(0)
    y_offset_spinbox.setValue(0)

    x_offset_spinbox.setMinimum(-65536)
    x_offset_spinbox.setMaximum(65536)

    y_offset_spinbox.setMinimum(-65536)
    y_offset_spinbox.setMaximum(65536)

    scale_spinbox = QDoubleSpinBox()
    scale_spinbox.setValue(1.)
    scale_spinbox.setSingleStep(0.1)

    reset_button = QPushButton()
    reset_button.setText('Reset')

    vlayout = QVBoxLayout()
    actor_layout = QHBoxLayout()

    spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
    actor_layout.addSpacerItem(spacer)

    x_offset_label = QLabel()
    x_offset_label.setText('x:')
    actor_layout.addWidget(x_offset_label)
    actor_layout.addWidget(x_offset_spinbox)
    y_offset_label = QLabel()
    y_offset_label.setText('y:')
    actor_layout.addWidget(y_offset_label)
    actor_layout.addWidget(y_offset_spinbox)
    scale_label = QLabel()
    scale_label.setText('scale:')
    actor_layout.addWidget(scale_label)
    actor_layout.addWidget(scale_spinbox)

    actor_layout.addWidget(reset_button)

    vlayout.addWidget(image_label)
    vlayout.addLayout(actor_layout)

    layout.addLayout(vlayout)

    return _ScalableImageViewerContext(image_label, x_offset_spinbox, y_offset_spinbox, scale_spinbox, reset_button, layout)


class SimpleViewer:
    def __init__(self, argv=[]):
        app = QApplication(argv)

        window = QDialog()
        #window.setWindowState(Qt.WindowMaximized)
        window.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        window.setWindowTitle('Viewer')

        layout = QVBoxLayout()
        window.setLayout(layout)
        self.main_layout = layout
        self.app = app
        self.window = window

    def addImage(self):
        from data.operator.image.tf.decoder import tf_decode_image
        from Viewer.canvas.align_corner.simple_painter import SimplePainter
        image_viewer_widget = construct_simple_image_viewer_on_qt_layout(self.main_layout)
        image = tf_decode_image("K:\\dataset\\coco\\images\\train2014\\COCO_train2014_000000000009.jpg")
        painter = SimplePainter.create_from_tf_image(image)
        h, w, c = image.shape
        painter.draw_bounding_box([0, 0, w - 1, h - 1])
        painter.draw_bounding_box_with_label([5,5,10,10], 'a')
        painter.draw_bounding_box([0,0,3,1])
        painter.draw_bounding_box([0,0,2,1])
        painter.draw_bounding_box([0, 0, 1, 1])
        #painter.draw_bounding_box_with_label([2,2,3,3], 'a')
        image_viewer_widget.set_painter(painter)

    def setWindowTitle(self, title: str):
        self.window.setWindowTitle(title)

    def runEventLoop(self):
        self.window.show()
        return self.app.exec_()

    def close(self):
        self.window.close()

if __name__ == '__main__':
    v=SimpleViewer()
    v.addImage()
    v.runEventLoop()
