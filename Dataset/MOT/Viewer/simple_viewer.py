from Viewer.viewer import Viewer
from PIL import Image
from typing import Dict
from Dataset.MOT.Base.dataset import MultipleObjectTrackingDataset


def get_label_text(label: str, additional_attributes: Dict):
    string = label
    if additional_attributes is not None:
        for name, value in additional_attributes.items():
            string += ' {}={}'.format(name, value)
    return string


def run_simple_viewer(dataset: MultipleObjectTrackingDataset):
    visitor = dataset.getVisitor()

    viewer = Viewer()

    for sequence in visitor:
        for frame in sequence:
            viewer.clear()
            image = Image.open(frame.getImagePath())
            viewer.drawImage(image)
            for object in frame:
                viewer.drawBoundingBoxAndLabel(object.getBoundingBox(),
                                               get_label_text(object.getClassLabel(), object.getAttributes()))
            viewer.update()
            viewer.pause(0.001)
