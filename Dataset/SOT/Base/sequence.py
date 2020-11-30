from __future__ import annotations
from typing import List, Dict
from .dataset import SingleObjectTrackingDataset
from .frame import SingleObjectTrackingDatasetFrameViewer


class SingleObjectTrackingDatasetSequence:
    frames: List
    category_id: int
    name: str
    attributes: Dict

    def __init__(self):
        self.frames = []
        self.category_id = None
        self.name = None
        self.fps = None
        self.attributes = {}


class SingleObjectTrackingDatasetSequenceViewer:
    sequence: SingleObjectTrackingDatasetSequence
    dataset: SingleObjectTrackingDataset

    def __init__(self, dataset: SingleObjectTrackingDataset, sequence: SingleObjectTrackingDatasetSequence):
        self.sequence = sequence
        self.dataset = dataset

    def getName(self):
        return self.sequence.name

    def getCategoryId(self):
        return self.sequence.category_id

    def getCategoryName(self):
        return self.dataset.category_names[self.sequence.category_id]

    def getFrame(self, index: int):
        return SingleObjectTrackingDatasetFrameViewer(self.dataset, self.sequence, self.sequence.frames[index])

    def getAttribute(self, name: str):
        return self.sequence.attributes[name]

    def hasAttribute(self, name: str):
        return name in self.sequence.attributes

    def __getitem__(self, index: int):
        return self.getFrame(index)

    def __len__(self):
        return len(self.sequence.frames)

    def hasAttributeFPS(self):
        return self.dataset.hasAttibuteFPS()

    def getFPS(self):
        return self.sequence.fps

    def hasAttributeCategory(self):
        return self.dataset.hasAttributeCategory()
