from typing import List, Tuple, Dict
import os


class BaseDataset:
    classNamesMapper: Dict[int, str]
    classIndexer: Dict[int, List[int]]
    image_paths: List[str]
    bounding_boxes: List[List[Tuple[float]]]
    classes: List[List[int]]

    def __init__(self, root_dir: str, class_names: Dict[int, str]):
        self.root_dir = root_dir
        self.classNamesMapper = class_names
        self.classIndexer = {}
        self.image_paths = []
        self.bounding_boxes = []
        self.classes = []
        self.classFilter = None

    def getNumberOfClasses(self):
        return len(self.classIndexer)

    def setClassNames(self, mapper: Dict[int, str]):
        self.classNamesMapper = mapper

    def getClassName(self, index: int):
        return self.classNamesMapper[index]

    def getClasses(self):
        return list(self.classIndexer.keys())

    def getRecordImagePath(self, index: int, classIndex:int=None):
        if classIndex is not None:
            index = self.classIndexer[classIndex][index]
        return os.path.join(self.root_dir, self.image_paths[index])

    def getRecordBoundingBoxes(self, index: int, classIndex:int=None):
        if classIndex is not None:
            index = self.classIndexer[classIndex][index]
        return self.bounding_boxes[index]

    def getRecordClasses(self, index: int, classIndex:int=None):
        if classIndex is not None:
            index = self.classIndexer[classIndex][index]
        return self.classes[index]

    def getNumberOfRecords(self, classIndex:int=None):
        if classIndex is not None:
            return len(self.classIndexer[classIndex])
        return len(self.image_paths)

    def addRecord(self, relative_path: str, bounding_boxes: List[Tuple[float]], classes: List[int]):
        currentRecordIndex = len(self.image_paths)

        self.image_paths.append(relative_path)
        self.bounding_boxes.append(bounding_boxes)
        self.classes.append(classes)

        for classIndex in classes:
            if classIndex not in self.classIndexer:
                self.classIndexer[classIndex] = []
            if len(self.classIndexer[classIndex]) > 0 and self.classIndexer[classIndex][-1] == currentRecordIndex:
                continue
            self.classIndexer[classIndex].append(currentRecordIndex)

    # stateful api
    def applyClassFilter(self, index: int):
        self.classFilter = index

    def clearClassFilter(self):
        self.classFilter = None

    def __len__(self):
        if self.classFilter is None:
            return len(self.image_paths)
        return len(self.classIndexer[self.classFilter])

    def __getitem__(self, index: int):
        return self.getRecordImagePath(index, self.classFilter), self.getRecordBoundingBoxes(index, self.classFilter), self.getRecordClasses(index, self.classFilter)


class MergeableDataset(BaseDataset):
    def __init__(self, class_names: Dict[int, str]):
        self.root_dir = ""
        self.classNamesMapper = class_names
        self.classIndexer = {}
        self.image_paths = []
        self.bounding_boxes = []
        self.classes = []
        self.classFilter = None

    def merge(self, dataset: BaseDataset, mapper: Dict[int, int]):
        for image_path, boundingBoxes, classes in dataset:
            new_classes = [mapper[class_] for class_ in classes if class_ in mapper]
            new_boundingBoxes = [boundingBox for boundingBox, class_ in zip(boundingBoxes, classes) if class_ in mapper]
            if len(new_classes) == 0:
                continue
            self.addRecord(image_path, new_boundingBoxes, new_classes)
