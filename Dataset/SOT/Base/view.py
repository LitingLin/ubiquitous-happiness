from typing import List, Union
from Dataset.SOT.Base.dataset import SingleObjectTrackingDataset
from Dataset.SOT.Base.sequence import SingleObjectTrackingDatasetSequenceViewer, SingleObjectTrackingDatasetSequence


class SingleObjectTrackingDatasetView:
    dataset: SingleObjectTrackingDataset
    sequences: List[SingleObjectTrackingDatasetSequence]

    def __init__(self, dataset: SingleObjectTrackingDataset):
        self.dataset = dataset
        self.sequences = dataset.sequences

    def getName(self):
        return self.dataset.name

    def getCategoryNameList(self):
        return self.dataset.category_names

    def getNumberOfCategories(self):
        return len(self.dataset.category_names)

    def getCategoryName(self, id_: int):
        return self.dataset.category_names[id_]

    def applyClassFilter(self, category: Union[int, str]):
        if isinstance(category, str):
            category = self.dataset.category_name_id_mapper[category]

        self.sequences = []
        for sequence in self.dataset.sequences:# SingleObjectTrackingDatasetSequence
            if sequence.category_id == category:
                self.sequences.append(sequence)

    def cleaFilter(self):
        self.sequences = self.dataset.sequences

    def __getitem__(self, index: int):
        return SingleObjectTrackingDatasetSequenceViewer(self.dataset, self.sequences[index])

    def __len__(self):
        return len(self.sequences)

    def applyIndicesFilter(self, indices: List[int]):
        self.sequences = [self.dataset.sequences[index] for index in indices]
