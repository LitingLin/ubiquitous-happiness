from Dataset.MOT.Base.constructor import MultipleObjectTrackingDatasetConstructor



class MultipleObjectTrackingSiamDatasetConstructor(MultipleObjectTrackingDatasetConstructor):
    def addExemplarImage(self, size: int, image_path: str):
