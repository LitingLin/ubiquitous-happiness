class SimpleAdHocManipulator:
    def __init__(self, dataset_attributes: dict):
        self.dataset_attributes = dataset_attributes

    def set_attribute(self, name: str, value):
        self.dataset_attributes[name] = value

    def get_attribute(self, name: str):
        return self.dataset_attributes[name]

    def has_attribute(self, name: str):
        return name in self.dataset_attributes

    def list_attribute_keys(self):
        return self.dataset_attributes.keys()
