
class StarkLabelGenerator:
    def __init__(self, search_feat_size, search_region_size):
        self.search_feat_size = search_feat_size
        self.search_region_size = search_region_size

    def __call__(self, bbox, is_positive):
        if is_positive:
            return label_generation(bbox, self.search_feat_size, self.search_region_size)
        else:
            return negative_label_generation(self.search_feat_size)
