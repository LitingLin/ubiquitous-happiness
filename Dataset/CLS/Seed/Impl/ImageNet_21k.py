from Dataset.CLS.Constructor.base import ImageClassificationDatasetConstructor
import os
import numpy as np


def construct_ImageNet_21k(constructor: ImageClassificationDatasetConstructor, seed):
    root_path = seed.root_path
    classes = os.listdir(root_path)
    classes = [class_ for class_ in classes if os.path.isdir(os.path.join(root_path, class_))]
    classes.sort()

    wordnet_id_file_path = os.path.join(os.path.dirname(__file__), 'imagenet21k_wordnet_ids.txt')
    wordnet_ids = np.genfromtxt(wordnet_id_file_path, dtype='str')
    category_id_name_map = {index_: str(wordnet_id) for index_, wordnet_id in enumerate(wordnet_ids)}

    for class_id, class_name in category_id_name_map.items():
        class_path = os.path.join(root_path, class_name)
        if not os.path.isdir(class_path):
            continue
        images = os.listdir(class_path)
        assert len(images) != 0
        images.sort()
        for image in images:
            image_path = os.path.join(class_path, image)
            with constructor.new_image() as image_constructor:
                image_constructor.set_path(image_path)
                image_constructor.set_category_id(class_id)
