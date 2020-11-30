import pathlib
from PIL import Image


def _set_image_path(root_path: pathlib.Path, image_path: str):
    image_path = pathlib.Path(image_path)
    image = Image.open(image_path)
    return image.size, str(image_path.relative_to(root_path))


def _get_or_allocate_category_id(category_name, category_names, category_name_id_mapper):
    if category_name not in category_name_id_mapper:
        id_ = len(category_names)
        category_names.append(category_name)
        category_name_id_mapper[category_name] = id_
    else:
        id_ = category_name_id_mapper[category_name]
    return id_