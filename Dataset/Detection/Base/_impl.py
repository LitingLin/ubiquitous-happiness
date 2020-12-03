import pathlib
from PIL import Image


def _set_image_path(root_path: pathlib.Path, image_path: str):
    image_path = pathlib.Path(image_path)
    image = Image.open(image_path)
    return image.size, str(image_path.relative_to(root_path))
