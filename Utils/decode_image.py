from native_extension import ImageDecoder

_image_decoder = ImageDecoder()


def decode_image_file(path: str):
    image = _image_decoder.decode(path)
    _image_decoder.close()
    return image
