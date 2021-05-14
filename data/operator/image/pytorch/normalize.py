def torch_image_normalize(image):
    return image.div(255.0).clamp(0.0, 1.0)
