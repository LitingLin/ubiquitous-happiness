import threading
from data.types.pixel_coordinate_system import PixelCoordinateSystem


context = threading.local()
context.pixel_coordinate_system = PixelCoordinateSystem.AlignCorner


def get_pixel_coordinate_system():
    return context.pixel_coordinate_system


class set_pixel_coordinate_system:
    def __init__(self, pixel_coordinate_system: PixelCoordinateSystem):
        self.prev = context.pixel_coordinate_system
        context.pixel_coordinate_system = pixel_coordinate_system

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        context.pixel_coordinate_system = self.prev


class use_half_pixel_center_coordinate_system:
    def __init__(self):
        self.prev = context.pixel_coordinate_system
        context.pixel_coordinate_system = PixelCoordinateSystem.HalfPixelCenter

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        context.pixel_coordinate_system = self.prev


class use_align_corner_coordinate_system:
    def __init__(self):
        self.prev = context.pixel_coordinate_system
        context.pixel_coordinate_system = PixelCoordinateSystem.AlignCorner

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        context.pixel_coordinate_system = self.prev


def is_using_half_pixel_center_coordinate_system():
    return context.pixel_coordinate_system == PixelCoordinateSystem.HalfPixelCenter


def is_using_align_corner_coordinate_system():
    return context.pixel_coordinate_system == PixelCoordinateSystem.AlignCorner

