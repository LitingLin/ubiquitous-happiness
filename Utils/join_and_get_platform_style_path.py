import os


def join_and_get_platform_style_path(*args):
    path = os.path.join(*args)
    if os.name == 'nt':
        path = path.replace('/', '\\')
    else:
        path = path.replace('\\', '/')
    return path
