from .stack import VOTStack
from Miscellaneous.yaml_ops import yaml_dump
from Miscellaneous.repo_root import get_repository_root
from Dataset.Config.path import get_path_from_config
import os


def prepare_vot_workspace(workspace_path: str, tracker_name: str, tracker_launch_command: str, version: VOTStack):
    if version == VOTStack.vot2020:
        dataset_path = get_path_from_config('VOT2020_PATH')
    elif version == VOTStack.vot2021:
        dataset_path = get_path_from_config('VOT2021_PATH')
    else:
        raise RuntimeError(f"Unsupported VOT version {version}")

    os.makedirs(workspace_path, exist_ok=True)
    os.mkdir(os.path.join(workspace_path, 'results'))
    os.symlink(dataset_path, os.path.join(workspace_path, 'sequences'), target_is_directory=True)

    yaml_dump({'registry': ['./trackers.ini'], 'stack': version.name}, os.path.join(workspace_path, 'config.yaml'))

    with open(os.path.join(workspace_path, 'trackers.ini'), 'w', newline='\n') as f:
        f.write(f'[{tracker_name}]\n')
        f.write(f'label = {tracker_name}\n')
        f.write('protocol = traxpython\n')
        f.write(f'command = {tracker_launch_command}\n')
        f.write(f'paths = {get_repository_root()}\n')
