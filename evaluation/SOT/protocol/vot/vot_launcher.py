import sys
import os
from vot.utilities.cli import main


def launch_vot_evaluation(workspace_path: str, name: str):
    old_sys_argv = sys.argv
    old_wd = os.getcwd()
    evaluation_args = ['evaluate', name]
    sys.argv = [old_sys_argv[0]] + evaluation_args
    os.chdir(workspace_path)
    try:
        main()
    finally:
        sys.argv = old_sys_argv
        os.chdir(old_wd)


def launch_vot_analysis(workspace_path: str):
    old_sys_argv = sys.argv
    old_wd = os.getcwd()
    evaluation_args = ['analysis']
    sys.argv = [old_sys_argv[0]] + evaluation_args
    os.chdir(workspace_path)
    try:
        main()
    finally:
        sys.argv = old_sys_argv
        os.chdir(old_wd)
