import sys
import os
from vot.utilities.cli import main


def launch_vot_evaluation(workspace_path: str, name: str):
    print(os.environ)
    print(os.environ['PATH'])
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
    # subprocess.check_call(f"vot evaluate {name}", shell=True, cwd=workspace_path)
