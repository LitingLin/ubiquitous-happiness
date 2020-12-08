import os

_current_path = os.path.abspath(os.path.join(__file__, os.pardir))
_build_path = os.path.join(_current_path, 'cmake-build')


def build_extension_cmake(cuda_path=None, verbose=False):
    import shutil
    import sys
    import os

    original_wd = os.getcwd()
    install_path = _current_path
    source_path = os.path.abspath(_current_path)

    if cuda_path is None:
        from torch.utils.cpp_extension import _find_cuda_home
        cuda_path = _find_cuda_home()

    def is_anaconda_dist():
        return os.path.exists(os.path.join(sys.prefix, 'conda-meta'))

    if sys.platform == 'win32':
        python_root_path = os.path.abspath(os.path.join(sys.executable, os.pardir))
    else:
        python_root_path = os.path.abspath(os.path.join(sys.executable, os.pardir, os.pardir))

    try:
        shutil.which('cmake')
        if sys.platform == 'win32':
            shutil.which('ninja')

            if is_anaconda_dist():
                def apply_envs(envs):
                    for key, value in envs.items():
                        os.environ[key] = value

                def get_vc_version():
                    conda_meta_path = os.path.join(sys.prefix, 'conda-meta')
                    conda_packages = os.listdir(conda_meta_path)
                    conda_packages = [conda_package for conda_package in conda_packages if
                                      conda_package.startswith('vc') and conda_package.endswith('.json')]
                    assert len(conda_packages) == 1
                    vc_package_path = os.path.join(conda_meta_path, conda_packages[0])
                    import json
                    with open(vc_package_path) as fid:
                        vc_package_meta = json.load(fid)
                        return vc_package_meta['version']

                import distutils._msvccompiler
                apply_envs(distutils._msvccompiler._get_vc_env('x64 -vcvars_ver={}'.format(get_vc_version())))

        if not os.path.exists(_build_path):
            os.mkdir(_build_path)

        os.chdir(_build_path)
        import subprocess
        if sys.platform == 'win32':
            cmake_command = ['cmake', source_path, '-DCMAKE_BUILD_TYPE=RelWithDebInfo', '-G', 'Ninja',
                             '-DCMAKE_CUDA_COMPILER={}'.format(os.path.join(cuda_path, 'bin', 'nvcc.exe')),
                             '-DPython3_ROOT_DIR={}'.format(python_root_path),
                             '-DCMAKE_INSTALL_PREFIX={}'.format(install_path)]
            if is_anaconda_dist():
                cmake_command.append('-DCMAKE_PREFIX_PATH={}'.format(os.path.join(python_root_path, 'Library')))

            subprocess.check_call(cmake_command)

            build_command = ['ninja']
            if verbose:
                build_command.append('-v')
            subprocess.check_call(build_command)
            subprocess.check_call(['ninja', 'install'])
        else:
            cmake_command = ['cmake', source_path, '-DCMAKE_BUILD_TYPE=RelWithDebInfo',
                             '-DCMAKE_CUDA_COMPILER={}'.format(os.path.join(cuda_path, 'bin', 'nvcc')),
                             '-DPython3_ROOT_DIR={}'.format(python_root_path),
                             '-DCMAKE_INSTALL_PREFIX={}'.format(install_path)]
            if is_anaconda_dist():
                cmake_command.append('-DCMAKE_PREFIX_PATH={}'.format(python_root_path))
            subprocess.check_call(cmake_command, cwd=_build_path)
            import multiprocessing
            make_kwargs = {}
            if verbose:
                make_kwargs = {'env': {'VERBOSE': '1'}}
            subprocess.check_call(['make', '-j{}'.format(multiprocessing.cpu_count()), 'VERBOSE=1'], cwd=_build_path)
            subprocess.check_call(['make', 'install'], cwd=_build_path)
    finally:
        os.chdir(original_wd)


def clean_up():
    import shutil
    try:
        shutil.rmtree(_build_path)
    except:
        pass
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--cuda_path', type=str, default=None)
    args = parser.parse_args()
    if args.clean:
        clean_up()
    else:
        build_extension_cmake(args.cuda_path, args.verbose)
