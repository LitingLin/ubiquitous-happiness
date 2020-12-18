import os
_current_path = os.path.dirname(__file__)
_build_path = os.path.join(_current_path, 'cmake-build')


def _to_unix_style_path(path: str):
    return path.replace('\\', '/')


def clean_up():
    import shutil
    try:
        shutil.rmtree(_build_path)
    except:
        pass


def build_extension_cmake(verbose=False):
    import shutil
    import sys
    import os

    current_path = os.path.abspath(os.path.join(__file__, os.pardir))
    build_path = os.path.join(current_path, 'cmake-build')
    install_path = current_path
    source_path = os.path.abspath(os.path.join(current_path, 'src'))

    def is_anaconda_dist():
        return os.path.exists(os.path.join(sys.prefix, 'conda-meta'))

    if sys.platform == 'win32':
        python_root_path = os.path.dirname(sys.executable)
        assert os.path.exists(os.path.join(python_root_path, 'python.exe'))
    else:
        python_root_path = os.path.abspath(os.path.join(os.path.dirname(sys.executable), os.pardir))
        assert os.path.exists(os.path.join(python_root_path, 'bin', 'python'))

    cmake_parameters = []
    cmake_parameters.append('-DPython3_ROOT_DIR={}'.format(_to_unix_style_path(python_root_path)))
    cmake_parameters.append('-DPython3_FIND_STRATEGY=LOCATION')
    cmake_parameters.append('-DCMAKE_INSTALL_PREFIX={}'.format(_to_unix_style_path(install_path)))
    if is_anaconda_dist():
        if sys.platform == 'win32':
            cmake_parameters.append('-DCMAKE_PREFIX_PATH={}'.format(_to_unix_style_path(os.path.join(python_root_path, 'Library'))))
        else:
            cmake_parameters.append('-DCMAKE_PREFIX_PATH={}'.format(python_root_path))

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

    if not os.path.exists(build_path):
        os.mkdir(build_path)

    os.chdir(build_path)
    import subprocess
    if sys.platform == 'win32':
        cmake_command = ['cmake', '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=RelWithDebInfo']
        cmake_command.extend(cmake_parameters)
        cmake_command.append(_to_unix_style_path(source_path))
        subprocess.check_call(cmake_command, cwd=_build_path)

        build_command = ['ninja']
        if verbose:
            build_command.append('-v')
        subprocess.check_call(build_command, cwd=_build_path)
        subprocess.check_call(['ninja', 'install'], cwd=_build_path)
    else:
        cmake_command = ['cmake', '-DCMAKE_BUILD_TYPE=RelWithDebInfo']
        cmake_command.extend(cmake_parameters)
        cmake_command.append(source_path)
        subprocess.check_call(cmake_command, cwd=_build_path)
        import multiprocessing
        build_command = ['make', '-j{}'.format(multiprocessing.cpu_count())]
        if verbose:
            build_command.append('VERBOSE=1')
        subprocess.check_call(build_command, cwd=_build_path)
        subprocess.check_call(['make', 'install'], cwd=_build_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    if args.clean:
        clean_up()
    else:
        build_extension_cmake(args.verbose)
