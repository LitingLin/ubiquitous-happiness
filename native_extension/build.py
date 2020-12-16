def build_extension_cmake(argv=()):
    import shutil
    import sys
    import os

    original_wd = os.getcwd()
    current_path = os.path.abspath(os.path.join(__file__, os.pardir))
    build_path = os.path.join(current_path, 'cmake-build')
    install_path = current_path
    source_path = os.path.abspath(os.path.join(current_path, 'src'))

    if 'clean' in argv:
        import shutil
        try:
            shutil.rmtree(build_path)
        except:
            pass
        return

    def is_anaconda_dist():
        return os.path.exists(os.path.join(sys.prefix, 'conda-meta'))

    if sys.platform == 'win32':
        python_root_path = os.path.dirname(sys.executable)
        assert os.path.exists(os.path.join(python_root_path, 'python.exe'))
    else:
        python_root_path = os.path.abspath(os.path.join(os.path.dirname(sys.executable), os.pardir))
        assert os.path.exists(os.path.join(python_root_path, 'python'))

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

        if not os.path.exists(build_path):
            os.mkdir(build_path)

        os.chdir(build_path)
        import subprocess
        if sys.platform == 'win32':
            cmake_command = ['cmake', source_path, '-DCMAKE_BUILD_TYPE=RelWithDebInfo', '-G', 'Ninja',
                             '-DPython3_ROOT_DIR={}'.format(python_root_path),
                             '-DCMAKE_INSTALL_PREFIX={}'.format(install_path)]
            if is_anaconda_dist():
                cmake_command.append('-DCMAKE_PREFIX_PATH={}'.format(os.path.join(python_root_path, 'Library')))

            subprocess.check_call(cmake_command)

            build_command = ['ninja']
            if '-v' in argv:
                build_command.append('-v')
            subprocess.check_call(build_command)
            subprocess.check_call(['ninja', 'install'])
        else:
            cmake_command = ['cmake', source_path, '-DCMAKE_BUILD_TYPE=RelWithDebInfo',
                             '-DPython3_ROOT_DIR={}'.format(python_root_path),
                             '-DCMAKE_INSTALL_PREFIX={}'.format(install_path)]
            if is_anaconda_dist():
                cmake_command.append('-DCMAKE_PREFIX_PATH={}'.format(python_root_path))
            subprocess.check_call(cmake_command)
            import multiprocessing
            make_kwargs = {}
            if '-v' in argv:
                make_kwargs = {'env': {'VERBOSE': '1'}}
            subprocess.check_call(['make', '-j{}'.format(multiprocessing.cpu_count())], **make_kwargs)
            subprocess.check_call(['make', 'install'])
    finally:
        os.chdir(original_wd)


if __name__ == '__main__':
    import sys
    build_extension_cmake(sys.argv)
