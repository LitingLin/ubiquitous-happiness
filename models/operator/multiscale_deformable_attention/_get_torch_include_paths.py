from torch.utils.cpp_extension import include_paths

if __name__ == '__main__':
    paths = [path.replace('\\', '/') for path in include_paths(False)]
    print(';'.join(paths), end='')
    #print(r'C:\Users\liting\miniconda3\envs\env-1\lib\site-packages\torch\include;C:\Users\liting\miniconda3\envs\env-1\lib\site-packages\torch\include\torch\csrc\api\include;C:\Users\liting\miniconda3\envs\env-1\lib\site-packages\torch\include\TH;C:\Users\liting\miniconda3\envs\env-1\lib\site-packages\torch\include\THC')