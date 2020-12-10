import torch
if __name__ == '__main__':
    config = torch.__config__.show()
    configs = config.split('\n')
    archs = set()
    for conf in configs:
        if 'NVCC arch' in conf:
            ss = conf.split(';')
            for s in ss:
                s = s.strip()
                if s.startswith('arch='):
                    cs = s[5:].split(',')
                    for c in cs:
                        v = c.split('_')
                        archs.add(int(v[1]))
    archs = [str(arch) for arch in archs]
    print(';'.join(archs), end='')
