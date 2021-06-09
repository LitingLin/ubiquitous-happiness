import time
import matplotlib
import matplotlib.pyplot as plt
import gc
import torch
import torch.cuda
import os
import psutil
import tqdm


def _draw(x, y, x_label, y_label, title):
    fig, ax = plt.subplots()

    ax.plot(x, y)

    ax.set(xlabel=x_label,
           ylabel=y_label,
           title=title)

    plt.draw()
    plt.show()


def _get_memory_usage(device: torch.device):
    MB = 1024 * 1024
    if 'cpu' in device.type:
        process = psutil.Process(os.getpid())
        return (process.memory_info().rss) / MB
    elif 'cuda' in device.type:
        return torch.cuda.max_memory_allocated() / MB


def draw_model_time_and_memory_consumption_respect_to_input_size(model, test_range, device: torch.device):
    x = []
    times = []
    memory_consumptions = []
    input_ = torch.zeros((1, 3, 128, 128), dtype=torch.float, device=device)
    with torch.no_grad():
        model(input_)
    for size in tqdm.tqdm(test_range):
        x.append(size)
        input_ = torch.zeros((1, 3, size, size), dtype=torch.float, device=device)
        gc.collect()
        if 'cuda' in device.type:
            torch.cuda.empty_cache()
        start = time.perf_counter()
        with torch.no_grad():
            model(input_)
        times.append(time.perf_counter() - start)
        memory_consumptions.append(_get_memory_usage(device))
    _draw(x, times, 'size', 's', 'time consumption')
    _draw(x, memory_consumptions, 'size', 'MB', 'memory consumption')


if __name__ == '__main__':
    from models.backbone.swint.swin_transformer_old import build_swin_base_patch4_window7_224
    network = build_swin_base_patch4_window7_224()
    cuda=True
    if cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if cuda:
        network.to(device)
        network.eval()
    draw_model_time_and_memory_consumption_respect_to_input_size(network, range(16, 768, 16), device)
