import concurrent.futures

from evaluation.SOT.runner import visualize_tracking_results

def _run_video_generation(sequence):
    visualize_tracking_results(['TransT-Swin', 'TransT-Swin-WH'], ['C:\\Users\\liting\\Documents\\TransT-Swin\\result', 'C:\\test\\ope\\transt-swin-bbox-size-limit-in-feat-space\\result'], sequence,
                       f'I:\\com\\{sequence}.mp4')

if __name__ == '__main__':

    sequences = ['skateboard-3']
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    for sequence in sequences:
        # thread_pool.submit(_run_video_generation, sequence)
        _run_video_generation(sequence)
