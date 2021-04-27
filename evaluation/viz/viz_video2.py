if __name__ == '__main__':
    from evaluation.SOT.runner import visualize_sequence

    sequences = ['skateboard-3']

    for sequence in sequences:
        visualize_sequence('C:\\Users\\liting\\Documents\\TransT-Swin\\result', sequence, f'C:\\Users\\liting\\Documents\\{sequence}.mp4')
