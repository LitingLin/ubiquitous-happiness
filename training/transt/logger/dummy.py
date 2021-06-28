class DummyLogger:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self

    def log_train(self, epoch, forward_stats, backward_stats):
        pass

    def log_test(self, epoch, summary):
        pass

    def watch(self, model):
        pass
