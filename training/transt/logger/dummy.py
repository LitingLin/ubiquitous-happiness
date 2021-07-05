class DummyLogger:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def log_train(self, epoch, forward_stats, backward_stats):
        pass

    def log_test(self, epoch, summary):
        pass

    def watch(self, model):
        pass
