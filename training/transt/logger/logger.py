class Logger:
    def __init__(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
