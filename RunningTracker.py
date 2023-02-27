import time


class RunningTracker:

    def __init__(self, func):
        self.func = func
        self.result = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.result = self.func()
            self.end_time = time.time()
        else:
            return False
