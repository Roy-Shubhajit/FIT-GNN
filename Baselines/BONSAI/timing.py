import time


class Timer:
    def __init__(self, context_string):
        self.context_string = context_string
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc_val, exc_trb):
        if exc_type is None:
            self.dur = time.perf_counter() - self.start_time
            print(f"{self.context_string}: {self.dur:.6f}s")
        else:
            print(f"An error occured, can't time\nError: {exc_val}")
            import traceback
            traceback.print_tb(exc_trb)
            return False
