import cProfile
import pstats
import io
from contextlib import contextmanager


@contextmanager
def profile(toprint=True):
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        if toprint:
            print(s.getvalue())
