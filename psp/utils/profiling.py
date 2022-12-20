import contextlib
import time


@contextlib.contextmanager
def profile(name: str):
    t0 = time.time()
    yield
    t1 = time.time()
    print(f'Executed "{name}" in {t1 - t0:.3f}s')
