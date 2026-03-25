import torch
import time
from functools import wraps


def print_vram_stats():

    # Converters to MB for easy reading
    alloc = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"VRAM Used: {alloc:.0f}MB | Peak: {peak:.0f}MB")


def start_cuda_timer():

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    return [start_event, end_event]


def read_cuda_timer_at_end(event, verbose=True):

    start_event, end_event = event[0], event[-1]
    end_event.record()
    torch.cuda.synchronize()  # Wait for GPU to actually finish
    elapsed_time = start_event.elapsed_time(end_event)  # Returns milliseconds
    if verbose:
        print(f"Forward Pass Time: {elapsed_time:.2f} ms")

    return elapsed_time




# general profilerdecorator
# this only gives "high levelverview" o"

def profile_vram_speed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Reset peak stats for a fresh measurement
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # 2. Execute the function
        result = func(*args, **kwargs)

        # 3. Synchronize to get accurate GPU time
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # 4. Calculate Stats
        duration_ms = (end_time - start_time) * 1000
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 2
            curr_vram = torch.cuda.memory_allocated() / 1024 ** 2

            print(f"[{func.__name__}] Time: {duration_ms:.2f}ms | "
                  f"Peak VRAM: {peak_vram:.1f}MB | "
                  f"Final VRAM: {curr_vram:.1f}MB")
        else:
            print(f"[{func.__name__}] Time: {duration_ms:.2f}ms (CPU)")

        return result

    return wrapper

