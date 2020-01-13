from fastprogress.fastprogress import master_bar, progress_bar
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Collection, Any
import os
from types import SimpleNamespace
import concurrent.futures

def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

def num_cpus()->int:
    "Get number of cpus"
    try:                   return len(os.sched_getaffinity(0))
    except AttributeError: return os.cpu_count()

_default_cpus = min(16, num_cpus())
defaults = SimpleNamespace(cpus=_default_cpus, cmap='viridis')

# def parallel(func, arr:Collection, max_workers:int=None):
    # "Call `func` on every element of `arr` in parallel using `max_workers`."
    # max_workers = ifnone(max_workers, defaults.cpus)
    # if max_workers<2: _ = [func(o,i) for i,o in enumerate(arr)]
    # else:
        # with ProcessPoolExecutor(max_workers=max_workers) as ex:
            # futures = [ex.submit(func,o,i) for i,o in enumerate(arr)]
    # for f in progress_bar(concurrent.futures.as_completed(futures), total=len(arr)):
        # pass
        # yield f.result()

def parallel(func, arr:Collection, max_workers:int=None):
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    max_workers = ifnone(max_workers, defaults.cpus)
    if max_workers<2: _ = [func(o,i) for i,o in enumerate(arr)]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(func,o,i) for i,o in enumerate(arr)]
            for f in progress_bar(concurrent.futures.as_completed(futures), total=len(arr)): pass
