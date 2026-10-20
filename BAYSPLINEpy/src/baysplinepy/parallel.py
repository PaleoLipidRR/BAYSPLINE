"""Running independent blocks of work on every CPU (PORTING.md UKP-08).

MATLAB's ``parfor`` hands the 500 posterior draws to however many workers the
pool has, and every worker pulls from its own random stream. Here the draws are
cut into a fixed number of blocks, each with its own child of one
:class:`numpy.random.SeedSequence`; how many CPUs run those blocks changes the
wall-clock and nothing else. ``seed=0`` on one core and on sixty-four gives the
same numbers, bit for bit.
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Iterable

BACKENDS = ("process", "thread")


def available_cpus() -> int:
    """CPUs this process may run on — the affinity mask where the OS has one.

    ``os.cpu_count()`` reports the machine, which over-counts inside a container
    or a batch-scheduler allocation.

    Returns:
        A positive integer.
    """
    if hasattr(os, "process_cpu_count"):          # Python >= 3.13
        n = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count()
    return max(1, n or 1)


def resolve_n_jobs(n_jobs: int | None) -> int:
    """joblib's convention: ``-1`` is every CPU, ``-2`` all but one, and so on.

    Args:
        n_jobs: Requested workers. ``None`` means ``-1``.

    Returns:
        Workers to start, at least one.

    Raises:
        ValueError: if `n_jobs` is 0.
    """
    if n_jobs is None:
        n_jobs = -1
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must be a positive count, or negative to count back from all CPUs")
    if n_jobs < 0:
        return max(1, available_cpus() + 1 + n_jobs)
    return n_jobs


def run_blocks(fn: Callable[..., Any], jobs: Iterable[tuple], n_jobs: int | None,
               backend: str = "process") -> list:
    """``[fn(*job) for job in jobs]``, spread over `n_jobs` workers, in order.

    Processes are the default. The sampler is hundreds of small array
    operations per iteration, so threads spend their time waiting on the GIL:
    measured on the demo record with four CPUs, threads were three times
    *slower* than one core and processes three times faster.

    A script run on macOS or Windows starts workers by re-importing itself, so
    it needs the usual ``if __name__ == "__main__":`` guard. Without one the
    pool breaks; this falls back to one core with a warning rather than failing.

    Args:
        fn: A top-level function (it must pickle, for ``"process"``).
        jobs: Argument tuples.
        n_jobs: Workers; see :func:`resolve_n_jobs`.
        backend: ``"process"`` or ``"thread"``.

    Returns:
        The results, in the order of `jobs`.
    """
    jobs = list(jobs)
    workers = min(resolve_n_jobs(n_jobs), len(jobs))
    if workers <= 1:
        return [fn(*job) for job in jobs]
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    pool: type[Executor] = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    try:
        with pool(max_workers=workers) as ex:
            return list(ex.map(fn, *zip(*jobs)))
    except BrokenProcessPool:
        warnings.warn("worker processes could not start (a script on macOS or Windows "
                      "needs an `if __name__ == '__main__':` guard); running on one core",
                      RuntimeWarning, stacklevel=3)
        return [fn(*job) for job in jobs]
