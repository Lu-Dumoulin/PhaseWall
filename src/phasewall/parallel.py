"""Running a function over many jobs, in parallel processes or not, with progress reports.

Progress: an optional callback progress(step, done, total), called after each job, with
`step` a short name ("simulate", "describe", ...). The notebook interface uses it for a
progress bar; without it nothing is reported. Results come back in the order of the jobs.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor

Progress = Callable[[str, int, int], None]


def run_jobs(func: Callable, jobs: Iterable, workers: int = 1, progress: Progress | None = None,
             step: str = "") -> list:
    """[func(job) for job in jobs], in `workers` processes when workers > 1."""
    jobs = list(jobs)
    out = []
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            for r in pool.map(func, jobs):
                out.append(r)
                if progress:
                    progress(step, len(out), len(jobs))
    else:
        for j in jobs:
            out.append(func(j))
            if progress:
                progress(step, len(out), len(jobs))
    return out
