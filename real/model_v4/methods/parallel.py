"""Windows-safe process setup helper; actual env factory supplied by future trainer."""
import multiprocessing as mp

def spawn_context():
    return mp.get_context('spawn')


def suggested_workers(logical_cpus, reserve=2):
    if logical_cpus < 1:
        raise ValueError('logical_cpus must be positive')
    return max(1, logical_cpus - max(0, reserve))
