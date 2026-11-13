"""Evaluation and seed split utilities: no model or environment dependency."""
from dataclasses import dataclass
import math
import random

@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    mean_s: float
    std_s: float
    min_s: float
    max_s: float
    reached_80: int
    reached_100: int
    reached_120: int
    reached_150: int


def summarize_survival(durations_s):
    data = [float(x) for x in durations_s]
    if not data or not all(math.isfinite(x) and x >= 0 for x in data):
        raise ValueError('expected finite nonnegative survival times')
    n = len(data)
    mean = sum(data)/n
    std = math.sqrt(sum((x-mean)**2 for x in data)/n)
    return EvaluationSummary(n, mean, std, min(data), max(data),
                             *(sum(x >= t for x in data) for t in (80, 100, 120, 150)))


def make_disjoint_seeds(train_count, validation_count, holdout_count, *, master_seed=0):
    sizes = [train_count, validation_count, holdout_count]
    if any(n < 0 for n in sizes):
        raise ValueError('seed counts must be nonnegative')
    total = sum(sizes)
    if total > 2**31:
        raise ValueError('too many seeds')
    selected = random.Random(master_seed).sample(range(2**31), total)
    a,b = sizes[:2]
    return {'train':selected[:a], 'validation':selected[a:a+b], 'holdout':selected[a+b:]}
