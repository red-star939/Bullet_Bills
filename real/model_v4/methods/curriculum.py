"""Adaptive backward-start curriculum (environment-agnostic).

This scheduler selects a TARGET time. The caller must obtain a valid environment
state through a genuine reset/snapshot or policy warm-up. This module never
teleports a state or claims synthetic observations are real rollouts.
"""
from dataclasses import dataclass, field
from collections import deque
import random

@dataclass(frozen=True)
class CurriculumConfig:
    minimum_s: float = 50.0
    maximum_s: float = 75.0
    window_s: float = 15.0
    shift_s: float = 5.0
    target_success: float = 0.75
    low_success: float = 0.35
    history_size: int = 100
    min_observations: int = 30

    def __post_init__(self):
        if not 0 <= self.minimum_s < self.maximum_s:
            raise ValueError('invalid range')
        if not 0 < self.window_s <= self.maximum_s - self.minimum_s:
            raise ValueError('invalid window')
        if not 0 <= self.low_success < self.target_success <= 1:
            raise ValueError('invalid thresholds')
        if self.history_size < self.min_observations or self.min_observations < 1:
            raise ValueError('invalid history')

@dataclass
class AdaptiveStartCurriculum:
    config: CurriculumConfig = field(default_factory=CurriculumConfig)
    seed: int = 0
    left_s: float | None = None

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.left_s = self.config.minimum_s if self.left_s is None else float(self.left_s)
        self._history = deque(maxlen=self.config.history_size)
        self._clamp()

    def _clamp(self):
        self.left_s = min(max(self.left_s, self.config.minimum_s),
                          self.config.maximum_s - self.config.window_s)

    def sample_target_s(self):
        return self.rng.uniform(self.left_s, self.left_s + self.config.window_s)

    def record_warmup(self, reached_target: bool):
        self._history.append(bool(reached_target))

    @property
    def warmup_success_rate(self):
        return (sum(self._history)/len(self._history)) if self._history else None

    def adapt(self):
        """Adjust window; no change before enough evidence. Returns new range."""
        if len(self._history) < self.config.min_observations:
            return self.range_s
        rate = self.warmup_success_rate
        if rate >= self.config.target_success:
            self.left_s += self.config.shift_s
        elif rate < self.config.low_success:
            self.left_s -= self.config.shift_s
        self._clamp()
        self._history.clear()
        return self.range_s

    @property
    def range_s(self):
        return (self.left_s, self.left_s + self.config.window_s)
