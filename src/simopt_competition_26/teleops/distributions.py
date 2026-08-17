"""Probability distributions used by the teleops simulation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

MINUTES_PER_HOUR = 60.0


@dataclass(frozen=True)
class Nhpp:
    """Piecewise-constant Poisson arrivals with one rate per clock hour."""

    average_rate: float
    hourly_rate_multipliers: tuple[float, ...]

    def sample(self, now: float, rng: random.Random) -> float:
        """Draw the minutes from ``now`` until the next arrival."""
        current = now
        while True:
            hour = int(current // MINUTES_PER_HOUR)
            rate = self.average_rate * self.hourly_rate_multipliers[hour % 24]
            next_hour = (hour + 1) * MINUTES_PER_HOUR

            delay = rng.expovariate(rate / MINUTES_PER_HOUR)
            if current + delay < next_hour:
                return current + delay - now
            current = next_hour


@dataclass(frozen=True)
class BimodalLognormal:
    """A positive-valued mixture of two lognormal distributions.

    Modes are expressed in minutes. ``weight`` is the probability of
    sampling from the first component, while each ``sigma`` controls that
    component's spread in log space. Smaller sigma values produce sharper
    peaks.
    """

    mode1: float
    mode2: float
    sigma1: float
    sigma2: float
    weight: float

    def sample(self, rng: random.Random) -> float:
        """Draw one duration in minutes using ``rng``."""
        if rng.random() < self.weight:
            mode = self.mode1
            sigma = self.sigma1
        else:
            mode = self.mode2
            sigma = self.sigma2

        # A lognormal variable with parameters (mu, sigma) has its mode at
        # exp(mu - sigma**2), so this choice puts the component at the
        # configured mode.
        mu = math.log(mode) + sigma**2
        return rng.lognormvariate(mu, sigma)
