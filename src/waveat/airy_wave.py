from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from .constants import DENS_SEAWATER, G_ACCEL


@dataclass(frozen=True, slots=True)
class AiryWave:
    h: float
    T: float
    rhow: float = DENS_SEAWATER

    omega: float = field(init=False)
    kh: float = field(init=False)
    k: float = field(init=False)
    L: float = field(init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.h) or self.h <= 0:
            raise ValueError("Depth must be a positive finite number.")
        if not np.isfinite(self.T) or self.T <= 0:
            raise ValueError("Period must be a positive finite number.")
        if not np.isfinite(self.rhow) or self.rhow <= 0:
            raise ValueError("Water density must be a positive finite number.")

        with np.errstate(over="ignore", under="ignore"):
            omega = float(2.0 * np.pi / self.T)
            dimensionless_frequency_squared = float(np.square(omega) * self.h / G_ACCEL)

        if (
            not np.isfinite(omega)
            or omega <= 0.0
            or not np.isfinite(dimensionless_frequency_squared)
            or dimensionless_frequency_squared <= 0.0
        ):
            raise ValueError(
                "Depth and period produce non-representable wave parameters."
            )

        def dispersion_residual(kh: float) -> float:
            return dimensionless_frequency_squared - kh * np.tanh(kh)

        kh = cast(
            float,
            brentq(
                dispersion_residual,
                0.0,
                dimensionless_frequency_squared + 1.0,
            ),
        )

        k = kh / self.h
        L = 2.0 * np.pi / k

        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "kh", kh)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "L", L)

    def velocity_potential_transfer(self, z: ArrayLike):
        cosh_ratio, _ = self._vertical_shape_ratios(z)
        return (1j * G_ACCEL / self.omega) * cosh_ratio

    def pressure_transfer(self, z: ArrayLike):
        return -self.rhow * 1j * self.omega * self.velocity_potential_transfer(z)

    def horizontal_velocity_transfer(self, z: ArrayLike):
        return -1j * self.k * self.velocity_potential_transfer(z)

    def vertical_velocity_transfer(self, z: ArrayLike):
        _, sinh_ratio = self._vertical_shape_ratios(z)
        return (1j * G_ACCEL / self.omega) * self.k * sinh_ratio

    def _validate_z(self, z: ArrayLike):
        z_arr = np.asarray(z)

        if not np.all(np.isfinite(z_arr)):
            raise ValueError("z must be finite.")

        if not np.all((-self.h <= z_arr) & (z_arr <= 0)):
            raise ValueError("z must be within the range [-h, 0].")

        return z_arr

    def _vertical_shape_ratios(self, z: ArrayLike):
        z_arr = self._validate_z(z)

        a = self.k * (z_arr + self.h)
        b = self.kh

        scaled_exponential = np.exp(a - b)
        denominator = 1.0 + np.exp(-2.0 * b)

        cosh_ratio = scaled_exponential * (1.0 + np.exp(-2.0 * a)) / denominator
        sinh_ratio = scaled_exponential * (-np.expm1(-2.0 * a)) / denominator

        return cosh_ratio, sinh_ratio
