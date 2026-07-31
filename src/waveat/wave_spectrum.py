import numpy as np
from numpy.typing import ArrayLike


def pierson_moskowitz_spectrum(omega: ArrayLike, Hs: float, Tp: float):
    omega_arr = np.asarray(omega, dtype=float)

    if not np.all(np.isfinite(omega_arr)) or np.any(omega_arr <= 0):
        raise ValueError("All omega values must be positive finite numbers.")
    if not np.isfinite(Hs) or Hs <= 0:
        raise ValueError(
            "Significant wave height (Hs) must be a positive finite number."
        )
    if not np.isfinite(Tp) or Tp <= 0:
        raise ValueError("Peak period (Tp) must be a positive finite number.")

    omega_p = 2 * np.pi / Tp
    x = omega_arr / omega_p
    with np.errstate(over="ignore", under="ignore"):
        log_spectrum = (
            np.log((5.0 / 16.0) * Hs**2 / omega_p) - 5.0 * np.log(x) - 1.25 * x ** (-4)
        )
    spectrum = np.exp(log_spectrum)

    return spectrum


def jonswap_spectrum(
    omega: ArrayLike,
    Hs: float,
    Tp: float,
    gamma: float = 3.3,
    *,
    sigma_a: float = 0.07,
    sigma_b: float = 0.09,
):
    omega_arr = np.asarray(omega, dtype=float)

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Gamma must be a positive finite number.")
    if (
        not np.isfinite(sigma_a)
        or not np.isfinite(sigma_b)
        or sigma_a <= 0
        or sigma_b <= 0
    ):
        raise ValueError("Sigma values must be positive finite numbers.")

    pm = pierson_moskowitz_spectrum(omega_arr, Hs, Tp)
    omega_p = 2 * np.pi / Tp
    sigma = np.where(omega_arr <= omega_p, sigma_a, sigma_b)

    A_gamma = 0.2 / (0.065 * gamma**0.803 + 0.135)
    r = np.exp(-0.5 * ((omega_arr - omega_p) / (sigma * omega_p)) ** 2)

    return A_gamma * pm * gamma**r
