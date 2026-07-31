from dataclasses import dataclass

import numpy as np
from numpy import complex128
from numpy.typing import ArrayLike, NDArray

from ..canopy import Orientation
from ..constants import G_ACCEL
from .basis import hyperbolic_pair
from .geometry import CanopyGeometry


@dataclass(frozen=True, slots=True)
class EulerBernoulliBeamParameters:
    kappa: complex128
    roots: NDArray[complex128]
    displacement_weights: NDArray[complex128]


@dataclass(frozen=True, slots=True)
class RigidBarParameters:
    kappa: complex128
    particular_factor: complex128
    moment_of_length: float
    pressure_integral_cosh: complex128
    pressure_integral_sinh: complex128
    rotational_stiffness: complex128
    rotation_load_factor: complex128
    pressure_load_factor: complex128
    motion_load_factor: complex128


def kappa_from_k(k: complex, D_over_omega: float, *, AA: float, n: float) -> complex128:
    return complex128(k / np.sqrt(AA - 1j * n * D_over_omega))


def euler_bernoulli_beam_parameters(
    k: complex128,
    D_over_omega: float,
    *,
    N: float,
    n: float,
    pn: float,
    A1: float,
    A2: float,
    rhos: float,
    EI: float,
    omega: float,
    rhow: float,
) -> EulerBernoulliBeamParameters:
    gamma = 1j * (1 + A1 + A2) + n * D_over_omega
    kappa = kappa_from_k(k, D_over_omega, AA=1 + A1 + A2, n=n)
    mass_per_length = rhos * A2 / N

    zeta = rhow * k * omega**2 * (A1 - 1j * D_over_omega * pn) / gamma
    if np.isclose(zeta, 0.0):
        raise ValueError(
            "The Euler-Bernoulli beam model has zero pressure-structure coupling."
        )

    lambda4 = (
        -(
            mass_per_length * omega**2
            + rhow / N * omega**2 * (1j * A1 + D_over_omega * pn) / gamma
        )
        / EI
    )
    xi = k / N * (A1 + A2 - 1j * n * D_over_omega) / gamma / EI

    roots_squared = np.roots(
        [1.0, -(kappa**2), lambda4, -(kappa**2 * lambda4 + xi * zeta)]
    )
    roots = np.sqrt(roots_squared).astype(complex128)
    displacement_weights = ((kappa**2 - roots**2) / zeta).astype(np.complex128)

    return EulerBernoulliBeamParameters(complex128(kappa), roots, displacement_weights)


def rigid_bar_parameters(
    k: complex128,
    D_over_omega: float,
    *,
    N: float,
    n: float,
    pn: float,
    A1: float,
    A2: float,
    rhos: float,
    K: float,
    l: float,
    orientation: Orientation,
    omega: float,
    rhow: float,
) -> RigidBarParameters:
    gamma = 1j * (1 + A1 + A2) + n * D_over_omega
    kappa = kappa_from_k(k, D_over_omega, AA=1 + A1 + A2, n=n)

    mass_per_length = rhos * A2 / N
    displaced_mass_per_length = rhow * A2 / N

    zeta = rhow * k * omega**2 * (A1 - 1j * D_over_omega * pn) / gamma
    particular_factor = zeta / kappa**2
    moment_of_length = l**3 / 3
    pressure_integral_cosh = l * np.sinh(kappa * l) / kappa
    pressure_integral_cosh -= (np.cosh(kappa * l) - 1) / kappa**2
    pressure_integral_sinh = l * np.cosh(kappa * l) / kappa
    pressure_integral_sinh -= np.sinh(kappa * l) / kappa**2

    buoyancy_sign = 1.0 if orientation == "downward" else -1.0
    buoyancy_restoring = (
        buoyancy_sign
        * 0.5
        * (mass_per_length - displaced_mass_per_length)
        * G_ACCEL
        * l**2
    )
    rotation_stiffness = (
        K
        + buoyancy_restoring
        - omega**2 * (mass_per_length + rhow * A1 / N) * moment_of_length
        + 1j * omega * rhow * D_over_omega * omega * pn / N * moment_of_length
    )
    rotation_load_factor = (
        1j * omega * rhow * (A1 + A2) / N / n + rhow * D_over_omega * omega / N
    )
    pressure_load_factor = 1j * n * k / rhow / omega / gamma
    motion_load_factor = n * omega * (1j * D_over_omega * pn - A1) / gamma

    return RigidBarParameters(
        complex128(kappa),
        complex128(particular_factor),
        float(moment_of_length),
        complex128(pressure_integral_cosh),
        complex128(pressure_integral_sinh),
        complex128(rotation_stiffness),
        complex128(rotation_load_factor),
        complex128(pressure_load_factor),
        complex128(motion_load_factor),
    )


def add_rigid_bar_layer_to_row(
    row: NDArray[np.complex128],
    geometry: CanopyGeometry,
    params: RigidBarParameters,
    zeta: float,
    theta_index: int,
    derivative_order: int = 0,
    scale: complex = 1.0,
) -> None:
    s = geometry.distance_from_root(zeta)
    if np.ndim(s) != 0:
        raise ValueError(
            "Rigid-bar model matrix assembly expects a scalar local coordinate."
        )
    s = float(s)

    sign = geometry.derivative_sign(derivative_order)
    basis_0, basis_1 = hyperbolic_pair(params.kappa, s, derivative_order)

    if derivative_order == 0:
        theta_basis = params.particular_factor * s
    elif derivative_order == 1:
        theta_basis = params.particular_factor
    else:
        theta_basis = 0.0

    row[2] += scale * sign * basis_0
    row[3] += scale * sign * basis_1
    row[theta_index] += scale * sign * theta_basis


def add_rigid_bar_rotation_row(
    row: NDArray[np.complex128],
    params: RigidBarParameters,
    theta_index: int,
) -> None:
    row[2] += (
        -params.rotation_load_factor
        * params.pressure_load_factor
        * params.pressure_integral_cosh
    )
    row[3] += (
        -params.rotation_load_factor
        * params.pressure_load_factor
        * params.pressure_integral_sinh
    )
    row[theta_index] += (
        params.rotational_stiffness
        - params.rotation_load_factor
        * params.moment_of_length
        * (
            params.pressure_load_factor * params.particular_factor
            + params.motion_load_factor
        )
    )


def rigid_bar_canopy_pressure_values(
    zeta: ArrayLike,
    geometry: CanopyGeometry,
    params: RigidBarParameters,
    pressure_a: complex128,
    pressure_b: complex128,
    theta: complex128,
    derivative_order: int = 0,
):
    local_z_arr = np.asarray(zeta)
    geometry.validate_z_from_canopy_bottom(local_z_arr)

    s = geometry.distance_from_root(local_z_arr)
    sign = geometry.derivative_sign(derivative_order)
    basis_0, basis_1 = hyperbolic_pair(params.kappa, s, derivative_order)

    if derivative_order == 0:
        theta_basis = params.particular_factor * s
    elif derivative_order == 1:
        theta_basis = params.particular_factor
    else:
        theta_basis = np.zeros_like(s, dtype=complex128)

    result = sign * (pressure_a * basis_0 + pressure_b * basis_1 + theta * theta_basis)
    return np.asarray(result, dtype=complex128)


def rigid_bar_displacement_value(
    zeta: ArrayLike,
    geometry: CanopyGeometry,
    theta: complex128,
    derivative_order: int = 0,
):
    local_z_arr = np.asarray(zeta)
    geometry.validate_z_from_canopy_bottom(local_z_arr)

    s = geometry.distance_from_root(local_z_arr)
    sign = geometry.derivative_sign(derivative_order)

    if derivative_order == 0:
        theta_basis = s
    elif derivative_order == 1:
        theta_basis = np.ones_like(s, dtype=complex128)
    else:
        theta_basis = np.zeros_like(s, dtype=complex128)

    result = sign * theta * theta_basis
    return np.asarray(result, dtype=complex128)
