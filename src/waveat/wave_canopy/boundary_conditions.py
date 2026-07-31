from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy import complex128
from numpy.typing import NDArray

from ..canopy import MechanicalModel
from ..constants import G_ACCEL
from .basis import add_modes_to_row, add_pair_to_row
from .geometry import CanopyGeometry
from .motion import (
    EulerBernoulliBeamParameters,
    RigidBarParameters,
    add_rigid_bar_layer_to_row,
    add_rigid_bar_rotation_row,
    euler_bernoulli_beam_parameters,
    kappa_from_k,
    rigid_bar_parameters,
)


@dataclass(frozen=True)
class BoundaryConditionContext:
    geometry: CanopyGeometry
    mechanical_model: MechanicalModel
    N: float
    n: float
    pn: float
    A1: float
    A2: float
    rhos: float
    EI: float
    K: float
    omega: float
    rhow: float


def scaled_determinant(matrix: NDArray[complex128]) -> complex128:
    row_norms = np.linalg.norm(matrix, axis=1)
    if np.any(row_norms == 0):
        raise ValueError("Boundary matrix contains an empty row.")
    return complex128(np.linalg.det(matrix / row_norms[:, np.newaxis]))


def empty_boundary_condition_matrix(
    context: BoundaryConditionContext,
) -> NDArray[complex128]:
    size = 0
    if context.mechanical_model == "fixed":
        size = 4 if not context.geometry.has_layer3 else 6
    elif context.mechanical_model == "rigid_bar":
        size = 5 if not context.geometry.has_layer3 else 7
    elif context.mechanical_model == "euler_bernoulli_beam":
        size = 8 if not context.geometry.has_layer3 else 10
    else:
        raise ValueError(f"Unknown mechanical model: {context.mechanical_model}")

    return np.zeros((size, size), dtype=complex128)


def euler_bernoulli_beam_parameters_from_context(
    context: BoundaryConditionContext,
    k: complex128,
    D_over_omega: float,
) -> EulerBernoulliBeamParameters:
    return euler_bernoulli_beam_parameters(
        k=k,
        D_over_omega=D_over_omega,
        N=context.N,
        n=context.n,
        pn=context.pn,
        A1=context.A1,
        A2=context.A2,
        rhos=context.rhos,
        EI=context.EI,
        omega=context.omega,
        rhow=context.rhow,
    )


def rigid_bar_parameters_from_context(
    context: BoundaryConditionContext,
    k: complex128,
    D_over_omega: float,
) -> RigidBarParameters:
    return rigid_bar_parameters(
        k=k,
        D_over_omega=D_over_omega,
        N=context.N,
        n=context.n,
        pn=context.pn,
        A1=context.A1,
        A2=context.A2,
        rhos=context.rhos,
        K=context.K,
        l=context.geometry.d2,
        orientation=context.geometry.orientation,
        omega=context.omega,
        rhow=context.rhow,
    )


CanopyRowAdder = Callable[
    [NDArray[complex128], float, int, complex],
    None,
]


def _assemble_fluid_boundary_rows(
    matrix: NDArray[complex128],
    rhs: NDArray[complex128],
    context: BoundaryConditionContext,
    k: complex128,
    *,
    add_canopy_to_row: CanopyRowAdder,
    layer3_start: int,
    normalized_pressure: bool,
) -> int:
    """Assemble the fluid boundary conditions and return the next row index."""
    geometry = context.geometry

    # Free surface at z=0.
    add_pair_to_row(matrix[0], 0, k, geometry.d1)
    if normalized_pressure:
        rhs[0] = context.rhow * G_ACCEL
    else:
        add_pair_to_row(
            matrix[0],
            0,
            k,
            geometry.d1,
            derivative_order=1,
            scale=-G_ACCEL / context.omega**2,
        )

    # Upper fluid-canopy interface at z=-d1.
    add_pair_to_row(matrix[1], 0, k, 0.0)
    add_canopy_to_row(matrix[1], geometry.d2, 0, -1.0)

    add_pair_to_row(matrix[2], 0, k, 0.0, derivative_order=1)
    add_canopy_to_row(matrix[2], geometry.d2, 1, -context.n)

    if not geometry.has_layer3:
        # Impermeable bottom at z=-h.
        add_canopy_to_row(matrix[3], 0.0, 1, 1.0)
        return 4

    # Lower canopy-fluid interface at z=-(d1+d2).
    add_canopy_to_row(matrix[3], 0.0, 0, 1.0)
    add_pair_to_row(matrix[3], layer3_start, k, geometry.d3, scale=-1.0)

    add_canopy_to_row(matrix[4], 0.0, 1, context.n)
    add_pair_to_row(
        matrix[4],
        layer3_start,
        k,
        geometry.d3,
        derivative_order=1,
        scale=-1.0,
    )

    # Impermeable bottom at z=-h.
    add_pair_to_row(matrix[5], layer3_start, k, 0.0, derivative_order=1)
    return 6


def fixed_boundary_condition_matrix(
    context: BoundaryConditionContext,
    k: complex128,
    kappa: complex128,
    *,
    normalized_pressure: bool = False,
) -> tuple[NDArray[complex128], NDArray[complex128]]:
    matrix = empty_boundary_condition_matrix(context)
    rhs = np.zeros(matrix.shape[0], dtype=complex128)

    def add_canopy_to_row(
        row: NDArray[complex128],
        zeta: float,
        derivative_order: int,
        scale: complex,
    ) -> None:
        add_pair_to_row(
            row,
            2,
            kappa,
            zeta,
            derivative_order=derivative_order,
            scale=scale,
        )

    _ = _assemble_fluid_boundary_rows(
        matrix,
        rhs,
        context,
        k,
        add_canopy_to_row=add_canopy_to_row,
        layer3_start=4,
        normalized_pressure=normalized_pressure,
    )

    return matrix, rhs


def rigid_bar_boundary_condition_matrix(
    context: BoundaryConditionContext,
    k: complex128,
    D_over_omega: float,
    *,
    normalized_pressure: bool = False,
) -> tuple[NDArray[complex128], NDArray[complex128]]:
    matrix = empty_boundary_condition_matrix(context)
    rhs = np.zeros(matrix.shape[0], dtype=complex128)

    params = rigid_bar_parameters_from_context(context, k, D_over_omega)
    theta_index = 4 if not context.geometry.has_layer3 else 6

    def add_canopy_to_row(
        row: NDArray[complex128],
        zeta: float,
        derivative_order: int,
        scale: complex,
    ) -> None:
        add_rigid_bar_layer_to_row(
            row,
            context.geometry,
            params,
            zeta=zeta,
            theta_index=theta_index,
            derivative_order=derivative_order,
            scale=scale,
        )

    next_row = _assemble_fluid_boundary_rows(
        matrix,
        rhs,
        context,
        k,
        add_canopy_to_row=add_canopy_to_row,
        layer3_start=4,
        normalized_pressure=normalized_pressure,
    )

    # Rigid-bar rotational moment balance about the hinge.
    add_rigid_bar_rotation_row(matrix[next_row], params, theta_index)

    return matrix, rhs


def euler_bernoulli_beam_boundary_condition_matrix(
    context: BoundaryConditionContext,
    k: complex128,
    D_over_omega: float,
    *,
    normalized_pressure: bool = False,
) -> tuple[NDArray[complex128], NDArray[complex128]]:
    matrix = empty_boundary_condition_matrix(context)
    rhs = np.zeros(matrix.shape[0], dtype=complex128)

    params = euler_bernoulli_beam_parameters_from_context(context, k, D_over_omega)
    roots = params.roots
    displacement_weights = params.displacement_weights

    def add_canopy_to_row(
        row: NDArray[complex128],
        zeta: float,
        derivative_order: int,
        scale: complex,
    ) -> None:
        add_modes_to_row(
            row,
            roots,
            local_z=zeta,
            derivative_order=derivative_order,
            scale=scale,
        )

    next_row = _assemble_fluid_boundary_rows(
        matrix,
        rhs,
        context,
        k,
        add_canopy_to_row=add_canopy_to_row,
        layer3_start=8,
        normalized_pressure=normalized_pressure,
    )

    if context.geometry.orientation == "upward":
        root_zeta, tip_zeta = 0.0, context.geometry.d2
    else:
        root_zeta, tip_zeta = context.geometry.d2, 0.0

    # Clamped root (displacement and slope) and free tip (moment and shear).
    for zeta, derivative_order in (
        (root_zeta, 0),
        (root_zeta, 1),
        (tip_zeta, 2),
        (tip_zeta, 3),
    ):
        add_modes_to_row(
            matrix[next_row],
            roots,
            local_z=zeta,
            derivative_order=derivative_order,
            weights=displacement_weights,
        )
        next_row += 1

    return matrix, rhs


def boundary_condition_matrix(
    context: BoundaryConditionContext,
    k: complex128,
    D_over_omega: float,
    *,
    normalized_pressure: bool = False,
) -> tuple[NDArray[complex128], NDArray[complex128]]:
    if context.mechanical_model == "fixed":
        kappa = kappa_from_k(
            k, D_over_omega, AA=1 + context.A1 + context.A2, n=context.n
        )
        return fixed_boundary_condition_matrix(
            context, k, kappa, normalized_pressure=normalized_pressure
        )
    elif context.mechanical_model == "rigid_bar":
        return rigid_bar_boundary_condition_matrix(
            context, k, D_over_omega, normalized_pressure=normalized_pressure
        )
    elif context.mechanical_model == "euler_bernoulli_beam":
        return euler_bernoulli_beam_boundary_condition_matrix(
            context, k, D_over_omega, normalized_pressure=normalized_pressure
        )
    else:
        raise ValueError(f"Unknown mechanical model: {context.mechanical_model}")
