import numpy as np
from numpy.typing import ArrayLike, NDArray


def hyperbolic_pair(
    k: complex,
    local_z: ArrayLike,
    derivative_order: int = 0,
):
    if not isinstance(derivative_order, (int, np.integer)) or derivative_order < 0:
        raise ValueError("derivative_order must be a non-negative integer.")

    local_z_arr = np.asarray(local_z)

    factor = k**derivative_order
    argument = k * local_z_arr

    if derivative_order % 2 == 0:
        return factor * np.cosh(argument), factor * np.sinh(argument)

    return factor * np.sinh(argument), factor * np.cosh(argument)


def add_pair_to_row(
    row: NDArray[np.complex128],
    start: int,
    k: complex,
    local_z: float,
    derivative_order: int = 0,
    scale: complex = 1.0,
) -> None:
    if row.ndim != 1:
        raise ValueError("row must be a 1D array.")
    if not np.issubdtype(row.dtype, np.complexfloating):
        raise ValueError("row must be of complex type.")
    if not 0 <= start < row.size - 1:
        raise IndexError("start index is out of bounds for the row array.")

    basis_0, basis_1 = hyperbolic_pair(k, local_z, derivative_order)
    row[start : start + 2] += scale * np.array([basis_0, basis_1], dtype=row.dtype)


def add_modes_to_row(
    row: NDArray[np.complex128],
    roots: NDArray[np.complex128],
    local_z: float,
    derivative_order: int = 0,
    scale: complex = 1.0,
    weights: NDArray[np.complex128] | None = None,
    start: int = 2,
) -> None:
    if roots.ndim != 1:
        raise ValueError("roots must be a 1D array.")
    if weights is not None and weights.shape != roots.shape:
        raise ValueError("weights must have the same shape as roots.")

    required_size = start + 2 * roots.size
    if start < 0 or required_size > row.size:
        raise IndexError("row does not have enough space for all modes.")

    for i, root in enumerate(roots):
        weight = 1.0 if weights is None else weights[i]
        add_pair_to_row(
            row, start + 2 * i, root, local_z, derivative_order, scale * weight
        )
