import numpy as np
import pytest

from waveat.constants import G_ACCEL
from waveat.wave_canopy.basis import (
    add_modes_to_row,
    add_pair_to_row,
    hyperbolic_pair,
)
from waveat.wave_canopy.boundary_conditions import (
    BoundaryConditionContext,
    boundary_condition_matrix,
    empty_boundary_condition_matrix,
    scaled_determinant,
)
from waveat.wave_canopy.geometry import CanopyGeometry
from waveat.wave_canopy.motion import (
    euler_bernoulli_beam_parameters,
    kappa_from_k,
    rigid_bar_canopy_pressure_values,
    rigid_bar_displacement_value,
    rigid_bar_parameters,
)

# =============================================================================
# 1. CanopyGeometry Tests
# =============================================================================


def test_geometry_upward_bottom_mounted():
    geom = CanopyGeometry(h=5.0, d1=4.0, d2=1.0, orientation="upward")

    assert geom.d3 == pytest.approx(0.0)
    assert geom.has_layer3 is False
    assert geom.canopy_top == pytest.approx(-4.0)
    assert geom.canopy_btm == pytest.approx(-5.0)
    assert geom.available_layers() == (1, 2)

    assert geom.layer_bounds(1) == (-4.0, 0.0)
    assert geom.layer_bounds(2) == (-5.0, -4.0)
    with pytest.raises(ValueError, match="Invalid layer number"):
        geom.layer_bounds(3)


def test_geometry_downward_suspended():
    geom = CanopyGeometry(h=5.0, d1=0.5, d2=1.0, orientation="downward")

    assert geom.d3 == pytest.approx(3.5)
    assert geom.has_layer3 is True
    assert geom.canopy_top == pytest.approx(-0.5)
    assert geom.canopy_btm == pytest.approx(-1.5)
    assert geom.available_layers() == (1, 2, 3)

    assert geom.layer_bounds(1) == (-0.5, 0.0)
    assert geom.layer_bounds(2) == (-1.5, -0.5)
    assert geom.layer_bounds(3) == (-5.0, -1.5)


@pytest.mark.parametrize("invalid_layer", [-1, 0, 4, 10])
def test_geometry_rejects_invalid_layer(invalid_layer):
    geom = CanopyGeometry(h=5.0, d1=1.0, d2=1.0, orientation="upward")
    with pytest.raises(ValueError, match="Invalid layer number"):
        geom.layer_bounds(invalid_layer)


def test_geometry_z_validations():
    geom = CanopyGeometry(h=5.0, d1=4.0, d2=1.0, orientation="upward")

    # Valid layer coordinates
    geom.validate_z_in_layer([-2.0, 0.0], layer=1)
    geom.validate_z_in_layer([-4.5], layer=2)

    # Invalid coordinates
    with pytest.raises(ValueError, match="z values must be within"):
        geom.validate_z_in_layer([-4.5], layer=1)  # z is in layer 2

    # Canopy-local coordinates zeta in [0, d2]
    geom.validate_z_from_canopy_bottom([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="Canopy-local coordinates"):
        geom.validate_z_from_canopy_bottom([-0.1])
    with pytest.raises(ValueError, match="Canopy-local coordinates"):
        geom.validate_z_from_canopy_bottom([1.1])


def test_geometry_coordinate_mappings_and_derivatives():
    geom_up = CanopyGeometry(h=5.0, d1=4.0, d2=1.0, orientation="upward")
    geom_down = CanopyGeometry(h=5.0, d1=0.5, d2=1.0, orientation="downward")

    # z_from_canopy_bottom: maps z -> zeta in [0, d2]
    assert geom_up.z_from_canopy_bottom(-4.5) == pytest.approx(0.5)

    # distance_from_root: upward vs downward
    assert geom_up.distance_from_root(0.2) == pytest.approx(0.2)
    assert geom_down.distance_from_root(0.2) == pytest.approx(0.8)

    # derivative_sign
    assert geom_up.derivative_sign(0) == 1
    assert geom_up.derivative_sign(1) == 1
    assert geom_down.derivative_sign(0) == 1
    assert geom_down.derivative_sign(1) == -1
    assert geom_down.derivative_sign(2) == 1
    assert geom_down.derivative_sign(3) == -1


# =============================================================================
# 2. Hyperbolic Basis Function Tests
# =============================================================================


def test_hyperbolic_pair_orders():
    k = 0.5 + 0.1j
    z = 1.2

    # Order 0: (cosh(kz), sinh(kz))
    c0, s0 = hyperbolic_pair(k, z, derivative_order=0)
    assert c0 == pytest.approx(np.cosh(k * z))
    assert s0 == pytest.approx(np.sinh(k * z))

    # Order 1: (k*sinh(kz), k*cosh(kz))
    c1, s1 = hyperbolic_pair(k, z, derivative_order=1)
    assert c1 == pytest.approx(k * np.sinh(k * z))
    assert s1 == pytest.approx(k * np.cosh(k * z))

    # Order 2: (k^2*cosh(kz), k^2*sinh(kz))
    c2, s2 = hyperbolic_pair(k, z, derivative_order=2)
    assert c2 == pytest.approx(k**2 * np.cosh(k * z))
    assert s2 == pytest.approx(k**2 * np.sinh(k * z))

    # Order 3: (k^3*sinh(kz), k^3*cosh(kz))
    c3, s3 = hyperbolic_pair(k, z, derivative_order=3)
    assert c3 == pytest.approx(k**3 * np.sinh(k * z))
    assert s3 == pytest.approx(k**3 * np.cosh(k * z))


@pytest.mark.parametrize("invalid_order", [-1, -5, "0", 1.5])
def test_hyperbolic_pair_rejects_invalid_order(invalid_order):
    with pytest.raises(
        ValueError, match="derivative_order must be a non-negative integer"
    ):
        hyperbolic_pair(0.5, 1.0, derivative_order=invalid_order)  # pyright: ignore[reportArgumentType]


def test_add_pair_to_row():
    row = np.zeros(6, dtype=np.complex128)
    k = 0.5
    add_pair_to_row(row, start=2, k=k, local_z=0.0, derivative_order=0, scale=2.0)

    # At local_z=0: cosh(0)=1, sinh(0)=0 -> row[2]=2.0, row[3]=0.0
    assert row[2] == pytest.approx(2.0)
    assert row[3] == pytest.approx(0.0)


def test_add_pair_to_row_validations():
    # Non-1D row
    with pytest.raises(ValueError, match="row must be a 1D array"):
        add_pair_to_row(
            np.zeros((2, 2), dtype=np.complex128), start=0, k=0.5, local_z=0.0
        )

    # Non-complex dtype
    with pytest.raises(ValueError, match="row must be of complex type"):
        add_pair_to_row(np.zeros(4, dtype=float), start=0, k=0.5, local_z=0.0)

    # Out of bounds start index
    row = np.zeros(4, dtype=np.complex128)
    with pytest.raises(IndexError, match="start index is out of bounds"):
        add_pair_to_row(row, start=3, k=0.5, local_z=0.0)
    with pytest.raises(IndexError, match="start index is out of bounds"):
        add_pair_to_row(row, start=-1, k=0.5, local_z=0.0)


def test_add_modes_to_row():
    row = np.zeros(8, dtype=np.complex128)
    roots = np.array([0.3 + 0.1j, 0.6 + 0.2j], dtype=np.complex128)
    weights = np.array([1.5, 2.5], dtype=np.complex128)

    add_modes_to_row(row, roots=roots, local_z=0.0, start=2, weights=weights, scale=1.0)

    # At z=0: cosh(0)=1 -> row[2]=1.5, row[4]=2.5
    assert row[2] == pytest.approx(1.5)
    assert row[3] == pytest.approx(0.0)
    assert row[4] == pytest.approx(2.5)
    assert row[5] == pytest.approx(0.0)


def test_add_modes_to_row_validations():
    row = np.zeros(6, dtype=np.complex128)
    roots = np.array([0.3, 0.6], dtype=np.complex128)

    # Roots non-1D
    with pytest.raises(ValueError, match="roots must be a 1D array"):
        add_modes_to_row(row, roots.reshape(1, 2), local_z=0.0)

    # Weights shape mismatch
    with pytest.raises(ValueError, match="weights must have the same shape as roots"):
        add_modes_to_row(
            row,
            roots,
            local_z=0.0,
            weights=np.array([1.0], dtype=np.complex128),
        )

    # Row capacity insufficient (needs start + 2*roots.size = 2 + 4 = 6)
    small_row = np.zeros(4, dtype=np.complex128)
    with pytest.raises(
        IndexError, match="row does not have enough space for all modes"
    ):
        add_modes_to_row(small_row, roots, local_z=0.0, start=2)


# =============================================================================
# 3. Motion & Parameter Helper Tests
# =============================================================================


def test_kappa_from_k():
    k = 0.5
    D_over_omega = 0.2
    AA = 1.5
    n = 0.98

    kappa = kappa_from_k(k, D_over_omega, AA=AA, n=n)
    expected = k / np.sqrt(AA - 1j * n * D_over_omega)
    assert kappa == pytest.approx(expected)


def test_euler_bernoulli_beam_parameters():
    k = np.complex128(0.4 - 0.02j)
    D_over_omega = 0.15

    params = euler_bernoulli_beam_parameters(
        k=k,
        D_over_omega=D_over_omega,
        N=80.0,
        n=0.99,
        pn=1.0,
        A1=0.005,
        A2=0.008,
        rhos=920.0,
        EI=2.08e-3,
        omega=1.25,
        rhow=1025.0,
    )

    assert isinstance(params.kappa, complex)
    assert params.roots.shape == (3,)
    assert params.displacement_weights.shape == (3,)
    assert np.all(np.isfinite(params.roots))
    assert np.all(np.isfinite(params.displacement_weights))


def test_rigid_bar_parameters_and_evaluators():
    k = np.complex128(0.4 - 0.02j)
    geom = CanopyGeometry(h=5.0, d1=4.0, d2=1.0, orientation="upward")

    params = rigid_bar_parameters(
        k=k,
        D_over_omega=0.15,
        N=80.0,
        n=0.99,
        pn=1.0,
        A1=0.005,
        A2=0.008,
        rhos=920.0,
        K=5.0,
        l=1.0,
        orientation="upward",
        omega=1.25,
        rhow=1025.0,
    )

    assert np.isfinite(params.rotational_stiffness)
    assert np.isfinite(params.moment_of_length)

    # Evaluate pressure and displacement in rigid bar canopy
    zeta = np.linspace(0.0, 1.0, 5)
    p_vals = rigid_bar_canopy_pressure_values(
        zeta=zeta,
        geometry=geom,
        params=params,
        pressure_a=np.complex128(1.0 + 0.1j),
        pressure_b=np.complex128(0.5 + 0.05j),
        theta=np.complex128(0.02 + 0.001j),
        derivative_order=0,
    )
    assert p_vals.shape == (5,)
    assert np.all(np.isfinite(p_vals))

    # Displacement at root (zeta=0) should be 0
    xs_vals = rigid_bar_displacement_value(
        zeta=zeta,
        geometry=geom,
        theta=np.complex128(0.02 + 0.001j),
        derivative_order=0,
    )
    assert xs_vals.shape == (5,)
    assert xs_vals[0] == pytest.approx(0.0)
    assert np.abs(xs_vals[-1]) > 0.0


# =============================================================================
# 4. Boundary Condition Assembly Tests
# =============================================================================


def test_scaled_determinant():
    matrix = np.array([[2.0, 0.0], [0.0, 4.0]], dtype=np.complex128)
    det = scaled_determinant(matrix)
    # Row normalized matrix has rows [1, 0] and [0, 1], det is 1.0
    assert det == pytest.approx(1.0)


def test_scaled_determinant_rejects_empty_row():
    matrix = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match="Boundary matrix contains an empty row"):
        scaled_determinant(matrix)


@pytest.mark.parametrize(
    "model,has_layer3,expected_dim",
    [
        ("fixed", False, 4),
        ("fixed", True, 6),
        ("rigid_bar", False, 5),
        ("rigid_bar", True, 7),
        ("euler_bernoulli_beam", False, 8),
        ("euler_bernoulli_beam", True, 10),
    ],
)
def test_empty_boundary_condition_matrix_dimensions(model, has_layer3, expected_dim):
    d1 = 0.5 if has_layer3 else 4.0
    geom = CanopyGeometry(h=5.0, d1=d1, d2=1.0, orientation="upward")
    context = BoundaryConditionContext(
        geometry=geom,
        mechanical_model=model,
        N=80.0,
        n=0.99,
        pn=1.0,
        A1=0.005,
        A2=0.008,
        rhos=900.0,
        EI=2.0e-3,
        K=5.0,
        omega=1.25,
        rhow=1025.0,
    )

    mat = empty_boundary_condition_matrix(context)
    assert mat.shape == (expected_dim, expected_dim)


def test_boundary_condition_matrix_assembly():
    geom = CanopyGeometry(h=5.0, d1=4.0, d2=1.0, orientation="upward")
    context = BoundaryConditionContext(
        geometry=geom,
        mechanical_model="fixed",
        N=80.0,
        n=0.99,
        pn=1.0,
        A1=0.005,
        A2=0.008,
        rhos=900.0,
        EI=2.0e-3,
        K=5.0,
        omega=1.25,
        rhow=1025.0,
    )

    k = np.complex128(0.4 - 0.02j)
    mat, rhs = boundary_condition_matrix(
        context, k=k, D_over_omega=0.15, normalized_pressure=True
    )

    assert mat.shape == (4, 4)
    assert rhs.shape == (4,)
    # Free surface dynamic pressure boundary condition: rhs[0] == rhow * g
    assert rhs[0] == pytest.approx(1025.0 * G_ACCEL)
