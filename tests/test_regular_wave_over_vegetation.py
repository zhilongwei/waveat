import numpy as np
import pytest

from waveat.airy_wave import AiryWave
from waveat.canopy import Canopy, CrossSection, VegetationElement
from waveat.constants import G_ACCEL
from waveat.regular_wave_over_vegetation import RegularWaveOverVegetation

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def wave_5m_5s() -> AiryWave:
    return AiryWave(h=5.0, T=5.0)


@pytest.fixture
def rect_cross_section() -> CrossSection:
    return CrossSection(b1=0.02, b2=0.005, shape="rectangle", CD=1.8, CA=0.5)


@pytest.fixture
def fixed_element(rect_cross_section: CrossSection) -> VegetationElement:
    return VegetationElement(
        cross_section=rect_cross_section,
        rhos=900.0,
        l=1.0,
        mechanical_model="fixed",
    )


@pytest.fixture
def rigid_bar_element(rect_cross_section: CrossSection) -> VegetationElement:
    return VegetationElement(
        cross_section=rect_cross_section,
        rhos=920.0,
        l=1.0,
        mechanical_model="rigid_bar",
        K=5.0,
    )


@pytest.fixture
def beam_element(rect_cross_section: CrossSection) -> VegetationElement:
    return VegetationElement(
        cross_section=rect_cross_section,
        rhos=950.0,
        l=1.0,
        mechanical_model="euler_bernoulli_beam",
        E=1.0e7,
    )


@pytest.fixture
def bottom_canopy(fixed_element: VegetationElement) -> Canopy:
    # 2-layer seabed mounted canopy
    return Canopy(
        vegetation_element=fixed_element,
        orientation="upward",
        N=100.0,
        h=5.0,
        root_position=-5.0,
    )


@pytest.fixture
def suspended_canopy(fixed_element: VegetationElement) -> Canopy:
    # 3-layer suspended canopy (water above and below canopy)
    return Canopy(
        vegetation_element=fixed_element,
        orientation="downward",
        N=100.0,
        h=5.0,
        root_position=-0.5,
    )


@pytest.fixture
def bottom_model(
    wave_5m_5s: AiryWave, bottom_canopy: Canopy
) -> RegularWaveOverVegetation:
    return RegularWaveOverVegetation(wave=wave_5m_5s, canopy=bottom_canopy, H=0.4)


@pytest.fixture
def suspended_model(
    wave_5m_5s: AiryWave, suspended_canopy: Canopy
) -> RegularWaveOverVegetation:
    return RegularWaveOverVegetation(wave=wave_5m_5s, canopy=suspended_canopy, H=0.4)


# =============================================================================
# 1. Constructor & Input Validation Tests
# =============================================================================


def test_regular_wave_type_checks(wave_5m_5s: AiryWave, bottom_canopy: Canopy):
    with pytest.raises(TypeError, match="wave must be an instance of AiryWave"):
        RegularWaveOverVegetation(wave="invalid", canopy=bottom_canopy, H=0.5)  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="canopy must be an instance of Canopy"):
        RegularWaveOverVegetation(wave=wave_5m_5s, canopy="invalid", H=0.5)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("H", [-0.5, 0.0, np.nan, np.inf])
def test_regular_wave_rejects_invalid_h(
    wave_5m_5s: AiryWave, bottom_canopy: Canopy, H: float
):
    with pytest.raises(ValueError, match="H must be a positive finite number"):
        RegularWaveOverVegetation(wave=wave_5m_5s, canopy=bottom_canopy, H=H)


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.1, np.nan, np.inf])
def test_regular_wave_rejects_invalid_alpha(
    wave_5m_5s: AiryWave, bottom_canopy: Canopy, alpha: float
):
    with pytest.raises(
        ValueError, match="alpha must be a finite number between 0 and 1"
    ):
        RegularWaveOverVegetation(
            wave=wave_5m_5s, canopy=bottom_canopy, H=0.5, alpha=alpha
        )


def test_regular_wave_rejects_invalid_drag_velocity(
    wave_5m_5s: AiryWave, bottom_canopy: Canopy
):
    with pytest.raises(ValueError, match="drag_velocity must be one of"):
        RegularWaveOverVegetation(
            wave=wave_5m_5s,
            canopy=bottom_canopy,
            H=0.5,
            drag_velocity="invalid",  # pyright: ignore[reportArgumentType]
        )


def test_regular_wave_rejects_mismatched_water_depth(bottom_canopy: Canopy):
    wave_10m = AiryWave(h=10.0, T=5.0)
    with pytest.raises(
        ValueError, match="Wave depth and canopy water depth must be the same"
    ):
        RegularWaveOverVegetation(wave=wave_10m, canopy=bottom_canopy, H=0.5)


# =============================================================================
# 2. Geometric and Physical Properties
# =============================================================================


def test_bottom_mounted_geometry_properties(bottom_model: RegularWaveOverVegetation):
    # h = 5.0, canopy upward from -5.0 with l=1.0 -> canopy in [-5.0, -4.0]
    # Layer 1 (above canopy): z in [-4.0, 0.0] -> d1 = 4.0
    # Layer 2 (canopy): d2 = 1.0
    # Layer 3 (below canopy): does not exist -> d3 = -5.0 + 5.0 = 0.0
    assert bottom_model.d1 == pytest.approx(4.0)
    assert bottom_model.d2 == pytest.approx(1.0)
    assert bottom_model.d3 == pytest.approx(0.0)
    assert bottom_model.has_layer3 is False
    assert bottom_model.pn == 1.0  # filter velocity
    assert bottom_model.total_inertia > 1.0


def test_suspended_geometry_properties(suspended_model: RegularWaveOverVegetation):
    # h = 5.0, canopy downward from -0.5 with l=1.0 -> canopy in [-1.5, -0.5]
    # Layer 1 (above): z in [-0.5, 0.0] -> d1 = 0.5
    # Layer 2 (canopy): d2 = 1.0
    # Layer 3 (below): z in [-5.0, -1.5] -> d3 = 3.5
    assert suspended_model.d1 == pytest.approx(0.5)
    assert suspended_model.d2 == pytest.approx(1.0)
    assert suspended_model.d3 == pytest.approx(3.5)
    assert suspended_model.has_layer3 is True


def test_pore_drag_velocity_prefactor(wave_5m_5s: AiryWave, bottom_canopy: Canopy):
    model = RegularWaveOverVegetation(
        wave=wave_5m_5s, canopy=bottom_canopy, H=0.4, drag_velocity="pore"
    )
    assert model.pn == pytest.approx(bottom_canopy.porosity)
    assert model.drag_prefactor == pytest.approx(
        bottom_canopy.D0 / (bottom_canopy.porosity**2)
    )


# =============================================================================
# 3. Limiting Case: Zero Vegetation Density (N = 0)
# =============================================================================


def test_zero_vegetation_density_reduces_to_pure_airy_wave(
    wave_5m_5s: AiryWave, fixed_element: VegetationElement
):
    bare_canopy = Canopy(
        vegetation_element=fixed_element,
        orientation="upward",
        N=0.0,
        h=5.0,
        root_position=-5.0,
    )
    model = RegularWaveOverVegetation(wave=wave_5m_5s, canopy=bare_canopy, H=0.5)

    D = model.find_linear_damping()
    assert D == 0.0
    assert np.real(model.k) == pytest.approx(wave_5m_5s.k, rel=1e-6)
    assert np.imag(model.k) == pytest.approx(0.0, abs=1e-10)

    # Wave height does not decay over distance
    x = np.linspace(0, 50, 6)
    heights, kr, ki = model.wave_heights_along_canopy(x)
    np.testing.assert_allclose(heights, 0.5)
    np.testing.assert_allclose(kr, wave_5m_5s.k, rtol=1e-6)
    np.testing.assert_allclose(ki, 0.0, atol=1e-10)


# =============================================================================
# 4. Solvers & Mechanical Models: Fixed, Rigid Bar, Euler-Bernoulli Beam
# =============================================================================


def test_fixed_canopy_damping_and_wavenumber(bottom_model: RegularWaveOverVegetation):
    D = bottom_model.find_linear_damping()
    assert np.isfinite(D) and D > 0.0
    # Wavenumber should have attenuation (ki < 0)
    assert np.real(bottom_model.k) > 0.0
    assert np.imag(bottom_model.k) < 0.0


def test_rigid_bar_canopy_damping_and_displacement(
    wave_5m_5s: AiryWave, rigid_bar_element: VegetationElement
):
    canopy = Canopy(
        vegetation_element=rigid_bar_element,
        orientation="upward",
        N=80.0,
        h=5.0,
        root_position=-5.0,
    )
    model = RegularWaveOverVegetation(wave=wave_5m_5s, canopy=canopy, H=0.4)

    D = model.find_linear_damping()
    assert np.isfinite(D) and D > 0.0
    assert np.imag(model.k) < 0.0

    # Blade displacement in canopy layer [-5.0, -4.0]
    z_canopy = np.linspace(-5.0, -4.0, 5)
    xs = model.h_xs(z_canopy)
    assert xs.shape == (5,)
    # At the fixed/hinged root (z = -5.0), displacement is 0
    assert np.abs(xs[0]) == pytest.approx(0.0, abs=1e-10)
    # Displacement increases towards the tip
    assert np.abs(xs[-1]) > np.abs(xs[0])


def test_euler_bernoulli_beam_canopy_damping_and_displacement(
    wave_5m_5s: AiryWave, beam_element: VegetationElement
):
    canopy = Canopy(
        vegetation_element=beam_element,
        orientation="upward",
        N=80.0,
        h=5.0,
        root_position=-5.0,
    )
    model = RegularWaveOverVegetation(wave=wave_5m_5s, canopy=canopy, H=0.4)

    D = model.find_linear_damping()
    assert np.isfinite(D) and D > 0.0
    assert np.imag(model.k) < 0.0

    # Beam deflection in canopy layer [-5.0, -4.0]
    z_canopy = np.linspace(-5.0, -4.0, 5)
    xs = model.h_xs(z_canopy)
    assert xs.shape == (5,)
    # Clamped base at z = -5.0 has 0 displacement
    assert np.abs(xs[0]) == pytest.approx(0.0, abs=1e-10)
    assert np.abs(xs[-1]) > 0.0


# =============================================================================
# 5. Boundary Condition & Physical Invariant Checks
# =============================================================================


def test_bottom_model_surface_and_seabed_boundary_conditions(
    bottom_model: RegularWaveOverVegetation,
):
    bottom_model.find_linear_damping()

    # Free surface dynamic condition: p(0) == rhow * g
    p_surface = bottom_model.h_p(0.0)
    assert np.abs(p_surface) == pytest.approx(bottom_model.rhow * G_ACCEL, rel=1e-5)

    # Seabed kinematic condition: vertical velocity w(-h) == 0
    w_seabed = bottom_model.h_w(-bottom_model.wave.h)
    assert np.abs(w_seabed) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("model_type", ["fixed", "rigid_bar", "euler_bernoulli_beam"])
def test_divergence_free_across_all_layers(
    wave_5m_5s: AiryWave,
    fixed_element: VegetationElement,
    rigid_bar_element: VegetationElement,
    beam_element: VegetationElement,
    model_type: str,
):
    element_map = {
        "fixed": fixed_element,
        "rigid_bar": rigid_bar_element,
        "euler_bernoulli_beam": beam_element,
    }
    canopy = Canopy(
        vegetation_element=element_map[model_type],
        orientation="downward",
        N=80.0,
        h=5.0,
        root_position=-0.5,
    )
    model = RegularWaveOverVegetation(wave=wave_5m_5s, canopy=canopy, H=0.4)
    model.find_linear_damping()
    k = model.k
    dz = 1.0e-5

    # Layer 1 (above canopy): z in (-0.5, 0.0)
    z_layer1 = np.linspace(-0.4, -0.1, 3)
    # Layer 2 (canopy layer): z in (-1.5, -0.5)
    z_layer2 = np.linspace(-1.3, -0.7, 3)
    # Layer 3 (below canopy): z in (-5.0, -1.5)
    z_layer3 = np.linspace(-4.5, -2.0, 3)

    for z in np.concatenate([z_layer1, z_layer2, z_layer3]):
        dw_dz = (model.h_w(z + dz) - model.h_w(z - dz)) / (2 * dz)
        u = model.h_u(z)

        # Incompressibility: \nabla \cdot u = -i*k*u + dw/dz = 0 across all layers
        divergence = -1j * k * u + dw_dz
        assert np.abs(divergence) == pytest.approx(0.0, abs=1e-6)


def test_clear_water_layers_are_irrotational(
    suspended_model: RegularWaveOverVegetation,
):
    suspended_model.find_linear_damping()
    k = suspended_model.k
    dz = 1.0e-5

    # Clear water layers: Layer 1 (-0.5, 0.0) and Layer 3 (-5.0, -1.5)
    z_layer1 = np.linspace(-0.4, -0.1, 5)
    z_layer3 = np.linspace(-4.5, -2.0, 5)

    for z in np.concatenate([z_layer1, z_layer3]):
        du_dz = (suspended_model.h_u(z + dz) - suspended_model.h_u(z - dz)) / (2 * dz)
        w = suspended_model.h_w(z)

        # Irrotational: \nabla \times u = du/dz + i*k*w = 0 in clear water
        vorticity = du_dz + 1j * k * w
        assert np.abs(vorticity) == pytest.approx(0.0, abs=1e-6)


def test_bottom_model_pressure_and_vertical_velocity_continuity_at_interface(
    bottom_model: RegularWaveOverVegetation,
):
    bottom_model.find_linear_damping()
    z_interface = -bottom_model.d1  # -4.0 m
    eps = 1.0e-6

    # Pressure continuity across layer 1 and 2
    p_layer1 = bottom_model.h_p1(z_interface)
    p_layer2 = bottom_model.h_p2(z_interface)
    np.testing.assert_allclose(p_layer1, p_layer2, rtol=1e-5)

    # Vertical velocity continuity across layer 1 and 2
    w_layer1 = bottom_model.h_w1(z_interface)
    w_layer2 = bottom_model.h_w2(z_interface)
    np.testing.assert_allclose(w_layer1, w_layer2, rtol=1e-5)

    # Piecewise global evaluation continuity
    np.testing.assert_allclose(
        bottom_model.h_p(z_interface + eps),
        bottom_model.h_p(z_interface - eps),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        bottom_model.h_w(z_interface + eps),
        bottom_model.h_w(z_interface - eps),
        rtol=1e-4,
    )


def test_suspended_model_pressure_and_vertical_velocity_continuity_at_interfaces(
    suspended_model: RegularWaveOverVegetation,
):
    suspended_model.find_linear_damping()
    eps = 1.0e-6

    # Upper interface at z = -d1 = -0.5
    z_upper = -suspended_model.d1
    p1 = suspended_model.h_p1(z_upper)
    p2_top = suspended_model.h_p2(z_upper)
    np.testing.assert_allclose(p1, p2_top, rtol=1e-5)

    w1 = suspended_model.h_w1(z_upper)
    w2_top = suspended_model.h_w2(z_upper)
    np.testing.assert_allclose(w1, w2_top, rtol=1e-5)

    np.testing.assert_allclose(
        suspended_model.h_p(z_upper + eps),
        suspended_model.h_p(z_upper - eps),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        suspended_model.h_w(z_upper + eps),
        suspended_model.h_w(z_upper - eps),
        rtol=1e-4,
    )

    # Lower interface at z = -(d1 + d2) = -1.5
    z_lower = -(suspended_model.d1 + suspended_model.d2)
    p2_bottom = suspended_model.h_p2(z_lower)
    p3 = suspended_model.h_p3(z_lower)
    np.testing.assert_allclose(p2_bottom, p3, rtol=1e-5)

    w2_bottom = suspended_model.h_w2(z_lower)
    w3 = suspended_model.h_w3(z_lower)
    np.testing.assert_allclose(w2_bottom, w3, rtol=1e-5)

    np.testing.assert_allclose(
        suspended_model.h_p(z_lower + eps),
        suspended_model.h_p(z_lower - eps),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        suspended_model.h_w(z_lower + eps),
        suspended_model.h_w(z_lower - eps),
        rtol=1e-4,
    )


# =============================================================================
# 6. Transfer Function Guards & Vectorization
# =============================================================================


def test_transfer_functions_require_prior_solution(
    bottom_model: RegularWaveOverVegetation,
):
    # Calling transfers before find_linear_damping or find_wavenumber raises RuntimeError
    with pytest.raises(RuntimeError, match="Cannot call"):
        bottom_model.h_p(-2.0)
    with pytest.raises(RuntimeError, match="Cannot call"):
        bottom_model.h_u(-2.0)
    with pytest.raises(RuntimeError, match="Cannot call"):
        bottom_model.h_w(-2.0)
    with pytest.raises(RuntimeError, match="Cannot call"):
        bottom_model.h_xs(-4.5)


def test_layer3_functions_reject_call_when_no_layer3(
    bottom_model: RegularWaveOverVegetation,
):
    bottom_model.find_linear_damping()
    with pytest.raises(
        ValueError, match="Layer 3 is not included in the canopy configuration"
    ):
        bottom_model.h_p3(-4.5)
    with pytest.raises(
        ValueError, match="Layer 3 is not included in the canopy configuration"
    ):
        bottom_model.h_u3(-4.5)
    with pytest.raises(
        ValueError, match="Layer 3 is not included in the canopy configuration"
    ):
        bottom_model.h_w3(-4.5)


def test_transfer_functions_reject_out_of_bounds_z(
    bottom_model: RegularWaveOverVegetation,
):
    bottom_model.find_linear_damping()
    with pytest.raises(ValueError, match=r"z must lie within \[-5.0, 0\]"):
        bottom_model.h_p(0.5)  # above surface
    with pytest.raises(ValueError, match=r"z must lie within \[-5.0, 0\]"):
        bottom_model.h_p(-5.5)  # below seabed


def test_transfer_functions_support_vectorized_inputs(
    bottom_model: RegularWaveOverVegetation,
):
    bottom_model.find_linear_damping()
    z = np.linspace(-5.0, 0.0, 21)

    p = bottom_model.h_p(z)
    u = bottom_model.h_u(z)
    w = bottom_model.h_w(z)

    assert p.shape == (21,)
    assert u.shape == (21,)
    assert w.shape == (21,)
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(w))


# =============================================================================
# 7. Spatial Decay (`wave_heights_along_canopy` & `set_H0`)
# =============================================================================


def test_wave_heights_along_canopy_decays_monotonically(
    bottom_model: RegularWaveOverVegetation,
):
    x = np.linspace(0.0, 60.0, 15)
    heights, kr, ki = bottom_model.wave_heights_along_canopy(x, H0=0.4)

    assert heights.shape == (15,)
    assert kr.shape == (15,)
    assert ki.shape == (15,)
    assert heights[0] == pytest.approx(0.4)
    # Heights must strictly decrease along the vegetated canopy
    assert np.all(np.diff(heights) < 0.0)
    assert heights[-1] < heights[0]
    # Attenuation rate ki must remain negative
    assert np.all(ki < 0.0)


def test_wave_heights_along_canopy_preserves_initial_h(
    bottom_model: RegularWaveOverVegetation,
):
    original_h = bottom_model.H
    x = np.linspace(0.0, 30.0, 5)
    bottom_model.wave_heights_along_canopy(x, H0=0.6)
    # Model's internal H must be restored after computation
    assert bottom_model.H == pytest.approx(original_h)


@pytest.mark.parametrize(
    "x_invalid",
    [
        np.array([[0.0, 1.0], [2.0, 3.0]]),  # 2D array
        np.array([]),  # empty
        np.array([0.0, np.nan, 10.0]),  # non-finite
        np.array([0.0, 10.0, 5.0]),  # not strictly increasing
        np.array([5.0, 5.0]),  # non-increasing (equal values)
    ],
)
def test_wave_heights_rejects_invalid_x(
    bottom_model: RegularWaveOverVegetation, x_invalid
):
    with pytest.raises(ValueError):
        bottom_model.wave_heights_along_canopy(x_invalid)


@pytest.mark.parametrize("H0_invalid", [-0.5, 0.0, np.nan, np.inf])
def test_wave_heights_rejects_invalid_h0(
    bottom_model: RegularWaveOverVegetation, H0_invalid
):
    with pytest.raises(ValueError, match="H0 must be a positive finite number"):
        bottom_model.wave_heights_along_canopy(np.linspace(0.0, 10.0, 5), H0=H0_invalid)


def test_set_h0_updates_height_and_recomputes_damping(
    bottom_model: RegularWaveOverVegetation,
):
    bottom_model.find_linear_damping()
    d_initial = bottom_model.D_over_omega

    bottom_model.set_H0(0.8)
    assert bottom_model.H == pytest.approx(0.8)
    # Higher wave height results in larger drag velocity and higher damping
    assert bottom_model.D_over_omega > d_initial


@pytest.mark.parametrize("H_invalid", [-0.5, 0.0, np.nan, np.inf])
def test_set_h0_rejects_invalid_h(bottom_model: RegularWaveOverVegetation, H_invalid):
    with pytest.raises(ValueError, match="H must be a positive finite number"):
        bottom_model.set_H0(H_invalid)


# =============================================================================
# 8. Solver Convergence & Parameter Validation
# =============================================================================


@pytest.mark.parametrize("rtol", [-1e-6, 0.0, np.nan, np.inf])
def test_find_linear_damping_rejects_invalid_rtol(
    bottom_model: RegularWaveOverVegetation, rtol
):
    with pytest.raises(ValueError, match="rtol must be a positive finite number"):
        bottom_model.find_linear_damping(rtol=rtol)


@pytest.mark.parametrize("max_itr", [-10, 0, 1.5, "100"])
def test_find_linear_damping_rejects_invalid_max_itr(
    bottom_model: RegularWaveOverVegetation, max_itr
):
    with pytest.raises(ValueError, match="max_itr must be a positive integer"):
        bottom_model.find_linear_damping(max_itr=max_itr)  # pyright: ignore[reportArgumentType]


def test_find_linear_damping_raises_when_max_iterations_exceeded(
    bottom_model: RegularWaveOverVegetation,
):
    # Setting max_itr=1 should fail to converge within strict rtol
    with pytest.raises(
        RuntimeError, match="Failed to converge to a solution for linear damping"
    ):
        bottom_model.find_linear_damping(rtol=1e-12, max_itr=1)
