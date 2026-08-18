import numpy as np
import pytest

from waveat.canopy import Canopy, CrossSection, VegetationElement
from waveat.irregular_wave_over_vegetation import IrregularWaveOverVegetation
from waveat.wave_spectrum import jonswap_spectrum

# =============================================================================
# Fixtures
# =============================================================================


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
    return Canopy(
        vegetation_element=fixed_element,
        orientation="upward",
        N=100.0,
        h=5.0,
        root_position=-5.0,
    )


@pytest.fixture
def irregular_model(bottom_canopy: Canopy) -> IrregularWaveOverVegetation:
    return IrregularWaveOverVegetation(canopy=bottom_canopy, Nz=10)


@pytest.fixture
def jonswap_grid() -> tuple[np.ndarray, np.ndarray]:
    Tp = 5.0
    Hs = 0.5
    omega_p = 2.0 * np.pi / Tp
    # Moderate resolution frequency grid spanning peak
    omegas = np.linspace(0.5 * omega_p, 2.5 * omega_p, 30)
    Sw = jonswap_spectrum(omegas, Hs=Hs, Tp=Tp, gamma=3.3)
    return omegas, Sw


# =============================================================================
# 1. Constructor & Input Validation
# =============================================================================


def test_irregular_wave_rejects_invalid_canopy():
    with pytest.raises(TypeError, match="canopy must be an instance of Canopy"):
        IrregularWaveOverVegetation(canopy="invalid", Nz=10)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("Nz", [-5, 0, 1, 1.5, "10"])
def test_irregular_wave_rejects_invalid_nz(bottom_canopy: Canopy, Nz):
    with pytest.raises(ValueError):
        IrregularWaveOverVegetation(
            canopy=bottom_canopy,
            Nz=Nz,  # pyright: ignore[reportArgumentType]
        )


def test_irregular_wave_geometry_properties(
    irregular_model: IrregularWaveOverVegetation,
):
    # Upward canopy in [-5.0, -4.0] with water depth 5.0
    assert irregular_model.d1 == pytest.approx(4.0)
    assert irregular_model.d2 == pytest.approx(1.0)
    assert irregular_model.d3 == pytest.approx(0.0)


# =============================================================================
# 2. Input Validation on Spectral Methods
# =============================================================================


def test_find_linear_damping_rejects_shape_mismatch(
    irregular_model: IrregularWaveOverVegetation,
):
    omegas = np.linspace(0.5, 2.0, 10)
    Sw = np.ones(8)
    with pytest.raises(ValueError, match="omegas and Sw must have the same shape"):
        irregular_model.find_linear_damping(omegas, Sw)


def test_find_linear_damping_rejects_multidimensional_arrays(
    irregular_model: IrregularWaveOverVegetation,
):
    omegas = np.ones((5, 2))
    Sw = np.ones((5, 2))
    with pytest.raises(ValueError, match="omegas and Sw must be one-dimensional"):
        irregular_model.find_linear_damping(omegas, Sw)


def test_find_linear_damping_rejects_insufficient_frequencies(
    irregular_model: IrregularWaveOverVegetation,
):
    with pytest.raises(ValueError, match="At least two frequency values are required"):
        irregular_model.find_linear_damping([1.0], [0.5])


@pytest.mark.parametrize(
    "omegas_invalid",
    [
        np.array([-0.5, 1.0, 1.5]),  # negative frequency
        np.array([0.0, 1.0, 1.5]),  # zero frequency
        np.array([1.0, np.nan, 2.0]),  # non-finite
        np.array([1.0, 0.8, 1.5]),  # not strictly increasing
        np.array([1.0, 1.0, 2.0]),  # non-increasing (equal)
    ],
)
def test_find_linear_damping_rejects_invalid_omegas(
    irregular_model: IrregularWaveOverVegetation, omegas_invalid
):
    Sw = np.ones_like(omegas_invalid)
    with pytest.raises(ValueError):
        irregular_model.find_linear_damping(omegas_invalid, Sw)


@pytest.mark.parametrize(
    "Sw_invalid",
    [
        np.array([-0.1, 0.5, 0.2]),  # negative spectrum value
        np.array([0.5, np.nan, 0.2]),  # non-finite
    ],
)
def test_find_linear_damping_rejects_invalid_sw(
    irregular_model: IrregularWaveOverVegetation, Sw_invalid
):
    omegas = np.array([0.5, 1.0, 1.5])
    with pytest.raises(ValueError):
        irregular_model.find_linear_damping(omegas, Sw_invalid)


def test_find_linear_damping_rejects_invalid_drag_velocity(
    irregular_model: IrregularWaveOverVegetation,
    jonswap_grid: tuple[np.ndarray, np.ndarray],
):
    omegas, Sw = jonswap_grid
    with pytest.raises(ValueError, match="drag_velocity must be either"):
        irregular_model.find_linear_damping(
            omegas,
            Sw,
            drag_velocity="invalid",  # pyright: ignore[reportArgumentType]
        )


# =============================================================================
# 3. Solver & Physical Results
# =============================================================================


def test_find_linear_damping_converges_with_filter_and_pore_velocities(
    irregular_model: IrregularWaveOverVegetation,
    jonswap_grid: tuple[np.ndarray, np.ndarray],
):
    omegas, Sw = jonswap_grid

    # Filter drag velocity
    irregular_model.find_linear_damping(omegas, Sw, drag_velocity="filter", rtol=1e-5)
    d_filter = irregular_model.D
    assert np.isfinite(d_filter) and d_filter > 0.0

    # Pore drag velocity
    irregular_model.find_linear_damping(omegas, Sw, drag_velocity="pore", rtol=1e-5)
    d_pore = irregular_model.D
    assert np.isfinite(d_pore) and d_pore > 0.0


def test_complex_wavenumber_attenuates_across_spectrum(
    irregular_model: IrregularWaveOverVegetation,
    jonswap_grid: tuple[np.ndarray, np.ndarray],
):
    omegas, Sw = jonswap_grid
    kr, ki = irregular_model.complex_wavenumber(omegas, Sw, rtol=1e-5)

    assert kr.shape == omegas.shape
    assert ki.shape == omegas.shape
    # All real wavenumbers must be positive
    assert np.all(kr > 0.0)
    # All imaginary wavenumbers must indicate attenuation (ki < 0)
    assert np.all(ki < 0.0)

    # For bottom-mounted canopies, longer waves (lower frequencies) penetrate deeper
    # into the water column and experience stronger canopy drag than short deep-water waves
    assert np.abs(ki[0]) > np.abs(ki[-1])


def test_wave_spectral_along_canopy_decays_monotonically(
    irregular_model: IrregularWaveOverVegetation,
    jonswap_grid: tuple[np.ndarray, np.ndarray],
):
    omegas, Sw = jonswap_grid
    x = np.linspace(0.0, 40.0, 5)

    Sws = irregular_model.wave_spectral_along_canopy(
        omegas, Sw, x, rtol=1e-5, max_itr=50
    )

    assert Sws.shape == (5, omegas.size)
    np.testing.assert_allclose(Sws[0], Sw)

    # Total spectral variance m0 = \int S(omega) domega
    m0_values = [np.trapezoid(s, omegas) for s in Sws]
    # Significant wave height Hs = 4 * sqrt(m0)
    Hs_values = [4.0 * np.sqrt(m0) for m0 in m0_values]

    # Hs must strictly decrease along canopy propagation distance
    assert np.all(np.diff(Hs_values) < 0.0)
    assert Hs_values[-1] < Hs_values[0]


def test_irregular_wave_with_flexible_beam_model(
    beam_element: VegetationElement,
    jonswap_grid: tuple[np.ndarray, np.ndarray],
):
    canopy = Canopy(
        vegetation_element=beam_element,
        orientation="upward",
        N=80.0,
        h=5.0,
        root_position=-5.0,
    )
    model = IrregularWaveOverVegetation(canopy=canopy, Nz=10)
    omegas, Sw = jonswap_grid

    kr, ki = model.complex_wavenumber(omegas, Sw, rtol=1e-5)
    assert np.all(kr > 0.0)
    assert np.all(ki < 0.0)
    assert np.isfinite(model.D) and model.D > 0.0


def test_find_linear_damping_raises_on_non_convergence(
    irregular_model: IrregularWaveOverVegetation,
    jonswap_grid: tuple[np.ndarray, np.ndarray],
):
    omegas, Sw = jonswap_grid
    with pytest.raises(RuntimeError, match="Failed to converge within 1 iterations"):
        irregular_model.find_linear_damping(omegas, Sw, rtol=1e-15, max_itr=1)
