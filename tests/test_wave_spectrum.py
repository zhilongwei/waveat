import numpy as np
import pytest

from waveat.wave_spectrum import jonswap_spectrum, pierson_moskowitz_spectrum


def test_pierson_moskowitz_spectrum_recovers_target_significant_wave_height():
    Hs = 2.0
    Tp = 8.0
    omega = np.linspace(0.2 / Tp, 200.0 / Tp, 20000)

    spectrum = pierson_moskowitz_spectrum(omega, Hs, Tp)
    m0 = np.trapezoid(spectrum, omega)

    assert 4.0 * np.sqrt(m0) == pytest.approx(Hs, rel=1e-6)


def test_jonswap_spectrum_recovers_target_significant_wave_height():
    Hs = 2.0
    Tp = 8.0
    gamma = 3.3
    omega = np.linspace(0.2 / Tp, 200.0 / Tp, 20000)

    spectrum = jonswap_spectrum(omega, Hs, Tp, gamma)
    m0 = np.trapezoid(spectrum, omega)

    assert 4.0 * np.sqrt(m0) == pytest.approx(Hs, rel=1e-3)


@pytest.mark.parametrize(
    "omega, Hs, Tp",
    [
        (np.array([0.0, 1.0]), 2.0, 8.0),
        (np.array([-0.1, 1.0]), 2.0, 8.0),
        (np.array([np.nan, 1.0]), 2.0, 8.0),
        (1.0, -2.0, 8.0),
        (1.0, 0.0, 8.0),
        (1.0, np.nan, 8.0),
        (1.0, 2.0, -8.0),
        (1.0, 2.0, 0.0),
        (1.0, 2.0, np.inf),
    ],
)
def test_pm_rejects_invalid_inputs(omega, Hs, Tp):
    with pytest.raises(ValueError):
        pierson_moskowitz_spectrum(omega, Hs, Tp)


def test_jonswap_rejects_invalid_gamma():
    with pytest.raises(ValueError):
        jonswap_spectrum(1.0, Hs=2.0, Tp=8.0, gamma=-1.0)


def test_jonswap_rejects_invalid_sigma():
    with pytest.raises(ValueError):
        jonswap_spectrum(1.0, Hs=2.0, Tp=8.0, sigma_a=0.0)
    with pytest.raises(ValueError):
        jonswap_spectrum(1.0, Hs=2.0, Tp=8.0, sigma_b=-0.01)


def test_jonswap_gamma_one_equals_pm():
    omega = np.linspace(0.1, 2.0, 100)
    pm = pierson_moskowitz_spectrum(omega, Hs=2.5, Tp=7.0)
    js = jonswap_spectrum(omega, Hs=2.5, Tp=7.0, gamma=1.0)
    np.testing.assert_allclose(js, pm, rtol=1e-4)
