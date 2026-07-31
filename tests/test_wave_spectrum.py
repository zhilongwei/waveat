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
