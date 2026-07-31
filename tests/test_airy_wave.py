import numpy as np
import pytest

from waveat.airy_wave import AiryWave
from waveat.constants import G_ACCEL

ABS_TOL = 1.0e-8  # Absolute tolerance for floating-point comparisons


def test_airy_wave_initialization():
    wave = AiryWave(h=10.0, T=6.0)

    assert wave.omega**2 == pytest.approx(G_ACCEL * wave.k * np.tanh(wave.k * wave.h))
    assert wave.kh == pytest.approx(wave.k * wave.h)
    assert wave.L == pytest.approx(2 * np.pi / wave.k)


def test_airy_wave_velocities_incompressible_and_irrotational():
    wave = AiryWave(h=10.0, T=6.0)
    dz = 1.0e-5

    for z in np.linspace(-0.8 * wave.h, -0.2 * wave.h, 10):
        du_dz = (
            wave.horizontal_velocity_transfer(z + dz)
            - wave.horizontal_velocity_transfer(z - dz)
        ) / (2 * dz)
        dw_dz = (
            wave.vertical_velocity_transfer(z + dz)
            - wave.vertical_velocity_transfer(z - dz)
        ) / (2 * dz)

        assert np.abs(
            -1j * wave.k * wave.horizontal_velocity_transfer(z) + dw_dz
        ) == pytest.approx(0.0, abs=ABS_TOL)
        assert np.abs(
            -1j * wave.k * wave.vertical_velocity_transfer(z) - du_dz
        ) == pytest.approx(0.0, abs=ABS_TOL)


def test_airy_wave_free_surface_conditions():
    wave = AiryWave(h=10.0, T=6.0)

    assert wave.velocity_potential_transfer(0.0) == pytest.approx(
        1j * G_ACCEL / wave.omega
    )
    assert wave.pressure_transfer(0.0) == pytest.approx(wave.rhow * G_ACCEL)
    assert wave.vertical_velocity_transfer(0.0) == pytest.approx(1j * wave.omega)
    assert wave.vertical_velocity_transfer(-wave.h) == pytest.approx(0.0)
