import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate

from .airy_wave import AiryWave
from .canopy import Canopy
from .constants import DENS_SEAWATER
from .regular_wave_over_vegetation import DragVelocity, RegularWaveOverVegetation


class IrregularWaveOverVegetation:
    def __init__(
        self,
        canopy: Canopy,
        Nz: int,
    ):
        if not isinstance(canopy, Canopy):
            raise TypeError("canopy must be an instance of Canopy.")
        if not isinstance(Nz, int) or Nz <= 0:
            raise ValueError("Nz must be a positive integer.")
        if not Nz >= 2:
            raise ValueError("Nz must be at least 2 to define a vertical grid.")

        self.canopy = canopy
        self.Nz = Nz

    @property
    def d1(self) -> float:
        return -self.canopy.z_bounds[1]

    @property
    def d2(self) -> float:
        return self.canopy.vegetation_element.height

    @property
    def d3(self) -> float:
        return self.canopy.z_bounds[0] + self.canopy.h

    def find_linear_damping(
        self,
        omegas: ArrayLike,
        Sw: ArrayLike,
        alpha=0.7,
        rtol=1.0e-8,
        max_itr=100,
        drag_velocity: DragVelocity = "filter",
        rhow: float = DENS_SEAWATER,
    ):
        omegas_arr = np.asarray(omegas, dtype=np.float64)
        Sw_arr = np.asarray(Sw, dtype=np.float64)

        if omegas_arr.ndim != 1 or Sw_arr.ndim != 1:
            raise ValueError("omegas and Sw must be one-dimensional arrays.")
        if omegas_arr.shape != Sw_arr.shape:
            raise ValueError("omegas and Sw must have the same shape.")
        if omegas_arr.size < 2:
            raise ValueError("At least two frequency values are required.")
        if not np.all(np.isfinite(omegas_arr)) or np.any(omegas_arr <= 0):
            raise ValueError("omegas must contain positive finite values.")
        if np.any(np.diff(omegas_arr) <= 0):
            raise ValueError("omegas must be strictly increasing.")
        if not np.all(np.isfinite(Sw_arr)) or np.any(Sw_arr < 0):
            raise ValueError("Sw must contain non-negative finite values.")
        if not isinstance(max_itr, int) or max_itr <= 0:
            raise ValueError("max_itr must be a positive integer.")

        z0 = 0.5 * sum(self.canopy.z_bounds)
        Su = np.empty_like(Sw_arr)

        for i, omega in enumerate(omegas_arr):
            T = 2 * np.pi / omega
            wave = AiryWave(h=self.canopy.h, T=T, rhow=rhow)
            hu = wave.horizontal_velocity_transfer(z0)
            Su[i] = np.abs(hu) ** 2 * Sw_arr[i]

        variance = float(integrate.simpson(Su, omegas_arr))
        sigma_u = float(np.sqrt(variance))

        # Initial guess of linear damping
        if drag_velocity == "filter":
            pn = 1.0
        elif drag_velocity == "pore":
            pn = self.canopy.n
        else:
            raise ValueError("drag_velocity must be either 'filter' or 'pore'.")

        D1 = self.canopy.D0 / pn**2 * np.sqrt(8.0 / np.pi) * sigma_u
        D2 = 0

        zz = np.linspace(self.canopy.z_bounds[0], self.canopy.z_bounds[1], self.Nz)

        err = 1.0
        itr_count = 0

        while err > rtol and itr_count < max_itr:
            Sur = np.empty((omegas_arr.size, zz.size), dtype=np.float64)

            for i, omega in enumerate(omegas_arr):
                T = 2.0 * np.pi / omega
                wave = AiryWave(h=self.canopy.h, T=T, rhow=rhow)

                model = RegularWaveOverVegetation(
                    wave=wave,
                    canopy=self.canopy,
                    H=1.0,
                    alpha=alpha,
                    drag_velocity=drag_velocity,
                )
                model.find_wavenumber(D1 / omega)

                ur = model.h_u2(zz) - 1j * model.pn * omega * model.h_xs(zz)
                Sur[i, :] = np.abs(ur) ** 2 * Sw_arr[i]

            variance = integrate.simpson(Sur, x=omegas_arr, axis=0)
            sigmas_ur = np.sqrt(variance)

            D2 = (
                self.canopy.D0
                / pn**2
                * np.sqrt(8.0 / np.pi)
                * integrate.simpson(sigmas_ur**3, zz)
                / integrate.simpson(sigmas_ur**2, zz)
            )
            err = np.abs(D2 - D1) / D1
            D1 = alpha * D2 + (1.0 - alpha) * D1
            itr_count += 1

        if not np.isfinite(err):
            raise RuntimeError(
                "Failed to converge: the damping error became non-finite."
            )

        if err > rtol:
            raise RuntimeError(f"Failed to converge within {max_itr} iterations.")

        self.D = D1

    def complex_wavenumber(
        self,
        omegas: ArrayLike,
        Sw: ArrayLike,
        alpha=0.7,
        rtol=1.0e-8,
        max_itr=100,
        drag_velocity: DragVelocity = "filter",
        rhow: float = DENS_SEAWATER,
    ):
        self.find_linear_damping(
            omegas,
            Sw,
            alpha=alpha,
            rtol=rtol,
            drag_velocity=drag_velocity,
            max_itr=max_itr,
            rhow=rhow,
        )

        kr = np.empty_like(omegas, dtype=np.float64)
        ki = np.empty_like(omegas, dtype=np.float64)

        for i, omega in enumerate(np.asarray(omegas)):
            T = 2.0 * np.pi / omega
            wave = AiryWave(h=self.canopy.h, T=T, rhow=rhow)
            model = RegularWaveOverVegetation(
                wave=wave,
                canopy=self.canopy,
                H=1.0,
                alpha=alpha,
                drag_velocity=drag_velocity,
            )
            k = model.find_wavenumber(self.D / omega)
            kr[i] = np.real(k)
            ki[i] = np.imag(k)

        return kr, ki

    def wave_spectral_along_canopy(
        self,
        omegas: ArrayLike,
        Sw: ArrayLike,
        x: ArrayLike,
        alpha=0.7,
        rtol=1.0e-8,
        max_itr=100,
        drag_velocity: DragVelocity = "filter",
        rhow: float = DENS_SEAWATER,
    ):
        Sws = []

        Sws.append(Sw)

        x_arr = np.asarray(x)
        for i in range(1, len(x_arr)):
            _, ki = self.complex_wavenumber(
                omegas,
                Sws[i - 1],
                alpha=alpha,
                rtol=rtol,
                max_itr=max_itr,
                drag_velocity=drag_velocity,
                rhow=rhow,
            )
            dx = x_arr[i] - x_arr[i - 1]
            Sws.append(Sws[i - 1] * np.exp(2.0 * ki * dx))

        return np.array(Sws)
