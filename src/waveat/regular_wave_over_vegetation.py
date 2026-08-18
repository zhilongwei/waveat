from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy import complex128, float64
from numpy.typing import ArrayLike, NDArray

from waveat.constants import G_ACCEL
from waveat.wave_canopy.basis import hyperbolic_pair
from waveat.wave_canopy.boundary_conditions import (
    BoundaryConditionContext,
    euler_bernoulli_beam_parameters_from_context,
    rigid_bar_parameters_from_context,
)

from .airy_wave import AiryWave
from .canopy import Canopy, MechanicalModel
from .wave_canopy.boundary_conditions import (
    boundary_condition_matrix as assemble_boundary_condition_matrix,
)
from .wave_canopy.boundary_conditions import (
    scaled_determinant,
)
from .wave_canopy.geometry import CanopyGeometry
from .wave_canopy.motion import (
    kappa_from_k as canopy_kappa_from_k,
)
from .wave_canopy.motion import (
    rigid_bar_canopy_pressure_values,
    rigid_bar_displacement_value,
)

DragVelocity = Literal["filter", "pore"]
_DRAG_VELOCITY_OPTIONS = {"filter", "pore"}

LayerTransferEvaluator = Callable[
    [npt.NDArray[float64], int],
    complex | npt.NDArray[complex128],
]

LayerRealEvaluator = Callable[
    [npt.NDArray[float64], int],
    float | npt.NDArray[float64],
]


class RegularWaveOverVegetation:
    def __init__(
        self,
        wave: AiryWave,
        canopy: Canopy,
        H: float,
        alpha: float = 0.50,
        drag_velocity: DragVelocity = "filter",
    ) -> None:
        if not isinstance(wave, AiryWave):
            raise TypeError("wave must be an instance of AiryWave.")
        if not isinstance(canopy, Canopy):
            raise TypeError("canopy must be an instance of Canopy.")
        if not np.isfinite(H) or H <= 0:
            raise ValueError("H must be a positive finite number.")
        if not np.isfinite(alpha) or not (0 < alpha <= 1):
            raise ValueError("alpha must be a finite number between 0 and 1.")
        if drag_velocity not in _DRAG_VELOCITY_OPTIONS:
            raise ValueError(
                f"drag_velocity must be one of {sorted(_DRAG_VELOCITY_OPTIONS)}."
            )

        self.wave = wave
        self.canopy = canopy
        self.H = H
        self.alpha = alpha
        self.drag_velocity: DragVelocity = drag_velocity
        self.rhow = wave.rhow

        # Check input consistency
        if not np.isclose(self.wave.h, self.canopy.h):
            raise ValueError("Wave depth and canopy water depth must be the same.")
        if self.canopy.porosity <= 0 or self.canopy.porosity > 1:
            raise ValueError(
                "Canopy porosity must be between 0 (exclusive) and 1 (inclusive)."
            )

        element_model: MechanicalModel = self.canopy.vegetation_element.mechanical_model
        self._mechanical_model: MechanicalModel = (
            "fixed" if self.canopy.N == 0 else element_model
        )
        self._geometry = self._build_geometry()

    def _build_geometry(self) -> CanopyGeometry:
        return CanopyGeometry(
            h=self.canopy.h,
            d1=self.d1,
            d2=self.d2,
            orientation=self.canopy.orientation,
        )

    @property
    def d1(self) -> float:
        return -self.canopy.z_bounds[1]

    @property
    def d2(self) -> float:
        return self.canopy.vegetation_element.height

    @property
    def d3(self) -> float:
        return self.canopy.z_bounds[0] + self.canopy.h

    @property
    def has_layer3(self) -> bool:
        return self._geometry.has_layer3

    @property
    def u0(self) -> complex128:
        return complex128(self.wave.horizontal_velocity_transfer(-self.d1) * self.H / 2)

    @property
    def total_inertia(self) -> float:
        return 1.0 + self.canopy.A1 + self.canopy.A2

    @property
    def pn(self) -> float:
        if self.drag_velocity == "filter":
            return 1.0
        if self.drag_velocity == "pore":
            return self.canopy.porosity
        raise ValueError(
            f"drag_velocity must be one of {sorted(_DRAG_VELOCITY_OPTIONS)}."
        )

    @property
    def drag_prefactor(self) -> float:
        return self.canopy.D0 / self.pn**2

    def _boundary_condition_context(self) -> BoundaryConditionContext:
        element = self.canopy.vegetation_element
        return BoundaryConditionContext(
            geometry=self._geometry,
            mechanical_model=self._mechanical_model,
            N=self.canopy.N,
            n=self.canopy.porosity,
            pn=self.pn,
            A1=self.canopy.A1,
            A2=self.canopy.A2,
            rhos=element.rhos,
            EI=element.flexural_rigidity,
            K=element.spring_stiffness,
            omega=self.wave.omega,
            rhow=self.rhow,
        )

    def _rigid_bar_parameters(self, k: complex128, D_over_omega: float):
        return rigid_bar_parameters_from_context(
            context=self._boundary_condition_context(),
            k=k,
            D_over_omega=D_over_omega,
        )

    def _euler_bernoulli_beam_parameters(
        self, k: complex128, D_over_omega: float
    ) -> tuple[complex128, NDArray[complex128], NDArray[complex128]]:
        params = euler_bernoulli_beam_parameters_from_context(
            context=self._boundary_condition_context(),
            k=k,
            D_over_omega=D_over_omega,
        )
        return params.kappa, params.roots, params.displacement_weights

    def _boundary_condition_matrix(
        self, k: complex128, D_over_omega: float, *, normalized_pressure: bool = False
    ) -> tuple[NDArray[complex128], NDArray[complex128]]:
        return assemble_boundary_condition_matrix(
            context=self._boundary_condition_context(),
            k=k,
            D_over_omega=D_over_omega,
            normalized_pressure=normalized_pressure,
        )

    def kappa_from_k(self, k: complex128, D_over_omega: float) -> complex128:
        if not np.isfinite(D_over_omega):
            raise ValueError("D_over_omega must be a finite number.")
        if D_over_omega < 0:
            raise ValueError("D_over_omega must be non-negative.")

        return canopy_kappa_from_k(
            k=k,
            D_over_omega=D_over_omega,
            AA=self.total_inertia,
            n=self.canopy.porosity,
        )

    def dispersion_relation_residual(
        self, k0: NDArray[float64], D_over_omega: float
    ) -> tuple[float, float]:
        k0_arr = np.asarray(k0, dtype=float)
        if k0_arr.shape != (2,):
            raise ValueError("k0 must contain exactly two values: (kr, ki).")
        if not np.isfinite(D_over_omega):
            raise ValueError("D_over_omega must be a finite number.")
        if D_over_omega < 0:
            raise ValueError("D_over_omega must be non-negative.")

        kr, ki = k0_arr

        k = complex128(kr, ki)
        if np.isclose(k, 0.0):
            raise ValueError("k must be non-zero.")

        kappa = self.kappa_from_k(k, D_over_omega)
        if np.isclose(kappa, 0.0):
            raise ValueError("kappa must be non-zero.")
        if np.abs(kappa) > np.abs(k):
            raise ValueError("|kappa| must be less than or equal to |k|.")
        if self._mechanical_model != "fixed":
            matrix, _ = self._boundary_condition_matrix(k, D_over_omega)
            determinant = scaled_determinant(matrix)
            return float(np.real(determinant)), float(np.imag(determinant))

        omega = self.wave.omega
        d1, d2, d3 = self.d1, self.d2, self.d3
        n = self.canopy.porosity

        T1 = (
            np.tanh(k * d1)
            + np.tanh(k * d3)
            + n * kappa / k * np.tanh(kappa * d2)
            + k / (n * kappa) * np.tanh(k * d1) * np.tanh(kappa * d2) * np.tanh(k * d3)
        )
        T2 = (
            1
            + np.tanh(k * d1) * np.tanh(k * d3)
            + n * kappa / k * np.tanh(k * d1) * np.tanh(kappa * d2)
            + k / (n * kappa) * np.tanh(kappa * d2) * np.tanh(k * d3)
        )
        R = omega * omega - G_ACCEL * k * T1 / T2

        return float(np.real(R)), float(np.imag(R))

    def _solve_pressure_coefficients(
        self,
        k: complex128,
        D_over_omega: float,
    ) -> NDArray[complex128]:
        matrix, rhs = self._boundary_condition_matrix(
            k, D_over_omega, normalized_pressure=True
        )
        try:
            return np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Failed to solve for pressure coefficients. The boundary condition matrix may be singular."
            ) from exc

    def _store_pressure_coefficients(
        self,
        k: complex128,
        D_over_omega: float,
        coefficients: NDArray[complex128],
    ) -> None:
        self._pressure_coefficients = coefficients
        self.k = k
        self.kappa = self.kappa_from_k(k, D_over_omega)
        self.D_over_omega = D_over_omega

        self.H_A1 = coefficients[0]
        self.H_B1 = coefficients[1]

        if self._mechanical_model == "fixed":
            self._canopy_layer_roots = np.array([self.kappa], dtype=complex128)
            self._canopy_layer_displacement_weights = None
            self.H_A2 = coefficients[2]
            self.H_B2 = coefficients[3]
            if self.has_layer3:
                self.H_A3 = coefficients[4]
                self.H_B3 = coefficients[5]
            return

        if self._mechanical_model == "rigid_bar":
            params = self._rigid_bar_parameters(k, D_over_omega)
            self.kappa = params.kappa
            self._rigid_bar_params = params
            self._canopy_layer_roots = np.array([params.kappa], dtype=complex128)
            self._canopy_layer_displacement_weights = None
            self.H_A2 = coefficients[2]
            self.H_B2 = coefficients[3]
            if self.has_layer3:
                self.H_A3 = coefficients[4]
                self.H_B3 = coefficients[5]
                self.H_Theta = coefficients[6]
            else:
                self.H_Theta = coefficients[4]
            return

        _, roots, displacement_weights = self._euler_bernoulli_beam_parameters(
            k, D_over_omega
        )
        self._canopy_layer_roots = roots
        self._canopy_layer_displacement_weights = displacement_weights
        self.H_P2_coefficients = coefficients[2:8]
        if self.has_layer3:
            self.H_A3 = coefficients[8]
            self.H_B3 = coefficients[9]

    def find_wavenumber(
        self,
        D_over_omega: float,
        initial_guess: complex128 | None = None,
    ) -> complex128:
        from scipy.optimize import fsolve

        x0_k = (
            getattr(self, "k", self.wave.k) if initial_guess is None else initial_guess
        )
        solution, _, ier, msg = fsolve(
            self.dispersion_relation_residual,
            x0=(np.real(x0_k), np.imag(x0_k)),
            args=(D_over_omega,),
            full_output=True,
            xtol=1.0e-12,
            maxfev=1000,
        )

        if ier != 1:
            raise RuntimeError(
                f"Failed to find wavenumber. fsolve did not converge: {msg}"
            )

        kr, ki = solution

        residual = self.dispersion_relation_residual(
            np.array([kr, ki], dtype=float), D_over_omega
        )
        residual_atol = 1.0e-8 if self._mechanical_model == "fixed" else 1.0e-10
        if not np.allclose(residual, (0.0, 0.0), atol=residual_atol):
            raise ValueError(
                "Failed to find a wavenumber that satisfies the dispersion relation."
            )

        k = complex128(kr, ki)
        coefficients = self._solve_pressure_coefficients(k, D_over_omega)
        self._store_pressure_coefficients(k, D_over_omega, coefficients)

        return k

    def _require_solution(self, caller: str) -> None:
        if not all(
            hasattr(self, name)
            for name in ("_pressure_coefficients", "k", "D_over_omega")
        ):
            raise RuntimeError(
                f"Cannot call {caller} before finding the wavenumber and solving for pressure coefficients."
            )

    def _canopy_layer_pressure_values(
        self, local_z: ArrayLike, derivative_order: int = 0
    ):
        if self._mechanical_model == "rigid_bar":
            return rigid_bar_canopy_pressure_values(
                zeta=local_z,
                geometry=self._geometry,
                params=self._rigid_bar_params,
                pressure_a=self.H_A2,
                pressure_b=self.H_B2,
                theta=self.H_Theta,
                derivative_order=derivative_order,
            )

        result = np.zeros_like(local_z, dtype=complex128)
        coefficients = self._pressure_coefficients
        roots = self._canopy_layer_roots
        for i, root in enumerate(roots):
            basis_0, basis_1 = hyperbolic_pair(root, local_z, derivative_order)
            result += (
                coefficients[2 + 2 * i] * basis_0 + coefficients[3 + 2 * i] * basis_1
            )
        return result

    def _canopy_layer_displacement_value(
        self, local_z: ArrayLike, derivative_order: int = 0
    ):
        if self._mechanical_model == "fixed":
            return np.zeros_like(np.asarray(local_z), dtype=complex128)
        if self._mechanical_model == "rigid_bar":
            return rigid_bar_displacement_value(
                local_z, self._geometry, self.H_Theta, derivative_order
            )

        result = np.zeros_like(local_z, dtype=complex128)
        coefficients = self.H_P2_coefficients
        roots = self._canopy_layer_roots
        displacement_weights = self._canopy_layer_displacement_weights
        if displacement_weights is None:
            raise RuntimeError(
                "Displacement weights are not available for the Euler-Bernoulli beam model."
            )

        for i, root in enumerate(roots):
            basis_0, basis_1 = hyperbolic_pair(root, local_z, derivative_order)
            mode = coefficients[2 * i] * basis_0 + coefficients[2 * i + 1] * basis_1
            result += displacement_weights[i] * mode
        return result

    def _pressure_in_layer(self, z: ArrayLike, layer: int, derivative_order: int = 0):
        self._require_solution(f"H_P{layer}")
        z_arr = np.asarray(z)
        self._geometry.validate_z_in_layer(z_arr, layer)

        if layer == 1:
            basis_0, basis_1 = hyperbolic_pair(
                self.k, self.d1 + z_arr, derivative_order=derivative_order
            )
            result = (
                self._pressure_coefficients[0] * basis_0
                + self._pressure_coefficients[1] * basis_1
            )
        elif layer == 2:
            result = self._canopy_layer_pressure_values(
                self._geometry.z_from_canopy_bottom(z_arr),
                derivative_order=derivative_order,
            )
        elif layer == 3 and self.has_layer3:
            offset = 8 if self._mechanical_model == "euler_bernoulli_beam" else 4
            basis_0, basis_1 = hyperbolic_pair(
                self.k, self.wave.h + z_arr, derivative_order=derivative_order
            )
            result = (
                self._pressure_coefficients[offset] * basis_0
                + self._pressure_coefficients[offset + 1] * basis_1
            )
        else:
            raise ValueError(f"Invalid layer number: {layer}. Must be 1, 2, or 3.")

        return np.asarray(result, dtype=complex128)

    def _horizontal_velocity_in_layer(self, z: ArrayLike, layer: int):
        self._require_solution(f"U_H{layer}")
        z_arr = np.asarray(z)
        self._geometry.validate_z_in_layer(z_arr, layer)

        if layer == 1 or layer == 3:
            result = (
                self.k
                / self.rhow
                / self.wave.omega
                * self._pressure_in_layer(z_arr, layer)
            )
            return np.asarray(result, dtype=complex128)

        gamma = 1j * self.total_inertia + self.canopy.porosity * self.D_over_omega
        local_z = self._geometry.z_from_canopy_bottom(z_arr)
        pressure = self._canopy_layer_pressure_values(local_z)
        displacement = self._canopy_layer_displacement_value(local_z)
        result = (
            1j
            * self.canopy.porosity
            * self.k
            / self.rhow
            / self.wave.omega
            / gamma
            * pressure
            + self.canopy.porosity
            * self.wave.omega
            * (1j * self.D_over_omega * self.pn - self.canopy.A1)
            / gamma
            * displacement
        )
        return np.asarray(result, dtype=complex128)

    def _vertical_velocity_in_layer(self, z: ArrayLike, layer: int):
        self._require_solution(f"U_V{layer}")
        z_arr = np.asarray(z)
        self._geometry.validate_z_in_layer(z_arr, layer)

        porosity_factor = self.canopy.porosity if layer == 2 else 1.0
        pressure_gradient = self._pressure_in_layer(z_arr, layer, derivative_order=1)
        result = 1j * porosity_factor / self.rhow / self.wave.omega * pressure_gradient

        return np.asarray(result, dtype=complex128)

    def _piecewise_evaluate(
        self,
        z: npt.ArrayLike,
        evaluator: LayerTransferEvaluator,
        caller: str,
    ) -> complex128 | npt.NDArray[complex128]:
        self._require_solution(caller)
        z_arr = np.asarray(z, dtype=float)
        result = np.empty_like(z_arr, dtype=np.complex128)
        assigned = np.zeros_like(z_arr, dtype=bool)

        for layer in self._geometry.available_layers():
            z_min, z_max = self._geometry.layer_bounds(layer)
            mask = (~assigned) & (z_min <= z_arr) & (z_arr <= z_max)
            if np.any(mask):
                result[mask] = evaluator(z_arr[mask], layer)
                assigned[mask] = True

        if not np.all(assigned):
            raise ValueError(f"z must lie within [-{self.wave.h}, 0].")

        if z_arr.ndim == 0:
            return complex128(result.item())

        return result

    def _piecewise_evaluate_real(
        self,
        z: npt.ArrayLike,
        evaluator: LayerRealEvaluator,
        caller: str,
    ) -> float | npt.NDArray[float64]:
        self._require_solution(caller)
        z_arr = np.asarray(z, dtype=float)
        result = np.empty_like(z_arr, dtype=np.float64)
        assigned = np.zeros_like(z_arr, dtype=bool)

        for layer in self._geometry.available_layers():
            z_min, z_max = self._geometry.layer_bounds(layer)
            mask = (~assigned) & (z_min <= z_arr) & (z_arr <= z_max)
            if np.any(mask):
                result[mask] = evaluator(z_arr[mask], layer)
                assigned[mask] = True

        if not np.all(assigned):
            raise ValueError(f"z must lie within [-{self.wave.h}, 0].")

        if z_arr.ndim == 0:
            return float64(result.item())

        return result

    def h_p(self, z: ArrayLike):
        return self._piecewise_evaluate(z, self._pressure_in_layer, "h_p")

    def h_p1(self, z: ArrayLike):
        return self._pressure_in_layer(z, layer=1)

    def h_p2(self, z: ArrayLike):
        return self._pressure_in_layer(z, layer=2)

    def h_p3(self, z: ArrayLike):
        if not self.has_layer3:
            raise ValueError("Layer 3 is not included in the canopy configuration.")
        return self._pressure_in_layer(z, layer=3)

    def h_u(self, z: ArrayLike):
        return self._piecewise_evaluate(z, self._horizontal_velocity_in_layer, "h_u")

    def h_u1(self, z: ArrayLike):
        return self._horizontal_velocity_in_layer(z, layer=1)

    def h_u2(self, z: ArrayLike):
        return self._horizontal_velocity_in_layer(z, layer=2)

    def h_u3(self, z: ArrayLike):
        if not self.has_layer3:
            raise ValueError("Layer 3 is not included in the canopy configuration.")
        return self._horizontal_velocity_in_layer(z, layer=3)

    def h_w(self, z: ArrayLike):
        return self._piecewise_evaluate(z, self._vertical_velocity_in_layer, "h_w")

    def h_w1(self, z: ArrayLike):
        return self._vertical_velocity_in_layer(z, layer=1)

    def h_w2(self, z: ArrayLike):
        return self._vertical_velocity_in_layer(z, layer=2)

    def h_w3(self, z: ArrayLike):
        if not self.has_layer3:
            raise ValueError("Layer 3 is not included in the canopy configuration.")
        return self._vertical_velocity_in_layer(z, layer=3)

    def h_xs(self, z: ArrayLike):
        self._require_solution("h_xs")
        z_arr = np.asarray(z, dtype=float)
        self._geometry.validate_z_in_layer(z_arr, layer=2)

        result = self._canopy_layer_displacement_value(
            self._geometry.z_from_canopy_bottom(z_arr)
        )

        return np.asarray(result, dtype=complex128)

    def _drag_velocity_transfer(self, z: float) -> complex128:
        velocity = self.h_u(z)
        if self._mechanical_model != "fixed":
            velocity -= 1j * self.wave.omega * self.pn * self.h_xs(z)
        return complex128(velocity)

    def find_linear_damping(self, rtol: float = 1.0e-6, max_itr: int = 100) -> float:
        from scipy.integrate import quad

        if not np.isfinite(rtol) or rtol <= 0:
            raise ValueError("rtol must be a positive finite number.")
        if not isinstance(max_itr, int) or max_itr <= 0:
            raise ValueError("max_itr must be a positive integer.")

        D1 = self.drag_prefactor * np.abs(self.u0)
        if D1 == 0.0:
            self.find_wavenumber(D_over_omega=0.0)
            return 0.0

        D2 = 0.0
        z_start, z_end = self._geometry.layer_bounds(layer=2)

        err = 1.0
        itr_count = 0
        while err > rtol and itr_count < max_itr:
            D_over_omega = D1 / self.wave.omega
            self.find_wavenumber(D_over_omega=D_over_omega)

            work1, _ = quad(
                lambda z: np.abs(self._drag_velocity_transfer(z)) ** 3, z_start, z_end
            )
            work2, _ = quad(
                lambda z: np.abs(self._drag_velocity_transfer(z)) ** 2, z_start, z_end
            )

            if not np.isfinite(work2) or work2 <= 0.0:
                raise RuntimeError(
                    "Failed to compute linear damping: integral of |U|^2 is non-positive or infinite."
                )

            D2 = self.drag_prefactor * 8 / 3 / np.pi * self.H / 2.0 * work1 / work2
            D_next = (1 - self.alpha) * D1 + self.alpha * D2
            err = abs(D_next - D1) / abs(D1) if D1 != 0 else abs(D_next)
            D1 = D_next

            itr_count += 1

        if err > rtol:
            raise RuntimeError(
                f"Failed to converge to a solution for linear damping within the specified tolerance of {rtol}."
            )

        self.find_wavenumber(D_over_omega=D1 / self.wave.omega)
        return D1

    def setH0(self, H: float) -> None:
        if not np.isfinite(H) or H <= 0:
            raise ValueError("H must be a positive finite number.")
        self.H = H
        self.find_linear_damping()

    def wave_heights_along_canopy(
        self,
        x: ArrayLike,
        H0: float | None = None,
    ):
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("x must be a one-dimensional array.")
        if x.size == 0:
            raise ValueError("x must contain at least one coordinate.")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite coordinates.")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing.")
        if H0 is not None and (not np.isfinite(H0) or H0 <= 0):
            raise ValueError("H0 must be a positive finite number.")

        Nx = x.size
        heights = np.empty(Nx, dtype=float64)
        kr = np.empty(Nx, dtype=float64)
        ki = np.empty(Nx, dtype=float64)

        original_H = self.H
        heights[0] = H0 if H0 is not None else original_H

        try:
            self.H = heights[0]

            for i in range(Nx - 1):
                self.find_linear_damping()
                kr[i] = np.real(self.k)
                ki[i] = np.imag(self.k)

                if ki[i] > 0:
                    raise RuntimeError(
                        f"Non-physical wavenumber: at x[{i}] = {x[i]}: ki = {ki[i]}"
                    )

                dx = x[i + 1] - x[i]
                heights[i + 1] = heights[i] * np.exp(ki[i] * dx)
                self.H = heights[i + 1]

            self.find_linear_damping()
            kr[-1] = np.real(self.k)
            ki[-1] = np.imag(self.k)
        finally:
            self.H = original_H

        self.find_linear_damping()

        return heights, kr, ki
