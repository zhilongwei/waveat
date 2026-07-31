from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..canopy import Orientation


@dataclass(frozen=True, slots=True)
class CanopyGeometry:
    h: float
    d1: float
    d2: float
    orientation: Orientation

    @property
    def d3(self) -> float:
        return self.h - self.d1 - self.d2

    @property
    def has_layer3(self) -> bool:
        return self.d3 > 0

    @property
    def canopy_top(self) -> float:
        return -self.d1

    @property
    def canopy_btm(self) -> float:
        return -self.d1 - self.d2

    def layer_bounds(self, layer: int) -> tuple[float, float]:
        if layer == 1:
            return -self.d1, 0.0
        elif layer == 2:
            return self.canopy_btm, self.canopy_top
        elif layer == 3 and self.has_layer3:
            return -self.h, self.canopy_btm
        else:
            raise ValueError(f"Invalid layer number: {layer}. Must be 1, 2, or 3.")

    def available_layers(self) -> tuple[int, ...]:
        return (1, 2, 3) if self.has_layer3 else (1, 2)

    def validate_z_in_layer(self, z: ArrayLike, layer: int) -> None:
        z_arr = np.asarray(z)

        z_min, z_max = self.layer_bounds(layer)
        if not np.all((z_arr >= z_min) & (z_arr <= z_max)):
            raise ValueError(
                f"z values must be within layer {layer} bounds [{z_min}, {z_max}]."
            )

    def validate_z_from_canopy_bottom(self, zeta: ArrayLike) -> None:
        zeta_arr = np.asarray(zeta)

        if not np.all((0.0 <= zeta_arr) & (zeta_arr <= self.d2)):
            raise ValueError(f"Canopy-local coordinates must be within [0, {self.d2}].")

    def z_from_canopy_bottom(self, z: ArrayLike):
        z_arr = np.asarray(z)

        return z_arr - self.canopy_btm

    def distance_from_root(self, zeta: ArrayLike):
        zeta_arr = np.asarray(zeta)

        if self.orientation == "upward":
            return zeta_arr
        else:
            return self.d2 - zeta_arr

    def derivative_sign(self, derivative_order: int) -> int:
        if self.orientation == "downward" and derivative_order % 2 == 1:
            return -1
        return 1
