from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np

Shape = Literal["rectangle", "circle"]
MechanicalModel = Literal["fixed", "rigid_bar", "euler_bernoulli_beam"]
Orientation = Literal["upward", "downward"]

_SHAPE_OPTIONS = get_args(Shape)
_MECHANICAL_MODEL_OPTIONS = get_args(MechanicalModel)
_ORIENTATION_OPTIONS = get_args(Orientation)


@dataclass(frozen=True, slots=True)
class CrossSection:
    b1: float
    b2: float
    shape: Shape
    CD: float = 2.0
    CA: float = 0.5

    def __post_init__(self) -> None:
        if self.shape not in _SHAPE_OPTIONS:
            raise ValueError(
                f"Invalid shape: {self.shape}. Must be one of {sorted(_SHAPE_OPTIONS)}."
            )

        if (
            self.b1 <= 0
            or self.b2 <= 0
            or not np.isfinite(self.b1)
            or not np.isfinite(self.b2)
        ):
            raise ValueError("b1 and b2 must be positive finite values.")

        if self.CD < 0 or not np.isfinite(self.CD):
            raise ValueError("CD must be a non-negative finite value.")

        if self.CA < 0 or not np.isfinite(self.CA):
            raise ValueError("CA must be a non-negative finite value.")

        if self.shape == "circle" and not np.isclose(
            self.b1, self.b2, rtol=1e-8, atol=0.0
        ):
            raise ValueError("For a circular cross section, b1 and b2 must be equal.")

    @property
    def area(self) -> float:
        if self.shape == "rectangle":
            return self.b1 * self.b2
        elif self.shape == "circle":
            return np.pi * self.b1 * self.b2 / 4
        else:
            raise ValueError(
                f"Invalid shape: {self.shape}. Must be one of {sorted(_SHAPE_OPTIONS)}."
            )

    @property
    def second_moment_of_area(self) -> float:
        if self.shape == "rectangle":
            return (self.b1 * self.b2**3) / 12.0
        elif self.shape == "circle":
            diameter = 0.5 * (self.b1 + self.b2)
            return (np.pi * diameter**4) / 64.0
        else:
            raise ValueError(
                f"Invalid shape: {self.shape}. Must be one of {sorted(_SHAPE_OPTIONS)}."
            )


@dataclass(frozen=True, slots=True)
class VegetationElement:
    cross_section: CrossSection
    rhos: float
    l: float
    mechanical_model: MechanicalModel
    K: float = 0.0
    E: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cross_section, CrossSection):
            raise TypeError("cross_section must be an instance of CrossSection.")

        if not np.isfinite(self.rhos) or self.rhos <= 0:
            raise ValueError("Material density must be a positive finite value.")

        if not np.isfinite(self.l) or self.l <= 0:
            raise ValueError("Height must be a positive finite value.")

        if self.mechanical_model not in _MECHANICAL_MODEL_OPTIONS:
            raise ValueError(
                f"Invalid mechanical_model: {self.mechanical_model}. Must be one of {sorted(_MECHANICAL_MODEL_OPTIONS)}."
            )

        if not np.isfinite(self.K) or self.K < 0:
            raise ValueError("Spring stiffness must be a non-negative finite value.")

        if self.mechanical_model != "rigid_bar" and self.K != 0.0:
            raise ValueError(
                "Spring stiffness K should be zero for mechanical models other than 'rigid_bar'."
            )

        if self.mechanical_model != "euler_bernoulli_beam" and self.E is not None:
            raise ValueError(
                "Young's modulus E should be None for mechanical models other than 'euler_bernoulli_beam'."
            )

        if self.mechanical_model == "euler_bernoulli_beam" and (
            self.E is None or not np.isfinite(self.E) or self.E <= 0
        ):
            raise ValueError(
                "Young's modulus E must be a positive finite value for 'euler_bernoulli_beam' mechanical model."
            )

    @property
    def flexural_rigidity(self) -> float:
        if self.mechanical_model == "euler_bernoulli_beam":
            if self.E is None:
                raise RuntimeError("Validated beam has no Young's modulus.")
            return self.E * self.cross_section.second_moment_of_area

        return np.inf

    @property
    def density(self) -> float:
        return self.rhos

    @property
    def height(self) -> float:
        return self.l

    @property
    def youngs_modulus(self) -> float | None:
        return self.E

    @property
    def spring_stiffness(self) -> float:
        return self.K


@dataclass(frozen=True, slots=True)
class Canopy:
    vegetation_element: VegetationElement
    orientation: Orientation
    N: float
    h: float
    root_position: float

    n: float = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.vegetation_element, VegetationElement):
            raise TypeError(
                "vegetation_element must be an instance of VegetationElement."
            )

        if self.orientation not in _ORIENTATION_OPTIONS:
            raise ValueError(
                f"Invalid orientation: {self.orientation}. Must be one of {sorted(_ORIENTATION_OPTIONS)}."
            )

        if not np.isfinite(self.N) or self.N < 0:
            raise ValueError(
                "Number of elements per unit area must be a non-negative finite value."
            )

        if not np.isfinite(self.h) or self.h <= 0:
            raise ValueError("Water depth must be a positive finite value.")

        if not np.isfinite(self.root_position):
            raise ValueError("Root position must be finite.")

        if self.vegetation_element.height >= self.h:
            raise ValueError(
                "Vegetation height cannot be equal to or larger than water depth."
            )

        canopy_bottom, canopy_top = self.z_bounds
        if canopy_bottom < -self.h:
            raise ValueError("The canopy penetrates the seabed.")
        if canopy_top >= 0.0:
            raise ValueError("The canopy must remain below the free surface.")

        n = 1.0 - self.N * self.vegetation_element.cross_section.area

        if not np.isfinite(n) or n <= 0:
            raise ValueError("The canopy is overfilled; porosity must be positive.")

        object.__setattr__(self, "n", n)

    @property
    def has_layer3(self) -> bool:
        canopy_bottom = self.z_bounds[0]
        return canopy_bottom > -self.h

    @property
    def porosity(self) -> float:
        return self.n

    @property
    def D0(self) -> float:
        return (
            0.5
            * self.vegetation_element.cross_section.CD
            * self.vegetation_element.cross_section.b1
            * self.N
        )

    @property
    def A1(self) -> float:
        return (
            self.vegetation_element.cross_section.CA
            * self.vegetation_element.cross_section.b1**2
            * np.pi
            / 4
            * self.N
        )

    @property
    def A2(self) -> float:
        return self.vegetation_element.cross_section.area * self.N

    @property
    def tip_position(self) -> float:
        if self.orientation == "upward":
            return self.root_position + self.vegetation_element.height
        else:
            return self.root_position - self.vegetation_element.height

    @property
    def z_bounds(self) -> tuple[float, float]:
        return min(self.root_position, self.tip_position), max(
            self.root_position, self.tip_position
        )
