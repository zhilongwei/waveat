import numpy as np
import pytest

from waveat.canopy import Canopy, CrossSection, VegetationElement

# =============================================================================
# CrossSection Tests
# =============================================================================


def test_cross_section_rectangle_properties():
    b1, b2 = 0.02, 0.005
    cs = CrossSection(b1=b1, b2=b2, shape="rectangle", CD=1.8, CA=0.6)

    assert cs.b1 == b1
    assert cs.b2 == b2
    assert cs.shape == "rectangle"
    assert cs.CD == 1.8
    assert cs.CA == 0.6
    assert cs.area == pytest.approx(b1 * b2)
    assert cs.second_moment_of_area == pytest.approx((b1 * b2**3) / 12.0)


def test_cross_section_circle_properties():
    d = 0.015
    cs = CrossSection(b1=d, b2=d, shape="circle")

    assert cs.b1 == d
    assert cs.b2 == d
    assert cs.shape == "circle"
    assert cs.CD == 2.0  # default
    assert cs.CA == 0.5  # default
    assert cs.area == pytest.approx(np.pi * d**2 / 4.0)
    assert cs.second_moment_of_area == pytest.approx(np.pi * d**4 / 64.0)


@pytest.mark.parametrize("invalid_shape", ["oval", "square", "", None])
def test_cross_section_rejects_invalid_shape(invalid_shape):
    with pytest.raises(ValueError, match="Invalid shape"):
        CrossSection(b1=0.01, b2=0.01, shape=invalid_shape)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "b1,b2",
    [
        (-0.01, 0.01),
        (0.0, 0.01),
        (0.01, -0.01),
        (0.01, 0.0),
        (np.nan, 0.01),
        (0.01, np.inf),
    ],
)
def test_cross_section_rejects_invalid_dimensions(b1, b2):
    with pytest.raises(ValueError, match="b1 and b2 must be positive finite values"):
        CrossSection(b1=b1, b2=b2, shape="rectangle")


@pytest.mark.parametrize("CD", [-0.1, np.nan, np.inf])
def test_cross_section_rejects_invalid_cd(CD):
    with pytest.raises(ValueError, match="CD must be a non-negative finite value"):
        CrossSection(b1=0.01, b2=0.01, shape="rectangle", CD=CD)


@pytest.mark.parametrize("CA", [-0.5, np.nan, -np.inf])
def test_cross_section_rejects_invalid_ca(CA):
    with pytest.raises(ValueError, match="CA must be a non-negative finite value"):
        CrossSection(b1=0.01, b2=0.01, shape="rectangle", CA=CA)


def test_cross_section_circle_requires_equal_axes():
    with pytest.raises(
        ValueError, match="For a circular cross section, b1 and b2 must be equal"
    ):
        CrossSection(b1=0.01, b2=0.02, shape="circle")


# =============================================================================
# VegetationElement Tests
# =============================================================================


@pytest.fixture
def rect_cross_section() -> CrossSection:
    return CrossSection(b1=0.02, b2=0.004, shape="rectangle")


def test_vegetation_element_fixed(rect_cross_section):
    ve = VegetationElement(
        cross_section=rect_cross_section,
        rhos=900.0,
        l=0.6,
        mechanical_model="fixed",
    )
    assert ve.cross_section == rect_cross_section
    assert ve.density == 900.0
    assert ve.height == 0.6
    assert ve.mechanical_model == "fixed"
    assert ve.spring_stiffness == 0.0
    assert ve.youngs_modulus is None
    assert ve.flexural_rigidity == np.inf


def test_vegetation_element_rigid_bar(rect_cross_section):
    ve = VegetationElement(
        cross_section=rect_cross_section,
        rhos=920.0,
        l=0.8,
        mechanical_model="rigid_bar",
        K=12.5,
    )
    assert ve.mechanical_model == "rigid_bar"
    assert ve.spring_stiffness == 12.5
    assert ve.youngs_modulus is None
    assert ve.flexural_rigidity == np.inf


def test_vegetation_element_euler_bernoulli_beam(rect_cross_section):
    E = 1.0e7
    ve = VegetationElement(
        cross_section=rect_cross_section,
        rhos=950.0,
        l=0.5,
        mechanical_model="euler_bernoulli_beam",
        E=E,
    )
    assert ve.mechanical_model == "euler_bernoulli_beam"
    assert ve.youngs_modulus == E
    expected_ei = E * rect_cross_section.second_moment_of_area
    assert ve.flexural_rigidity == pytest.approx(expected_ei)


def test_vegetation_element_type_check_cross_section():
    with pytest.raises(
        TypeError, match="cross_section must be an instance of CrossSection"
    ):
        VegetationElement(
            cross_section="invalid",  # pyright: ignore[reportArgumentType]
            rhos=900.0,
            l=0.5,
            mechanical_model="fixed",
        )


@pytest.mark.parametrize("rhos", [-900.0, 0.0, np.nan, np.inf])
def test_vegetation_element_rejects_invalid_rhos(rect_cross_section, rhos):
    with pytest.raises(
        ValueError, match="Material density must be a positive finite value"
    ):
        VegetationElement(
            cross_section=rect_cross_section,
            rhos=rhos,
            l=0.5,
            mechanical_model="fixed",
        )


@pytest.mark.parametrize("length", [-0.5, 0.0, np.nan, np.inf])
def test_vegetation_element_rejects_invalid_length(rect_cross_section, length):
    with pytest.raises(ValueError, match="Height must be a positive finite value"):
        VegetationElement(
            cross_section=rect_cross_section,
            rhos=900.0,
            l=length,
            mechanical_model="fixed",
        )


def test_vegetation_element_rejects_invalid_mechanical_model(rect_cross_section):
    with pytest.raises(ValueError, match="Invalid mechanical_model"):
        VegetationElement(
            cross_section=rect_cross_section,
            rhos=900.0,
            l=0.5,
            mechanical_model="spring_hinged",  # pyright: ignore[reportArgumentType]
        )


@pytest.mark.parametrize("K", [-1.0, np.nan, np.inf])
def test_vegetation_element_rejects_invalid_spring_stiffness(rect_cross_section, K):
    with pytest.raises(
        ValueError, match="Spring stiffness must be a non-negative finite value"
    ):
        VegetationElement(
            cross_section=rect_cross_section,
            rhos=900.0,
            l=0.5,
            mechanical_model="rigid_bar",
            K=K,
        )


def test_vegetation_element_rejects_nonzero_k_for_non_rigid_bar(rect_cross_section):
    with pytest.raises(
        ValueError,
        match="Spring stiffness K should be zero for mechanical models other than 'rigid_bar'",
    ):
        VegetationElement(
            cross_section=rect_cross_section,
            rhos=900.0,
            l=0.5,
            mechanical_model="fixed",
            K=5.0,
        )


def test_vegetation_element_rejects_e_for_non_beam(rect_cross_section):
    with pytest.raises(
        ValueError,
        match="Young's modulus E should be None for mechanical models other than 'euler_bernoulli_beam'",
    ):
        VegetationElement(
            cross_section=rect_cross_section,
            rhos=900.0,
            l=0.5,
            mechanical_model="fixed",
            E=1e6,
        )


@pytest.mark.parametrize("E", [None, -1e6, 0.0, np.nan, np.inf])
def test_vegetation_element_beam_requires_valid_e(rect_cross_section, E):
    with pytest.raises(
        ValueError,
        match="Young's modulus E must be a positive finite value for 'euler_bernoulli_beam'",
    ):
        VegetationElement(
            cross_section=rect_cross_section,
            rhos=900.0,
            l=0.5,
            mechanical_model="euler_bernoulli_beam",
            E=E,
        )


# =============================================================================
# Canopy Tests
# =============================================================================


@pytest.fixture
def fixed_element(rect_cross_section) -> VegetationElement:
    return VegetationElement(
        cross_section=rect_cross_section,
        rhos=900.0,
        l=0.5,
        mechanical_model="fixed",
    )


def test_canopy_upward_seabed_mounted(fixed_element):
    canopy = Canopy(
        vegetation_element=fixed_element,
        orientation="upward",
        N=100.0,
        h=5.0,
        root_position=-5.0,
    )

    assert canopy.orientation == "upward"
    assert canopy.N == 100.0
    assert canopy.h == 5.0
    assert canopy.root_position == -5.0
    assert canopy.tip_position == pytest.approx(-4.5)
    assert canopy.z_bounds == (-5.0, -4.5)
    assert canopy.has_layer3 is False  # Seabed mounted (no layer below canopy)

    expected_porosity = 1.0 - 100.0 * fixed_element.cross_section.area
    assert canopy.porosity == pytest.approx(expected_porosity)
    assert canopy.n == pytest.approx(expected_porosity)

    # Check hydrodynamic prefactors
    cs = fixed_element.cross_section
    assert canopy.D0 == pytest.approx(0.5 * cs.CD * cs.b1 * 100.0)
    assert canopy.A1 == pytest.approx(cs.CA * cs.b1**2 * (np.pi / 4.0) * 100.0)
    assert canopy.A2 == pytest.approx(cs.area * 100.0)


def test_canopy_downward_suspended(fixed_element):
    canopy = Canopy(
        vegetation_element=fixed_element,
        orientation="downward",
        N=50.0,
        h=10.0,
        root_position=-1.0,
    )

    assert canopy.orientation == "downward"
    assert canopy.tip_position == pytest.approx(-1.5)
    assert canopy.z_bounds == (-1.5, -1.0)
    assert canopy.has_layer3 is True  # Water exists below suspended canopy


def test_canopy_type_check_vegetation_element():
    with pytest.raises(
        TypeError, match="vegetation_element must be an instance of VegetationElement"
    ):
        Canopy(
            vegetation_element="invalid",  # pyright: ignore[reportArgumentType]
            orientation="upward",
            N=100.0,
            h=5.0,
            root_position=-5.0,
        )


def test_canopy_rejects_invalid_orientation(fixed_element):
    with pytest.raises(ValueError, match="Invalid orientation"):
        Canopy(
            vegetation_element=fixed_element,
            orientation="horizontal",  # pyright: ignore[reportArgumentType]
            N=100.0,
            h=5.0,
            root_position=-5.0,
        )


@pytest.mark.parametrize("N", [-10.0, np.nan, np.inf])
def test_canopy_rejects_invalid_n(fixed_element, N):
    with pytest.raises(
        ValueError,
        match="Number of elements per unit area must be a non-negative finite value",
    ):
        Canopy(
            vegetation_element=fixed_element,
            orientation="upward",
            N=N,
            h=5.0,
            root_position=-5.0,
        )


@pytest.mark.parametrize("h", [-5.0, 0.0, np.nan, np.inf])
def test_canopy_rejects_invalid_depth(fixed_element, h):
    with pytest.raises(ValueError, match="Water depth must be a positive finite value"):
        Canopy(
            vegetation_element=fixed_element,
            orientation="upward",
            N=100.0,
            h=h,
            root_position=-5.0,
        )


@pytest.mark.parametrize("root_pos", [np.nan, np.inf, -np.inf])
def test_canopy_rejects_nonfinite_root_position(fixed_element, root_pos):
    with pytest.raises(ValueError, match="Root position must be finite"):
        Canopy(
            vegetation_element=fixed_element,
            orientation="upward",
            N=100.0,
            h=5.0,
            root_position=root_pos,
        )


def test_canopy_rejects_vegetation_taller_than_water_depth(rect_cross_section):
    ve_tall = VegetationElement(
        cross_section=rect_cross_section,
        rhos=900.0,
        l=5.0,  # equals water depth
        mechanical_model="fixed",
    )
    with pytest.raises(
        ValueError,
        match="Vegetation height cannot be equal to or larger than water depth",
    ):
        Canopy(
            vegetation_element=ve_tall,
            orientation="upward",
            N=100.0,
            h=5.0,
            root_position=-5.0,
        )


def test_canopy_rejects_seabed_penetration(fixed_element):
    # h = 5.0, seabed at -5.0, root at -5.1
    with pytest.raises(ValueError, match="The canopy penetrates the seabed"):
        Canopy(
            vegetation_element=fixed_element,
            orientation="upward",
            N=100.0,
            h=5.0,
            root_position=-5.1,
        )


def test_canopy_rejects_surface_penetration(fixed_element):
    # root at -0.2 with upward orientation and length 0.5 reaches +0.3 (above surface)
    with pytest.raises(
        ValueError, match="The canopy must remain below the free surface"
    ):
        Canopy(
            vegetation_element=fixed_element,
            orientation="upward",
            N=100.0,
            h=5.0,
            root_position=-0.2,
        )


def test_canopy_rejects_overfilled_porosity():
    huge_cs = CrossSection(b1=1.0, b2=1.0, shape="rectangle")
    ve = VegetationElement(
        cross_section=huge_cs,
        rhos=900.0,
        l=0.5,
        mechanical_model="fixed",
    )
    with pytest.raises(
        ValueError, match="The canopy is overfilled; porosity must be positive"
    ):
        Canopy(
            vegetation_element=ve,
            orientation="upward",
            N=2.0,  # N * area = 2.0 > 1.0
            h=5.0,
            root_position=-5.0,
        )
