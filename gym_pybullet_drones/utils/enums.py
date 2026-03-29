from enum import Enum

import numpy as np

class DroneModel(Enum):
    """Drone models enumeration class."""

    CF2X = "cf2x"   # Bitcraze Craziflie 2.0 in the X configuration
    CF2P = "cf2p"   # Bitcraze Craziflie 2.0 in the + configuration
    RACE = "racer"  # Racer drone in the X configuration
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class DroneModelFormat(Enum):
    CF2X = "urdf"
    CF2P = "urdf"
    RACE = "urdf"
    A = "hardcode"
    B = "hardcode"
    C = "hardcode"
    D = "hardcode"
    E = "hardcode"


class HardcodedDroneModels:
    """Parameters from `DroneSpecs.txt` (Examples 1–5 → DroneModel A–E).

    Values marked ``?`` in the spec are set to ``0.0`` with ``# TODO: was ?`` comments.
    ``THRUST2WEIGHT_RATIO`` is not in DroneSpecs; a placeholder is used (TODO).
    Collision capsule uses ``COLLISION_R = prop_radius`` (or ``0.5 * L`` if prop radius was ? / 0).
    Single scalar ``drag coeff`` from the spec is applied to ``[d, d, d]`` (URDF often uses separate xy/z).
    """

    # TODO: replace with value from identification — not provided in DroneSpecs.txt
    _THRUST2WEIGHT_RATIO = 2.25

    @staticmethod
    def is_hardcoded(drone_model):
        return DroneModelFormat[drone_model.name].value == "hardcode"

    @staticmethod
    def raw_tuple(drone_model):
        """Same tuple shape as ``MultiModelBaseAviary._parseURDFParametersFromURDF`` return."""
        if drone_model == DroneModel.A:
            return HardcodedDroneModels._spec_example_1()
        if drone_model == DroneModel.B:
            return HardcodedDroneModels._spec_example_2()
        if drone_model == DroneModel.C:
            return HardcodedDroneModels._spec_example_3()
        if drone_model == DroneModel.D:
            return HardcodedDroneModels._spec_example_4()
        if drone_model == DroneModel.E:
            return HardcodedDroneModels._spec_example_5()
        raise ValueError('No hardcoded spec for drone_model={!r}'.format(drone_model))

    @staticmethod
    def _collision_capsule(L, prop_radius):
        if prop_radius > 0:
            r = prop_radius
        else:
            r = max(0.5 * L, 1e-3)  # TODO: prop radius was ? in DroneSpecs.txt — using L-based fallback
        h = 2.0 * r
        zoff = 0.0
        return h, r, zoff

    @staticmethod
    def _assemble(M, L, IXX, IYY, IZZ, KF, KM, MAX_SPEED_KMH, PROP_RADIUS, drag_scalar):
        THRUST2WEIGHT_RATIO = HardcodedDroneModels._THRUST2WEIGHT_RATIO
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET = HardcodedDroneModels._collision_capsule(L, PROP_RADIUS)
        GND_EFF_COEFF = 0.0  # TODO: was ? in DroneSpecs.txt
        DW_COEFF_1 = 0.0  # TODO: was ? in DroneSpecs.txt
        DW_COEFF_2 = 0.0  # TODO: was ? in DroneSpecs.txt
        DW_COEFF_3 = 0.0  # TODO: was ? in DroneSpecs.txt
        # Single aggregate drag from spec; URDF normally splits drag_coeff_xy / drag_coeff_z
        DRAG_COEFF = np.array([drag_scalar, drag_scalar, drag_scalar], dtype=np.float64)
        return (M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET,
                MAX_SPEED_KMH, GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3)

    @staticmethod
    def _spec_example_1():
        """DroneSpecs Example 1 (small quadcopter) → DroneModel.A."""
        M = 0.35
        L = 0.08
        IXX = 2.5e-4
        IYY = 2.5e-4
        IZZ = 4.0e-4
        KF = (4e-8) ** 2 * 7000.0 * 7000.0
        KM = 0.015 * KF
        MAX_SPEED_KMH = 1.9
        PROP_RADIUS = 0.08
        drag_scalar = 0.2
        return HardcodedDroneModels._assemble(M, L, IXX, IYY, IZZ, KF, KM, MAX_SPEED_KMH, PROP_RADIUS, drag_scalar)

    @staticmethod
    def _spec_example_2():
        """DroneSpecs Example 2 (large quadcopter) → DroneModel.B."""
        M = 0.85
        L = 0.18
        IXX = 3.1e-3
        IYY = 3.1e-3
        IZZ = 5.0e-3
        KF = (8e-6) ** 2 * 1000.0 * 1000.0
        KM = 0.008 * KF
        MAX_SPEED_KMH = 1.0
        PROP_RADIUS = 0.17
        drag_scalar = 0.64
        return HardcodedDroneModels._assemble(M, L, IXX, IYY, IZZ, KF, KM, MAX_SPEED_KMH, PROP_RADIUS, drag_scalar)

    @staticmethod
    def _spec_example_3():
        """DroneSpecs Example 3 (mid-size) → DroneModel.C. MMOI z from MMOI line (4e-3), not J bullet."""
        M = 0.55
        L = 0.107
        IXX = 1.5e-3
        IYY = 1.5e-3
        IZZ = 4.0e-3
        KF = (7.5e-7) ** 2 * 4000.0 * 4000.0
        KM = 0.01 * KF
        MAX_SPEED_KMH = 1.5
        PROP_RADIUS = 0.11
        drag_scalar = 0.34
        return HardcodedDroneModels._assemble(M, L, IXX, IYY, IZZ, KF, KM, MAX_SPEED_KMH, PROP_RADIUS, drag_scalar)

    @staticmethod
    def _spec_example_4():
        """DroneSpecs Example 4 → DroneModel.D. J bullet ignored; use MMOI x/y/z lines."""
        M = 0.45
        L = 0.087
        IXX = 2.0e-3
        IYY = 2.0e-3
        IZZ = 3.0e-3
        KF = (4.3e-7) ** 2 * 5000.0 * 5000.0
        KM = 0.012 * KF
        MAX_SPEED_KMH = 1.6
        PROP_RADIUS = 0.12  # TODO: Made this number up
        drag_scalar = 0.34
        return HardcodedDroneModels._assemble(M, L, IXX, IYY, IZZ, KF, KM, MAX_SPEED_KMH, PROP_RADIUS, drag_scalar)

    @staticmethod
    def _spec_example_5():
        """DroneSpecs Example 5 → DroneModel.E."""
        M = 0.65
        L = 0.127
        IXX = 1.5e-3
        IYY = 1.5e-3
        IZZ = 5.0e-3
        KF = (2.3e-6) ** 2 * 3000.0 * 3000.0
        KM = 0.009 * KF
        MAX_SPEED_KMH = 1.3
        PROP_RADIUS = 0.12
        drag_scalar = 0.48
        return HardcodedDroneModels._assemble(M, L, IXX, IYY, IZZ, KF, KM, MAX_SPEED_KMH, PROP_RADIUS, drag_scalar)

################################################################################

class Physics(Enum):
    """Physics implementations enumeration class."""

    PYB = "pyb"                         # Base PyBullet physics update
    DYN = "dyn"                         # Explicit dynamics model
    PYB_GND = "pyb_gnd"                 # PyBullet physics update with ground effect
    PYB_DRAG = "pyb_drag"               # PyBullet physics update with drag
    PYB_DW = "pyb_dw"                   # PyBullet physics update with downwash
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw" # PyBullet physics update with ground effect, drag, and downwash

################################################################################

class ImageType(Enum):
    """Camera capture image type enumeration class."""

    RGB = 0     # Red, green, blue (and alpha)
    DEP = 1     # Depth
    SEG = 2     # Segmentation by object id
    BW = 3      # Black and white

################################################################################

class ActionType(Enum):
    """Action type enumeration class."""
    RPM = "rpm"                 # RPMS
    PID = "pid"                 # PID control
    VEL = "vel"                 # Velocity input (using PID control)
    ONE_D_RPM = "one_d_rpm"     # 1D (identical input to all motors) with RPMs
    ONE_D_PID = "one_d_pid"     # 1D (identical input to all motors) with PID control

################################################################################

class ObservationType(Enum):
    """Observation type enumeration class."""
    KIN = "kin"     # Kinematic information (pose, linear and angular velocities)
    RGB = "rgb"     # RGB camera capture in each drone's POV


################################################################################

class ModelResamplePolicy(Enum):
    """When to resample the active drone model in multi-model environments."""

    EACH_RESET = "each_reset"
    """Sample a new active model on every `reset()`."""

    FIRST_RESET_ONLY = "first_reset_only"
    """Sample once on the first `reset()` after env construction; keep the same model thereafter."""
