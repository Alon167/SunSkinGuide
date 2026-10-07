"""Global, user-tunable parameters.  Everything is in millimetres.

World frame: +X = figure's LEFT (viewer's right), -Y = front, +Z = up,
feet soles at z = 0.
"""
from dataclasses import dataclass, field


@dataclass
class Params:
    # ---- global knobs requested in the brief
    HEIGHT: float = 200.0            # total figure height (ears top), mm
    FUR_STRENGTH: float = 1.0        # 0 = smooth, 1 = default, >1 = more
    CLEARANCE: float = 0.15          # per-side snap clearance (the 3 folders override)
    WITH_BASE: bool = True
    BASE_DIAMETER: float = 96.0      # computed up if the feet need more room
    BASE_THICK: float = 6.0

    # ---- voxel sizes
    VOXEL_FUR: float = 0.25
    VOXEL_SMOOTH: float = 0.30
    VOXEL_FACE: float = 0.12

    # ---- snap pins  (diameter, length)
    TORSO_PIN: tuple = (12.0, 9.0)
    LEG_PIN: tuple = (10.0, 7.0)
    GLOVE_PIN: tuple = (8.0, 6.0)
    BASE_PIN: tuple = (10.0, 6.0)
    BARB: float = 0.40               # barb overhang
    BARB_CHAMFER_DEG: float = 35.0   # lead-in angle
    SLOT_W: float = 1.3
    SLOT_FRAC: float = 0.70
    PIN_FILLET: float = 1.0
    BASE_PIN_FILLET: float = 1.5
    SOCKET_EXTRA: float = 0.5        # socket deeper than pin
    FLAT_DEPTH: float = 1.0          # anti-rotation flat depth
    TONGUE_LEN: float = 4.0          # wrist/leg overlap length

    # ---- fur
    BUMP_DIAM: tuple = (0.9, 1.4)
    BUMP_HEIGHT: tuple = (0.35, 0.60)
    FUR_ELONGATION: float = 1.3
    FUR_NOISE_AMP: float = 0.15
    JOINT_SMOOTH_MM: float = 2.0

    # ---- face part fits
    FACE_CLEARANCE: float = 0.10
    MUZZLE_RECESS: float = 1.0
    NOSE_RECESS: float = 0.8
    EYE_RECESS: float = 1.5

    seed: int = 7
    preview: bool = False            # coarse & fast


CLEARANCES = (0.10, 0.15, 0.20)

# colours (sRGB 0-255)
COLORS = {
    "brown": (176, 106, 52),
    "cream": (236, 208, 170),
    "black": (24, 24, 26),
    "white": (245, 245, 245),
    "red": (200, 22, 24),
    "gray": (150, 150, 155),
}
