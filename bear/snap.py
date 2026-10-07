"""Exact (CAD-like) annular split snap pins + matching sockets, built with
manifold3d and applied to the meshes by boolean operations."""
import math
from dataclasses import dataclass
import numpy as np
import trimesh
import manifold3d as mf

SEG = 72


def to_man(tm):
    return mf.Manifold(mf.Mesh(np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.uint32)))


def to_tri(man):
    me = man.to_mesh()
    return trimesh.Trimesh(np.asarray(me.vert_properties[:, :3], float), np.asarray(me.tri_verts), process=False)


def revolve(profile, seg=SEG):
    cs = mf.CrossSection([[tuple(map(float, p)) for p in profile]])
    return mf.Manifold.revolve(cs, seg)


def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube((x1 - x0, y1 - y0, z1 - z0), False).translate((x0, y0, z0))


def cyl(r, z0, z1, seg=SEG):
    return mf.Manifold.cylinder(z1 - z0, r, r, seg).translate((0, 0, z0))


def mat34(R, t):
    M = np.zeros((3, 4))
    M[:, :3] = R
    M[:, 3] = t
    return M


@dataclass
class Joint:
    name: str
    O: np.ndarray            # centre of the mating plane (world)
    a: np.ndarray            # axis, from the pin-bearing part INTO the socket part
    flat: np.ndarray         # world direction of the anti-rotation flat
    D: float                 # pin diameter
    L: float                 # pin length (beyond the tongue)
    tongue_r: float = 0.0    # collar radius (0 = none)
    tongue_L: float = 0.0
    fillet: float = 1.0

    def R(self):
        z = np.asarray(self.a, float)
        z = z / np.linalg.norm(z)
        x = np.asarray(self.flat, float)
        x = x - z * (x @ z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.stack([x, y, z], 1)      # columns = local axes in world

    def T(self):
        return mat34(self.R(), np.asarray(self.O, float))


def barb_geometry(P, r, z0, L):
    """returns z_bm (barb back face), z_top (end of lead-in)"""
    z_bm = z0 + L - 1.9
    z_top = z_bm + 0.25 + P.BARB / math.tan(math.radians(P.BARB_CHAMFER_DEG))
    return z_bm, z_top


def pin_solid(j: Joint, P):
    """Pin (+ collar/tongue) in the joint's local frame (z into the socket part)."""
    r = j.D / 2
    b = P.BARB
    Lc = j.tongue_L if j.tongue_r > 0 else 0.0
    f = j.fillet
    prof = [(0, Lc - 0.8), (r + f, Lc - 0.8), (r + f, Lc)]
    for th in np.linspace(-90, -180, 7)[1:]:
        t = math.radians(th)
        prof.append((r + f + f * math.cos(t), Lc + f + f * math.sin(t)))
    z_bm, z_top = barb_geometry(P, r, Lc, j.L)
    tipz = Lc + j.L
    prof += [(r, z_bm), (r + b, z_bm), (r + b, z_bm + 0.25), (r + 0.02, z_top),
             (r, tipz - 0.5), (r - 0.5, tipz), (0, tipz)]
    pin = revolve(prof)
    # anti-rotation flat
    xf = r - P.FLAT_DEPTH
    pin = pin ^ box(-r - 3, xf, -r - 3, r + 3, Lc - 1.0, tipz + 1)
    # four prongs: two crossed slots, 70 % of the length
    sl = P.SLOT_FRAC * j.L
    w = P.SLOT_W
    big = r + b + 2
    slots = box(-big, big, -w / 2, w / 2, tipz - sl, tipz + 1) + box(-w / 2, w / 2, -big, big, tipz - sl, tipz + 1)
    pin = pin - slots
    if j.tongue_r > 0:
        pin = pin + cyl(j.tongue_r, -1.5, Lc + 0.2)
    else:
        pin = pin + cyl(r * 0.9, -1.5, 0.2)
    return pin


def socket_void(j: Joint, P, clr):
    """Void to subtract from the receiving part, local frame."""
    r = j.D / 2
    b = P.BARB
    Lc = j.tongue_L if j.tongue_r > 0 else 0.0
    zf = Lc + 0.25 if j.tongue_r > 0 else 0.0
    rh = r + clr
    rg = r + b + clr
    z_bm, z_top = barb_geometry(P, r, Lc, j.L)
    z_w = z_bm - 0.05
    bottom = Lc + j.L + P.SOCKET_EXTRA
    ent = 0.5
    prof = [(0, zf - 0.2), (rh + ent + 0.2, zf - 0.2), (rh, zf + ent), (rh, z_w), (rg, z_w), (rg, z_top),
            (rh, z_top + (rg - rh)), (rh, bottom), (0, bottom)]
    hole = revolve(prof)
    xf = r - P.FLAT_DEPTH + clr
    hole = hole ^ box(-r - 5, xf, -r - 5, r + 5, zf - 1, bottom + 1)
    # relief ring so the pin-base fillet clears the entrance
    hole = hole + cyl(r + j.fillet + clr, zf - 0.3, zf + j.fillet + 0.05)
    if j.tongue_r > 0:
        hole = hole + cyl(j.tongue_r + clr, -0.3, Lc + 0.25)
    return hole


def add_pin(tm, j: Joint, P):
    """Union the pin onto trimesh `tm`."""
    pin = pin_solid(j, P).transform(j.T())
    return to_tri(to_man(tm) + pin)


def cut_socket(tm, j: Joint, P, clr):
    void = socket_void(j, P, clr).transform(j.T())
    return to_tri(to_man(tm) - void)
