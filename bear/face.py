"""Small face parts (eyes, highlight pins, muzzle, nose) + the matching recesses
that are carved into the head.  All CAD-exact (manifold3d) except the muzzle
velvet, which comes from a fine SDF."""
import math
import numpy as np
import trimesh
import manifold3d as mf
from .sdfkit import Program, frame, INTER
from .snap import to_man, to_tri, box, cyl, mat34, revolve
from . import design as D

# ---- nominal sizes
EYE_R = 4.8                 # eye part radius (9.6 mm wide)
DISH_DEPTH = 0.55           # shallow dish depth (design.py _dish)
EYE_DOME_H = 2.55           # dome height above the dish floor  (=> ~2 mm over the face)
HOLE_D, PIN_D, HOLE_DEPTH = 2.0, 1.9, 1.4
HIGHLIGHT_UV = (-1.9, 2.0)  # local offset (viewer-left, up): light from upper-left

MUZ_R = (13.0, 10.5, 10.0)  # right, up, axis radii
MUZ_SINK = 3.5              # ellipsoid centre sits this deep below the face surface


def local_frame_eye(e):
    """Matrix columns (x,y,z) with z = eye axis, x ~ world +x."""
    z = e["a"]
    x = np.array([1.0, 0, 0]) - z * z[0]
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def sphere_cap(base_r, h, seg=64):
    R = (base_r ** 2 + h ** 2) / (2 * h)
    sph = mf.Manifold.sphere(R, seg).translate((0, 0, h - R))
    return sph ^ box(-R, R, -R, R, 0, h + 1), R


def eye_part(P):
    """Black eye, local frame: dome base plane at z=0, plug below to -EYE_RECESS.
    Returns (manifold, info)."""
    cap, R = sphere_cap(EYE_R, EYE_DOME_H)
    plug = cyl(EYE_R, -P.EYE_RECESS, 0.01)
    eye = cap + plug
    u, v = HIGHLIGHT_UV
    rho = math.hypot(u, v)
    zs = EYE_DOME_H - R + math.sqrt(R * R - rho * rho)      # dome surface height at hole centre
    hole = cyl(HOLE_D / 2, zs - HOLE_DEPTH, zs + 1.0).translate((u, v, 0))
    eye = eye - hole
    return eye, dict(R=R, hole_uv=(u, v), hole_z=zs - HOLE_DEPTH, surf_z=zs)


def highlight_pin(info):
    return cyl(PIN_D / 2, 0, HOLE_DEPTH)


# ---------------------------------------------------------------- muzzle
def muz_matrix(muz):
    a = muz["a"]
    up = muz["up"]
    right = np.cross(up, a) * -1.0
    # make sure right is world +x
    if right[0] < 0:
        right = -right
    return np.stack([right, np.cross(a, right), a], 1), right, np.cross(a, right), a


def muz_surface_height(muz, uv):
    """height (along a, relative to muzzle S) of the muzzle ellipsoid at local (u,v)."""
    u, v = uv
    t = 1 - (u / MUZ_R[0]) ** 2 - (v / MUZ_R[1]) ** 2
    return -MUZ_SINK + MUZ_R[2] * math.sqrt(max(t, 0))


def muzzle_floor(head_prog, muz, P):
    """Flat floor offset (along a, relative to S): 1 mm below the lowest
    point where the muzzle ellipsoid meets the head surface."""
    _, right, upv, a = muz_matrix(muz)
    S = muz["S"]
    th = np.linspace(0, math.pi, 90)
    ph = np.linspace(0, 2 * math.pi, 180)
    TH, PH = np.meshgrid(th, ph)
    loc = np.stack([MUZ_R[0] * np.sin(TH) * np.cos(PH), MUZ_R[1] * np.sin(TH) * np.sin(PH),
                    -MUZ_SINK + MUZ_R[2] * np.cos(TH)], -1).reshape(-1, 3)
    W = S + loc[:, 0:1] * right + loc[:, 1:2] * upv + loc[:, 2:3] * a
    d = head_prog.eval(W)
    sel = np.abs(d) < 0.12
    s = loc[sel, 2]
    return float(s.min()) - P.MUZZLE_RECESS


def muzzle_sdf(muz, floor):
    Rm, right, upv, a = muz_matrix(muz)
    S = muz["S"]
    p = Program()
    M = np.stack([right, upv, a])           # rows = world->local
    p.ell(S - a * MUZ_SINK, MUZ_R, M=M, k=0)
    p.plane(-a, float(-(a @ S) - floor), op=INTER, k=0)    # keep a.(p-S) >= floor
    return p


def muz_local_to_world(muz, u, v, s):
    Rm, right, upv, a = muz_matrix(muz)
    return muz["S"] + right * u + upv * v + a * s


def rounded_triangle(w, h, r, apex_down=True, seg=32):
    """2D rounded inverted triangle: top edge width w at y=+h, apex at y=0."""
    tri = mf.CrossSection([[(0.0, 0.0), (w / 2, h), (-w / 2, h)]])
    # shrink then grow with round joins -> rounded corners
    return tri.offset(-r, mf.JoinType.Round, 2.0, seg).offset(r, mf.JoinType.Round, 2.0, seg)


NOSE_W, NOSE_H = 9.6, 7.2
NOSE_V0 = 0.9            # apex position (local v on muzzle)
NOSE_APEX = 2.7          # dome apex above the muzzle surface
_MUZ_PROG = None


def muz_prog_local():
    global _MUZ_PROG
    if _MUZ_PROG is None:
        _MUZ_PROG = Program().ell((0, 0, -MUZ_SINK), MUZ_R, k=0)
    return _MUZ_PROG


def nose_footprint():
    """rounded inverted triangle with its centroid at the origin (nose-local x,y)."""
    cs = rounded_triangle(NOSE_W * 1.35, NOSE_H * 1.3, 1.9)
    x0, y0, x1, y1 = cs.bounds()
    cs = cs.translate((-(x0 + x1) / 2, -y0)).scale((NOSE_W / (x1 - x0), NOSE_H / (y1 - y0)))
    return cs.translate((0.0, -2.0 * NOSE_H / 3.0))


def nose_prism(clr, z0, z1):
    cs = nose_footprint()
    if clr:
        cs = cs.offset(clr, mf.JoinType.Round, 2.0, 24)
    return mf.Manifold.extrude(cs, z1 - z0).translate((0, 0, z0))


def nose_frame(P):
    """T (3x4, nose-local -> muzzle-local), floor z (nose-local) and prism data."""
    mp = muz_prog_local()
    vc = NOSE_V0 + 2.0 * NOSE_H / 3.0
    q = mp.raymarch((0, vc, 40.0), (0, 0, -1))
    n = mp.grad(q[None])[0]
    n /= np.linalg.norm(n)
    x = np.array([1.0, 0, 0])
    y = np.cross(n, x)
    Rn = np.stack([x, y, n], 1)
    T = mat34(Rn, q)
    pts = [p for poly in nose_footprint().to_polygons() for p in poly]
    hs = []
    for (px, py) in pts:
        o = q + Rn @ np.array([px, py, 20.0])
        hit = mp.raymarch(o, -n)
        hs.append((Rn.T @ (hit - q))[2] if hit is not None else 0.0)
    return dict(T=T, floor=min(hs) - P.NOSE_RECESS)


def mouth_curve():
    L = [(0, NOSE_V0 - 0.2), (0, -1.4), (0, -3.2), (-2.2, -4.0), (-4.6, -4.2), (-6.4, -3.8)]
    R = [(0, -3.2), (2.2, -4.0), (4.8, -4.0), (6.7, -3.2)]
    return [L, R]


def segment_tool(p0, p1, r, seg=16):
    s0 = mf.Manifold.sphere(r, seg).translate(tuple(p0))
    s1 = mf.Manifold.sphere(r, seg).translate(tuple(p1))
    return mf.Manifold.batch_hull([s0, s1])


def muzzle_features(muz, P):
    """Return dict of manifold tools in muzzle-local (u,v,a) coordinates, positioned
    in world via matrix transform later."""
    return None


def build_muzzle_parts(head_prog, muz, P, clr):
    """Returns dict with muzzle mesh pieces & head-side cutters (world coords)."""
    floor = muzzle_floor(head_prog, muz, P)
    Rm, right, upv, a = muz_matrix(muz)
    S = muz["S"]
    Tm = mat34(Rm, S)                                  # muzzle-local -> world

    nf = nose_frame(P)
    return dict(floor=floor, T=Tm, nose=nf)


def muzzle_ellipsoid_man(grow=0.0):
    sph = mf.Manifold.sphere(1.0, 96)
    sc = np.diag([MUZ_R[0] + grow, MUZ_R[1] + grow, MUZ_R[2] + grow])
    return sph.transform(mat34(sc, (0, 0, -MUZ_SINK)))


PEG_R, PEG_L, PEG_UV = 2.0, 1.8, (0.0, 5.0)


def head_cutters(face, head_prog, P, info):
    """Manifolds (world) to subtract from the body: eye recesses, muzzle recess, peg hole."""
    cut = []
    for s in D.S2:
        e = face["eyes"][s]
        Rm = local_frame_eye(e)
        floor = -(DISH_DEPTH + P.EYE_RECESS)
        c = cyl(EYE_R + P.FACE_CLEARANCE, floor, 6.0)
        cut.append(c.transform(mat34(Rm, e["S"])))
    muz = face["muz"]
    floor = info["floor"]
    Rm = info["T"][:, :3]
    ell = muzzle_ellipsoid_man(P.FACE_CLEARANCE) ^ box(-30, 30, -30, 30, floor, 30)
    cut.append(ell.transform(info["T"]))
    return cut


def head_pegs(info, P):
    """Locating post standing on the muzzle-recess floor (muzzle back has the matching hole)."""
    floor = info["floor"]
    peg = cyl(PEG_R, floor - 0.8, floor + PEG_L).translate((PEG_UV[0], PEG_UV[1], 0))
    return [peg.transform(info["T"])]


def muzzle_final(mesh_tm, info, P):
    """Take the velvet muzzle mesh (world, from SDF) -> add peg, carve nose recess + mouth.
    Returns trimesh in LOCAL muzzle coords (floor plane at z = 0, face up) and the
    transform to place it back (local->world 3x4)."""
    T = info["T"]
    floor = info["floor"]
    m = to_man(mesh_tm)
    # to local coordinates
    Rm = T[:, :3]
    inv = np.zeros((3, 4))
    inv[:, :3] = Rm.T
    inv[:, 3] = -Rm.T @ T[:, 3]
    m = m.transform(inv)
    # locating hole in the flat back (matching post stands on the head recess floor)
    hole = cyl(PEG_R + P.FACE_CLEARANCE, floor - 0.2, floor + PEG_L + 0.4).translate((PEG_UV[0], PEG_UV[1], 0))
    m = m - hole
    # nose recess
    nf = info["nose"]
    m = m - nose_prism(P.FACE_CLEARANCE, nf["floor"], nf["floor"] + 12).transform(nf["T"])
    # mouth + philtrum grooves
    r = 0.55
    for poly in mouth_curve():
        for p0, p1 in zip(poly[:-1], poly[1:]):
            q0 = (p0[0], p0[1], muz_surface_height(None, p0) + r - 0.62)
            q1 = (p1[0], p1[1], muz_surface_height(None, p1) + r - 0.62)
            m = m - segment_tool(q0, q1, r)
    # shift so that the floor is z = 0 (print orientation: face up)
    m = m.translate((0, 0, -floor))
    return m, floor


def nose_part(info, P):
    nf = info["nose"]
    fl = nf["floor"]
    prism = nose_prism(0.0, fl, NOSE_APEX + 2.0)
    zc = fl + 0.8
    dome = mf.Manifold.sphere(1.0, 64).transform(mat34(np.diag([6.6, 6.0, NOSE_APEX - zc]), (0, 0, zc)))
    nose = prism ^ dome
    return nose.translate((0, 0, -fl))            # floor at z = 0


