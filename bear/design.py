"""SDF design of every part of the boxing teddy bear (assembled coordinates)."""
import math
import numpy as np
from .sdfkit import Program, frame, rot_axes, UNION, SUB, INTER

Z_WAIST = 80.0          # waistband top == torso cut plane
Z_HEM = 45.0            # shorts bottom == leg top plane
LEG_X = 20.0
FOOT_TOE_DEG = 12.0
HEAD_C = np.array([0.0, -4.0, 165.5])
S2 = (1, -1)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------- geometry
def arm_points(s):
    S = np.array([s * 47.0, 0.0, 123.0])
    E = np.array([s * 54.0, -2.0, 105.0])
    W = np.array([s * 58.5, -6.0, 90.5])
    g = unit(W - E)
    return S, E, W, g


def glove_frame(s):
    S, E, W, g = arm_points(s)
    M = frame(g, hint=(0, -1, 0))      # rows: x(front), y, z=g
    return W, M


def surf_pts(ref, rays):
    pts = []
    for o, d in rays:
        q = ref.raymarch(o, d)
        if q is not None:
            pts.append(q)
    return np.array(pts)


def carve_polyline(p, ref, pts, r, depth, k=0.5):
    """Subtract a capsule chain whose centres sit `depth` below the surface
    sampled at pts (points assumed on the ref surface)."""
    if len(pts) < 2:
        return
    n = ref.grad(pts)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    c = pts + n * (r - depth)
    for a, b in zip(c[:-1], c[1:]):
        p.cap(a, b, r, r, op=SUB, k=k)


def carve_tapered(p, ref, pts, rs, depths, k=1.2):
    """Groove along pts with per-point radius / depth (tapers to nothing at the ends)."""
    n = ref.grad(pts)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    rs = np.asarray(rs, float)
    depths = np.asarray(depths, float)
    c = pts + n * (rs - depths)[:, None]
    for i in range(len(pts) - 1):
        p.cap(c[i], c[i + 1], rs[i], rs[i + 1], op=SUB, k=k)


def add_polyline(p, ref, pts, r, height, k=2.0):
    n = ref.grad(pts)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    c = pts + n * (r - height)
    for a, b in zip(c[:-1], c[1:]):
        p.cap(a, b, r, r, op=UNION, k=k)


# ------------------------------------------------------------------- head
def add_head(p):
    p.ell(HEAD_C, (40.0, 34.5, 30.0), k=3)
    for s in S2:
        p.ell((s * 26.5, -13.0, 157.0), (15.0, 19.0, 13.5), k=9)   # plush cheeks
    p.ell((0, -15.0, 147.5), (21.0, 17.0, 9.5), k=8)                # chin / jaw
    p.ell((0, 6.0, 161.0), (35.0, 26.0, 26.0), k=6)                 # back of head
    for s in S2:
        EC, n_e = ear_geom(s)
        p.ell(EC, EAR_R, M=frame(n_e, hint=(0, 0, 1)), k=2.5)
    return p


EAR_R = (12.5, 12.5, 5.6)
EAR_TOP = 199.5


def ear_geom(s):
    n_e = unit((s * 0.30, -0.88, 0.37))
    M = frame(n_e, hint=(0, 0, 1))
    ext = math.sqrt(sum((EAR_R[i] * M[i, 2]) ** 2 for i in range(3)))   # z extent
    EC = np.array([s * 28.5, 3.5, EAR_TOP - ext])
    return EC, n_e


def face_layout(head):
    """Locate eyes, brows, muzzle, nose, mouth on the head surface."""
    L = {}
    zfront = (0, 1, 0)
    # eyes
    eyes = {}
    for s in S2:
        x, z = s * 14.5, 169.0
        q = head.raymarch((x, -120, z), zfront)
        n = head.grad(q[None])[0]
        n = unit(n)
        a = unit(0.40 * n + 0.60 * np.array([0, -1.0, 0]))   # outward axis
        q = head.raymarch(q + a * 30, -a)
        eyes[s] = dict(S=q, a=a)
    L["eyes"] = eyes
    # muzzle
    q = head.raymarch((0, -120, 157.0), zfront)
    n = unit(head.grad(q[None])[0])
    n[0] = 0
    a = unit(n)
    q = head.raymarch(q + a * 30, -a)
    up = unit(np.array([0, 0, 1.0]) - a * a[2])
    L["muz"] = dict(S=q, a=a, up=up, right=np.cross(up, a) * -1)
    # brows (polyline on surface)
    brows = {}
    for s in S2:
        xs = np.linspace(25.5, 5.5, 8)
        zs = 177.2 + (xs - 5.5) / 20.0 * 4.2
        pts = surf_pts(head, [((s * x, -120, z), zfront) for x, z in zip(xs, zs)])
        brows[s] = pts
    L["brows"] = brows
    xs0 = np.zeros(6)
    zs0 = np.linspace(175.5, 183.0, 6)
    L["crease"] = surf_pts(head, [((0, -120, z), zfront) for z in zs0])
    return L


# ------------------------------------------------------------------- body
def body_program(face):
    """Head + upper torso + arms down to the wrist cut planes."""
    p = Program()
    add_head(p)
    # ------- torso masses
    p.ell((0, 0, 138.0), (24, 16, 10), k=6)                         # trap mass / neck
    p.ell((0, 0, 113.0), (39.5, 27.0, 27.5), k=6)                   # ribcage
    p.ell((0, -1, 90.0), (31.0, 22.5, 16.0), k=6)                   # abdomen
    for s in S2:
        p.ell((s * 31.5, 9.5, 108.0), (12.0, 13.0, 23.0), M=rot_axes(0, 0, s * 9), k=5)   # lats
        p.ell((s * 29.5, -5.0, 90.0), (7.5, 15.0, 13.0), M=rot_axes(0, 0, -s * 10), k=4)  # obliques
        p.ell((s * 18.0, -18.0, 117.5), (18.5, 10.0, 12.5), M=rot_axes(0, -s * 7, 0), k=2.5)  # pecs
        p.cap((s * 6, 4, 142), (s * 37, 1, 128), 9.5, 8.0, k=6)     # trapezius
        p.ell((s * 17, 19.5, 120.0), (12.5, 7.0, 11.0), M=rot_axes(0, s * 12, 0), k=4)  # shoulder blades
        p.cap((s * 6.0, 19.0, 83), (s * 7.0, 21.0, 108), 5.0, 5.5, k=4)   # erector ridges
        # abs: 3 rows x 2
        for z in (100.5, 91.7, 83.0):
            p.rbox((s * 9.4, -19.6, z), (8.3, 4.2, 4.1), 3.0, k=1.8)
    # arms
    for s in S2:
        S, E, W, g = arm_points(s)
        p.ell(S + np.array([-s * 1.5, 0.0, 0]), (16.0, 16.0, 15.5), k=5)         # deltoid
        p.cap(S, E, 13.0, 11.8, k=4)                                             # upper arm
        p.ell((s * 50.5, -6.5, 113.5), (9.5, 9.5, 12.5), M=rot_axes(0, s * 4, 0), k=3)   # biceps
        p.ell((s * 52.5, 6.5, 112.5), (9.0, 9.0, 13.5), M=rot_axes(0, s * 4, 0), k=3)    # triceps
        p.ell(E, (12.0, 12.0, 11.5), k=3)                                        # elbow
        p.cap(E, W, 12.2, 10.4, k=3)                                              # forearm
        p.ell(E * 0.45 + W * 0.55 + np.array([s * 0.5, -1.0, 0]), (11.6, 11.3, 12.5),
              M=frame(g, hint=(1, 0, 0)).T.T, k=3)                               # forearm belly
    # ------- cut planes
    p.plane((0, 0, -1), -Z_WAIST, op=INTER, k=0)       # keep z >= waist
    for s in S2:
        S, E, W, g = arm_points(s)
        p.plane(g, float(g @ W), op=INTER, k=0)       # keep points before wrist plane
    ref = Program()
    ref.rows = list(p.rows)
    # ------- grooves & facial sculpting (subtract / add)
    front = (0, 1, 0)
    back = (0, -1, 0)
    # linea alba
    pts = surf_pts(ref, [((0, -120, z), front) for z in np.arange(80.5, 106.5, 3.0)])
    carve_polyline(p, ref, pts, 1.0, 0.9, k=0.8)
    # ab rows
    for zg in (96.0, 87.2):
        xs = np.linspace(-17, 17, 10)
        pts = surf_pts(ref, [((x, -120, zg + 0.012 * x * x - 1.5), front) for x in xs])
        carve_polyline(p, ref, pts, 1.0, 1.1, k=0.6)
    # pec centre + lower pec line
    pts = surf_pts(ref, [((0, -120, z), front) for z in np.arange(106.0, 128.0, 3.0)])
    carve_polyline(p, ref, pts, 1.0, 1.2, k=0.6)
    for s in S2:
        xs = np.linspace(2.5, 33, 10)
        pts = surf_pts(ref, [((s * x, -120, 105.2 + 0.0055 * x * x), front) for x in xs])
        carve_polyline(p, ref, pts, 1.0, 1.2, k=0.8)
        # oblique / serratus lines
        for i, z0 in enumerate((92, 99)):
            pts = surf_pts(ref, [((s * (25 + 2.5 * t), -120, z0 + 3.0 * t), front) for t in range(5)])
            carve_polyline(p, ref, pts, 0.8, 0.8, k=0.6)
    # spine groove + scapula/trap lines
    pts = surf_pts(ref, [((0, 120, z), back) for z in np.arange(82, 140, 3.0)])
    carve_polyline(p, ref, pts, 1.3, 1.4, k=0.8)
    for s in S2:
        # scapula lower edge + lat edge grooves
        xs = np.linspace(6, 27, 7)
        pts = surf_pts(ref, [((s * x, 120, 113.0 + 0.30 * (x - 6)), back) for x in xs])
        carve_tapered(p, ref, pts, [0.6, 1.0, 1.1, 1.1, 1.1, 1.0, 0.6], [0.0, 0.8, 1.0, 1.0, 1.0, 0.8, 0.0], k=0.8)
        zs = np.linspace(106, 84, 7)
        pts = surf_pts(ref, [((s * (38.5 - 0.55 * (106 - z)), 120, z), back) for z in zs])
        carve_tapered(p, ref, pts, [0.6, 1.0, 1.2, 1.2, 1.1, 1.0, 0.6], [0.0, 0.7, 0.9, 0.9, 0.9, 0.6, 0.0], k=0.8)
    # deltoid / arm separation grooves
    for s in S2:
        S, E, W, g = arm_points(s)
        for z0, zr in ((117.0, 4.0),):
            pts = surf_pts(ref, [((s * (x), -120, z0 - 0.5 * (x - 36)), front) for x in np.linspace(37, 55, 6)])
        # biceps/triceps split
    # ears: inner bowl
    for s in S2:
        EC, n_e = ear_geom(s)
        Rb = 14.0
        p.ell(EC + n_e * (EAR_R[2] + Rb - 2.7), (Rb, Rb, Rb), op=SUB, k=1.3)
    # brows
    for s in S2:
        pts = face["brows"][s]
        # slightly sunk capsule chain gives a soft ridge
        n = ref.grad(pts)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        c = pts - n * 0.5
        for a, b in zip(c[:-1], c[1:]):
            p.cap(a, b, 3.3, 3.0, k=2.0)
    pts = face["crease"]
    n = ref.grad(pts)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    c = pts + n * 0.1
    for a, b in zip(c[:-1], c[1:]):
        p.cap(a, b, 0.9, 0.9, op=SUB, k=0.8)
    # eye dishes (shallow, smooth)  -- region carve: cylinder AND outside offset(-0.5)
    for s in S2:
        e = face["eyes"][s]
        _dish(p, e["S"], e["a"], 7.3, 0.55)
    return p


def _dish(p, S, a, radius, depth):
    """Shallow circular recess following the surface (uses current program as ref)."""
    # implemented through a thin rounded cylinder subtracted at the surface:
    # a flat-ish ellipsoid disc whose lower face sits `depth` below the surface at its centre
    M = frame(a, hint=(0, 0, 1))
    c = S + a * (3.0 - depth)       # disc centre: 3 mm thick above, depth below
    p.ecyl(c, radius, radius, 3.0, 0.9, op=SUB, k=0.6, M=M)


# -------------------------------------------------------------------- legs
def foot_frame(s):
    th = s * FOOT_TOE_DEG
    M = rot_axes(0, 0, th)               # world->local rows
    pivot = np.array([s * LEG_X, 0.0, 0.0])
    return pivot, M


def leg_program(s):
    p = Program()
    pivot, M = foot_frame(s)
    R = M.T                               # local->world

    def L(v):                              # local offset -> world
        return pivot + R @ np.asarray(v, float)

    # foot
    p.rbox(L((0, -7.5, 7.0)), (15.8, 22.8, 8.0), 7.0, M=M, k=3)
    p.ell(L((0, 6.0, 11.5)), (13.0, 11.5, 8.5), M=M, k=5)                   # heel / ankle
    toes_x = (-11.6, -3.9, 3.9, 11.6)
    toes_y = (-23.0, -25.0, -25.0, -23.0)
    for tx, ty in zip(toes_x, toes_y):
        p.ell(L((tx, ty + 2.0, 6.5)), (4.7, 6.2, 5.6), M=M, k=1.5)
    # leg
    p.cap((s * LEG_X, 2.0, 13.5), (s * LEG_X, 0.5, 47.0), 14.0, 17.2, k=4)
    p.ell((s * LEG_X, 6.0, 28.0), (13.5, 11.5, 13.5), k=4)                   # calf
    p.ell((s * LEG_X, -7.0, 37.0), (11.0, 7.0, 8.5), k=4)                   # thigh front
    p.plane((0, 0, -1), 0.0, op=INTER, k=0)                                 # flat sole z>=0
    p.plane((0, 0, 1), Z_HEM, op=INTER, k=0)                                # leg top
    ref = Program()
    ref.rows = list(p.rows)
    # toe grooves (3)
    for tx in (-7.75, 0.0, 7.75):
        a = L((tx, -34, 12.5))
        b = L((tx, -17, 12.8))
        c = L((tx, -22, 4.0))
        p.cap(a, b, 0.85, 0.85, op=SUB, k=0.5)
        p.cap(L((tx, -31, 12.0)), L((tx, -29, 2.5)), 0.85, 0.85, op=SUB, k=0.5)
    return p


# ------------------------------------------------------------------ shorts
def shorts_program():
    p = Program()
    # hip block + leg tubes
    p.ecyl((0, 0, 62.5), 34.5, 25.5, 13.0, 6.0, k=3)
    for s in S2:
        p.cap((s * LEG_X, 0.5, 72.0), (s * LEG_X, 0.5, 47.0), 19.4, 20.4, k=5)
    # waistband (ribbed)
    p.ecyl((0, 0, 75.5), 35.5, 26.5, 4.5, 2.0, k=1.0)
    p.plane((0, 0, 1), Z_WAIST, op=INTER, k=0)       # keep z <= waist top
    p.plane((0, 0, -1), -Z_HEM, op=INTER, k=0)       # keep z >= hem plane
    ref = Program()
    ref.rows = list(p.rows)
    # rolled hems
    for s in S2:
        p.ering((s * LEG_X, 0.5, Z_HEM + 1.3), 20.2, 20.2, 1.35, k=0.8)
    # waistband rib grooves
    for z in (74.0, 77.0):
        p.ering((0, 0, z), 35.6, 26.6, 0.55, op=SUB, k=0.2)
    # fabric gathers under the waistband (soft, tapered)
    rng = np.random.default_rng(3)
    angs = np.linspace(0, 2 * math.pi, 18, endpoint=False) + rng.normal(0, 0.07, 18)
    for th in angs:
        d = (-math.cos(th), -math.sin(th), 0)
        zs = [70.6, 67.6, 64.0 - rng.uniform(0, 2.5)]
        pts = []
        for z in zs:
            o = np.array([math.cos(th) * 80, math.sin(th) * 60, z])
            q = ref.raymarch(o, d)
            if q is not None:
                pts.append(q)
        if len(pts) == 3:
            carve_tapered(p, ref, np.array(pts), [0.7, 1.7, 1.0], [0.0, 0.42, 0.0], k=1.8)
    for s in S2:
        # outer-side hem slit
        q = ref.raymarch((s * 80, 0.5, 49.0), (-s, 0, 0))
        if q is not None:
            p.rbox((q[0], q[1], Z_HEM + 3.5), (3.5, 0.65, 4.8), 0.3, op=SUB, k=0.1)
    p.plane((0, 0, 1), Z_WAIST, op=INTER, k=0)
    p.plane((0, 0, -1), -Z_HEM, op=INTER, k=0)
    return p


# ------------------------------------------------------------------ gloves
def glove_program(s):
    W, M = glove_frame(s)
    R = M.T

    def Lg(v):
        return W + R @ np.asarray(v, float)

    p = Program()
    ex, ey, g = M[0], M[1], M[2]
    inner = s                                   # local-y sign towards the body
    # cuff
    p.cap(Lg((0, 0, -3.0)), Lg((0, 0, 13.5)), 12.6, 12.6, k=1.0)
    # body (puffy)
    p.rbox(Lg((0.5, 0, 27.0)), (13.8, 15.2, 13.0), 8.5, M=M, k=4)
    p.ell(Lg((1.5, 0, 26.5)), (15.0, 16.0, 14.2), M=M, k=4)
    p.ell(Lg((5.0, 0, 31.0)), (11.0, 14.0, 9.5), M=M, k=3)                  # knuckle bulge
    # thumb: capsule hugging the body on the inner-front side
    ta, tb = Lg((6.0, inner * 12.2, 17.5)), Lg((7.5, inner * 10.5, 33.0))
    p.cap(ta, tb, 5.4, 5.0, k=1.5)
    p.plane(-g, float(-g @ W), op=INTER, k=0)          # flat top (wrist) plane
    ref = Program()
    ref.rows = list(p.rows)
    # cuff / body seam ring
    p.torus(Lg((0, 0, 13.0)), 12.7, 0.6, M=M, op=SUB, k=0.2)
    # thumb seam: polyline around thumb outline on glove body
    t = np.linspace(0, 1, 9)
    axis_pts = np.array([ta + (tb - ta) * ti for ti in t])
    side = ex * 1.0
    seam = []
    for q in axis_pts:
        o = q + ex * 14.0 + ey * inner * 0.0
        # march from outside toward the thumb axis
        r = ref.raymarch(q + ex * 16, -ex)
        if r is not None:
            seam.append(r)
    if len(seam) > 1:
        carve_polyline(p, ref, np.array(seam), 0.55, 0.5, k=0.3)
    # panel seams (top/side)
    for sgn in (1, -1):
        pts = []
        for th in np.linspace(-0.3, 1.2, 8):
            o = Lg((14 * math.cos(th) + 2, sgn * 18 * math.sin(th), 40.0 + 6))
            q = ref.raymarch(Lg((2 + 30 * math.cos(th), sgn * 30 * math.sin(th) * 1.0, 33.0 + 3 * th)),
                             -(ex * math.cos(th) + ey * sgn * math.sin(th)))
            if q is not None:
                pts.append(q)
        if len(pts) > 1:
            carve_polyline(p, ref, np.array(pts), 0.5, 0.45, k=0.3)
    # compression wrinkles near knuckles
    for i, off in enumerate((-8, -2.5, 3.0, 8.5)):
        q1 = ref.raymarch(Lg((20, off * 1.4, 33.5)), -ex)
        q2 = ref.raymarch(Lg((20, off * 1.4 + 0.8, 40.0)), -ex)
        if q1 is not None and q2 is not None:
            carve_polyline(p, ref, np.array([q1, q2]), 0.7, 0.4, k=0.5)
    for yy in (-6.0, 6.0):
        q1 = ref.raymarch(Lg((20, yy, 16.0)), -ex)
        q2 = ref.raymarch(Lg((20, yy * 1.6, 21.0)), -ex)
        if q1 is not None and q2 is not None:
            carve_polyline(p, ref, np.array([q1, q2]), 0.7, 0.4, k=0.5)
    return p


# -------------------------------------------------------------------- base
def base_program(diam, thick):
    p = Program()
    p.ecyl((0, 0, -thick / 2), diam / 2, diam / 2, thick / 2, 2.4, k=1.0)
    return p
