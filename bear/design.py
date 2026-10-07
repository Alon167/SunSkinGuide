"""SDF design of every part of the boxing teddy bear (assembled coordinates)."""
import math
import numpy as np
from .sdfkit import Program, frame, rot_axes, UNION, SUB, INTER

Z_WAIST = 80.0          # waistband top == torso cut plane
Z_HEM = 45.0            # shorts bottom == leg top plane
LEG_X = 22.0
FOOT_TOE_DEG = 12.0
HEAD_C = np.array([0.0, -4.0, 165.5])
S2 = (1, -1)
UPPER_ARM_LEN = 22.0
FOREARM_LEN = 27.0


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------- geometry
def arm_points(s):
    """shoulder, elbow, wrist, forearm axis.  Relaxed boxer stance: upper arm hangs
    down/outward (~17 deg), elbow bent ~40 deg so the forearm points forward-down."""
    S = np.array([s * 46.5, -2.0, 123.0])
    u = unit((s * 0.26, -0.30, -0.92))
    E = S + UPPER_ARM_LEN * u
    g = unit((s * 0.10, -0.80, -0.59))
    W = E + FOREARM_LEN * g
    return S, E, W, g


def glove_frame(s):
    """rows: x = up-ish, y = towards the body centre line (palm side), z = cuff axis."""
    S, E, W, g = arm_points(s)
    inward = np.array([-s, 0.0, 0.0])
    ey = unit(inward - g * (inward @ g))
    ex = unit(np.array([0, 0, 1.0]) - g * g[2])
    ez = g
    M = np.stack([ex, ey, ez])
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
    p.ell(HEAD_C, (41.0, 35.0, 30.5), k=3)
    for s in S2:
        p.ell((s * 28.0, -12.0, 154.0), (16.5, 20.0, 15.0), k=10)   # plush cheeks, low and wide
    p.ell((0, -12.0, 148.0), (26.0, 22.0, 11.5), k=10)               # round jaw (no flat underside)
    p.ell((0, 6.0, 161.0), (36.0, 27.0, 27.0), k=8)                 # back of head
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
def _perp(v, ref):
    v = unit(v)
    r = np.asarray(ref, float)
    return unit(r - v * (r @ v))


def body_program(face, arms=True):
    """Head + upper torso + arms down to the wrist cut planes (all rounded masses)."""
    p = Program()
    add_head(p)
    # ------- torso masses (big blends)
    p.ell((0, 1, 140.0), (32.0, 21.0, 13.0), k=10)                  # trapezius mass, carries the head
    p.ell((0, -12.0, 141.0), (29.0, 16.0, 10.5), k=9)               # fills under the chin: no ledge
    p.ell((0, 0, 114.0), (41.0, 27.5, 28.0), k=8)                   # ribcage / chest
    p.ell((0, -1, 91.0), (27.5, 21.0, 16.5), k=8)                   # abdomen (narrow waist -> V taper)
    for s in S2:
        p.ell((s * 30.0, 10.0, 108.0), (12.5, 13.5, 23.0), M=rot_axes(0, 0, s * 14), k=6)    # flared lats
        p.ell((s * 25.5, -4.0, 90.5), (8.5, 14.0, 13.0), M=rot_axes(0, 0, -s * 8), k=4)      # obliques
        p.ell((s * 18.5, -18.5, 118.0), (20.5, 11.5, 14.0), M=rot_axes(0, -s * 8, 0), k=3.5)  # pecs
        p.cap((s * 8, 3, 145), (s * 40, 1, 130), 11.0, 9.5, k=9)    # trapezius slope head -> shoulder
        p.ell((s * 17, 19.5, 120.0), (12.5, 7.0, 11.0), M=rot_axes(0, s * 12, 0), k=4)       # shoulder blades
        p.cap((s * 6.0, 19.0, 83), (s * 7.0, 21.0, 108), 5.0, 5.5, k=4)                       # erectors
        for z in (100.5, 91.8, 83.2):                                                         # abs pillows
            p.ell((s * 9.2, -19.6, z), (8.4, 5.6, 4.8), k=2.4)
    # ------- arms
    if arms:
        for s in S2:
            S, E, W, g = arm_points(s)
            u = unit(E - S)
            Mu = frame(u, hint=(1, 0, 0))
            front = _perp(u, (0, -1, 0))
            back = -front
            p.ell(S + np.array([0.0, 0.0, 2.0]), (18.5, 17.8, 17.8), k=6)             # cannonball deltoid
            p.cap(S, E, 14.2, 12.8, k=4)                                               # upper arm
            p.ell((S + E) / 2 + front * 4.5, (10.8, 10.8, 13.0), M=Mu, k=3.5)             # biceps
            p.ell((S + E) / 2 + back * 5.5 + np.array([s * 1.5, 0, 0]), (10.2, 10.2, 14.0), M=Mu, k=3.5)  # triceps
            p.ell(E, (13.4, 13.4, 12.8), k=3.5)                                         # elbow
            p.cap(E, W, 13.4, 11.2, k=3)                                                # forearm taper
            Mg = frame(g, hint=(1, 0, 0))
            p.ell(E + g * 7.5, (13.0, 12.6, 13.0), M=Mg, k=3.5)                         # forearm belly
    # ------- cut planes
    p.plane((0, 0, -1), -Z_WAIST, op=INTER, k=0)       # keep z >= waist
    if arms:
        for s in S2:
            S, E, W, g = arm_points(s)
            p.plane(g, float(g @ W), op=INTER, k=0)    # keep points before the wrist plane
    ref = Program()
    ref.rows = list(p.rows)
    front = (0, 1, 0)
    back = (0, -1, 0)
    # linea alba
    pts = surf_pts(ref, [((0, -120, z), front) for z in np.arange(80.5, 106.5, 3.0)])
    carve_tapered(p, ref, pts, np.full(len(pts), 1.1), np.r_[0.4, np.full(len(pts) - 2, 1.0), 0.3], k=1.2)
    # soft ab-row grooves
    for zg in (96.2, 87.4):
        xs = np.linspace(-16, 16, 9)
        pts = surf_pts(ref, [((x, -120, zg + 0.012 * x * x - 1.4), front) for x in xs])
        carve_tapered(p, ref, pts, np.full(len(pts), 1.3), np.r_[0.3, np.full(len(pts) - 2, 1.0), 0.3], k=1.4)
    # pec centre + lower pec line
    pts = surf_pts(ref, [((0, -120, z), front) for z in np.arange(106.0, 128.0, 3.0)])
    carve_tapered(p, ref, pts, np.full(len(pts), 1.1), np.r_[0.3, np.full(len(pts) - 2, 1.3), 0.3], k=1.2)
    for s in S2:
        xs = np.linspace(2.5, 33, 10)
        pts = surf_pts(ref, [((s * x, -120, 105.2 + 0.0055 * x * x), front) for x in xs])
        carve_tapered(p, ref, pts, np.full(len(pts), 1.3), np.r_[0.3, np.full(len(pts) - 2, 1.3), 0.3], k=1.6)
        for z0 in (92, 99):
            pts = surf_pts(ref, [((s * (25 + 2.5 * t), -120, z0 + 3.0 * t), front) for t in range(5)])
            carve_tapered(p, ref, pts, [0.7, 1.0, 1.0, 1.0, 0.7], [0.0, 0.6, 0.8, 0.6, 0.0], k=1.2)
    # spine groove + scapula / lat lines
    pts = surf_pts(ref, [((0, 120, z), back) for z in np.arange(82, 140, 3.0)])
    carve_tapered(p, ref, pts, np.full(len(pts), 1.5), np.r_[0.3, np.full(len(pts) - 2, 1.4), 0.3], k=1.2)
    for s in S2:
        xs = np.linspace(6, 27, 7)
        pts = surf_pts(ref, [((s * x, 120, 113.0 + 0.30 * (x - 6)), back) for x in xs])
        carve_tapered(p, ref, pts, [0.7, 1.1, 1.2, 1.2, 1.2, 1.1, 0.7], [0.0, 0.8, 1.0, 1.0, 1.0, 0.8, 0.0], k=1.2)
        zs = np.linspace(106, 84, 7)
        pts = surf_pts(ref, [((s * (38.0 - 0.5 * (106 - z)), 120, z), back) for z in zs])
        carve_tapered(p, ref, pts, [0.7, 1.1, 1.3, 1.3, 1.2, 1.1, 0.7], [0.0, 0.7, 0.9, 0.9, 0.9, 0.6, 0.0], k=1.2)
    # ears: inner bowl
    for s in S2:
        EC, n_e = ear_geom(s)
        Rb = 14.0
        p.ell(EC + n_e * (EAR_R[2] + Rb - 2.7), (Rb, Rb, Rb), op=SUB, k=1.3)
    # brows
    for s in S2:
        pts = face["brows"][s]
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
    R = M.T

    def L(v):
        return pivot + R @ np.asarray(v, float)

    # plush paw: big rounded pad + 4 toe bumps
    p.ell(L((0, -7.0, 7.5)), (16.4, 23.0, 10.5), M=M, k=3)
    p.ell(L((0, 5.0, 13.0)), (14.5, 13.0, 9.5), M=M, k=6)                    # heel / ankle
    toes_x = (-11.4, -3.8, 3.8, 11.4)
    toes_y = (-23.0, -25.8, -25.8, -23.0)
    toes_r = ((4.9, 6.3, 5.8), (5.0, 6.5, 6.2), (5.0, 6.5, 6.2), (4.9, 6.3, 5.8))
    for tx, ty, tr in zip(toes_x, toes_y, toes_r):
        p.ell(L((tx, ty + 1.2, 6.8)), tr, M=M, k=2.0)
    # chunky leg, tapering to the ankle
    p.cap((s * LEG_X, 2.0, 12.0), (s * LEG_X, 0.5, 47.0), 14.8, 19.6, k=7)
    p.ell((s * LEG_X, 6.5, 28.0), (15.5, 12.5, 14.5), k=7)                    # calf
    p.ell((s * LEG_X, -6.5, 38.0), (15.8, 10.5, 10.5), k=7)                   # thigh front
    p.plane((0, 0, -1), 0.0, op=INTER, k=0)                                   # flat sole z>=0
    p.plane((0, 0, 1), Z_HEM, op=INTER, k=0)                                  # leg top
    ref = Program()
    ref.rows = list(p.rows)
    for tx in (-7.6, 0.0, 7.6):                                               # 3 toe grooves
        p.cap(L((tx, -33.0, 11.5)), L((tx, -20.5, 12.5)), 1.2, 1.2, op=SUB, k=1.2)
        p.cap(L((tx, -31.5, 11.0)), L((tx, -29.0, 3.5)), 1.2, 1.2, op=SUB, k=1.2)
    return p


# ------------------------------------------------------------------ shorts
def shorts_program():
    p = Program()
    # hips: wide ellipsoid, legs flare out towards the hems
    p.ell((0, 0, 64.0), (42.0, 29.0, 19.0), k=6)
    for s in S2:
        p.cap((s * LEG_X, 0.5, 70.0), (s * LEG_X, 0.5, 47.0), 17.4, 24.0, k=8)
    # thick waistband with three rounded ribs
    p.ecyl((0, 0, 75.4), 36.6, 27.0, 4.6, 2.0, k=1.5)
    for z in (72.4, 75.4, 78.3):
        p.ering((0, 0, z), 37.0, 27.4, 1.75, k=0.6)
    # inverted-V split between the legs
    p.ell((0, 0, 41.0), (5.8, 40.0, 10.5), op=SUB, k=2.5)
    p.plane((0, 0, 1), Z_WAIST, op=INTER, k=0)
    p.plane((0, 0, -1), -Z_HEM, op=INTER, k=0)
    ref = Program()
    ref.rows = list(p.rows)
    for s in S2:   # rolled hems
        p.ering((s * LEG_X, 0.5, Z_HEM + 1.4), 24.3, 24.3, 1.4, k=1.0)
    # broad, very shallow fabric waves under the waistband (smooth, no sharp dents)
    rng = np.random.default_rng(3)
    angs = np.linspace(0, 2 * math.pi, 9, endpoint=False) + rng.normal(0, 0.1, 9)
    for th in angs:
        d = (-math.cos(th), -math.sin(th), 0)
        zc = 63.5 + rng.uniform(-1.5, 1.5)
        o = np.array([math.cos(th) * 80, math.sin(th) * 60, zc])
        q = ref.raymarch(o, d)
        if q is None:
            continue
        n = ref.grad(q[None])[0]
        n /= np.linalg.norm(n)
        r = 6.0
        p.ell(q + n * (r - 0.35), (r, r, rng.uniform(9.0, 12.0)), op=SUB, k=8.0)
    for s in S2:
        q = ref.raymarch((s * 80, 0.5, 49.0), (-s, 0, 0))     # outer hem notch
        if q is not None:
            p.ell((q[0] - s * 0.5, q[1], Z_HEM + 3.8), (3.2, 0.7, 4.8), op=SUB, k=0.1)
    p.plane((0, 0, 1), Z_WAIST, op=INTER, k=0)
    p.plane((0, 0, -1), -Z_HEM, op=INTER, k=0)
    return p


# ------------------------------------------------------------------ gloves
def glove_program(s):
    """Smooth puffy glove, local frame: x up, y palm side (towards body), z cuff axis."""
    W, M = glove_frame(s)
    R = M.T

    def Lg(v):
        return W + R @ np.asarray(v, float)

    p = Program()
    ex, ey, g = M[0], M[1], M[2]
    p.cap(Lg((0, 0, -3.0)), Lg((0, 0, 13.5)), 13.6, 13.6, k=1.0)                # thick cuff band
    p.ell(Lg((1.0, -0.5, 28.0)), (20.0, 18.5, 16.5), M=M, k=5)                  # main puff
    p.ell(Lg((2.0, -1.5, 32.5)), (19.0, 18.0, 12.5), M=M, k=5)                  # knuckle / fist end
    p.ell(Lg((2.5, -3.5, 22.0)), (15.5, 15.5, 11.0), M=M, k=5)                  # back-of-hand fullness
    ta, tb = Lg((9.5, 14.0, 16.0)), Lg((12.0, 10.0, 35.0))                        # ONE thumb, upper inner side
    p.cap(ta, tb, 7.0, 6.0, k=2.5)
    p.plane(-g, float(-g @ W), op=INTER, k=0)
    ref = Program()
    ref.rows = list(p.rows)
    # cuff seam + stitching
    p.torus(Lg((0, 0, 13.0)), 13.3, 0.6, M=M, op=SUB, k=0.2)
    p.torus(Lg((0, 0, 6.5)), 13.2, 0.55, M=M, op=SUB, k=0.2)
    for i in range(26):
        th = 2 * math.pi * i / 26
        pos = Lg((13.2 * math.cos(th), 13.2 * math.sin(th), 10.0))
        nrm = ex * math.cos(th) + ey * math.sin(th)
        p.ell(pos + nrm * 0.15, (0.55, 0.55, 0.55), op=SUB, k=0.1)
    # thumb seam (front edge) and a panel seam along the back
    pts = []
    for z in np.linspace(14, 38, 9):
        h = ref.raymarch(Lg((0, -40, z)), ey)
        if h is not None:
            pts.append(h)
    if len(pts) > 2:
        carve_tapered(p, ref, np.array(pts), [0.5] * len(pts), [0.0] + [0.45] * (len(pts) - 2) + [0.0], k=0.3)
    pts = []
    for z in np.linspace(14, 36, 9):
        h = ref.raymarch(Lg((40, 0, z)), -ex)
        if h is not None:
            pts.append(h)
    if len(pts) > 2:
        carve_tapered(p, ref, np.array(pts), [0.5] * len(pts), [0.0] + [0.45] * (len(pts) - 2) + [0.0], k=0.3)
    # few soft compression wrinkles near the thumb root and the knuckles (shallow dents)
    for (lx, ly, lz, ra) in ((6.0, 12.5, 14.5, 3.0), (13.5, 3.0, 34.5, 3.2), (13.0, -6.0, 33.0, 3.0), (10.5, 12.0, 30.0, 2.6)):
        c = Lg((lx, ly, lz))
        h = ref.raymarch(c + (ex * lx + ey * ly) * 0.0 + (ex * 8 + ey * ly * 0.5) * 1.0, -(ex * 8 + ey * ly * 0.5))
        if h is not None:
            n = ref.grad(h[None])[0]
            n /= np.linalg.norm(n)
            p.ell(h + n * (ra - 0.4), (ra, ra * 0.55, ra * 1.7), M=frame(unit(np.cross(n, g)), hint=n), op=SUB, k=2.0)
    return p


# -------------------------------------------------------------------- base
def base_program(diam, thick):
    p = Program()
    p.ecyl((0, 0, -thick / 2), diam / 2, diam / 2, thick / 2, 2.4, k=1.0)
    return p
