"""Builds every part (assembled coordinates) and caches the heavy meshes."""
import os, time, pickle, hashlib, json
import numpy as np
import trimesh
import manifold3d as mf
from . import design as D, regions as RG, face as FC
from .config import Params
from .sdfkit import Program
from .meshing import mesh_program, clean
from .fur import make_bumps
from .snap import Joint, pin_solid, socket_void, to_man, to_tri, mat34, cyl, revolve

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "cache")
os.makedirs(CACHE, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cache_key(P: Params, name):
    d = {k: v for k, v in P.__dict__.items() if k not in ("CLEARANCE",)}
    src = ""
    for fn in ("design.py", "regions.py", "fur.py", "sdfkit.py", "meshing.py", "build.py"):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fn)) as fh:
            src += fh.read()
    s = json.dumps(d, sort_keys=True, default=str) + name + hashlib.md5(src.encode()).hexdigest()
    return hashlib.md5(s.encode()).hexdigest()[:12]


def cached(P, name, fn):
    f = os.path.join(CACHE, f"{name}_{cache_key(P, name)}.pkl")
    if os.path.exists(f):
        with open(f, "rb") as fh:
            return pickle.load(fh)
    r = fn()
    with open(f, "wb") as fh:
        pickle.dump(r, fh, protocol=4)
    return r


def snap_planes(m, planes, tol=0.06):
    """Project vertices lying within tol of a design cut plane exactly onto it
    (keeps mating faces flat after simplification)."""
    v = m.vertices.copy()
    for n, off in planes:
        n = np.asarray(n, float) / np.linalg.norm(n)
        d = v @ n - off
        sel = (np.abs(d) < tol) | ((d > 0) & (d < 1.5))      # on / marginally beyond the cut plane
        v[sel] -= np.outer(d[sel], n)
    return trimesh.Trimesh(v, m.faces, process=False)


def simplify(m, tol, planes=()):
    M = to_man(m)
    r = to_tri(M.simplify(tol))
    if planes:
        r = to_tri(to_man(snap_planes(r, planes)))
    return r


def vox(P, v):
    return v * (2.2 if P.preview else 1.0)


# ------------------------------------------------------------------ joints
def make_joints(P: Params):
    J = {}
    J["torso"] = Joint("torso", np.array([0, -0.5, D.Z_WAIST]), np.array([0, 0, -1.0]), np.array([0, 1.0, 0]),
                       *P.TORSO_PIN, fillet=P.PIN_FILLET)
    for s, nm in ((1, "L"), (-1, "R")):
        J["leg" + nm] = Joint("leg" + nm, np.array([s * D.LEG_X, 0, D.Z_HEM]), np.array([0, 0, 1.0]),
                              np.array([0, 1.0, 0]), *P.LEG_PIN, tongue_r=11.0, tongue_L=P.TONGUE_LEN,
                              fillet=P.PIN_FILLET)
        S, E, W, g = D.arm_points(s)
        J["glove" + nm] = Joint("glove" + nm, W, g, np.array([0, 1.0, 0]), *P.GLOVE_PIN, tongue_r=8.2,
                                tongue_L=P.TONGUE_LEN, fillet=P.PIN_FILLET)
        J["foot" + nm] = Joint("foot" + nm, np.array([s * D.LEG_X, 0, 0.0]), np.array([0, 0, 1.0]),
                               np.array([0, 1.0, 0]), *P.BASE_PIN, fillet=P.BASE_PIN_FILLET)
    return J


# -------------------------------------------------------------------- body
def build_body(P: Params, face, head_prog, info):
    def run():
        t = time.time()
        body = D.body_program(face)
        lo, hi = (-74, -44, 78), (74, 38, 202)
        muz_prog = FC.muzzle_sdf(face["muz"], -50.0)
        muz_ell = Program()
        muz_ell.rows = muz_prog.rows[:1]
        bumps = make_bumps(body, lo, hi, None, RG.body_region(face, muz_ell, P.JOINT_SMOOTH_MM), RG.body_flow,
                           seed=P.seed, strength=P.FUR_STRENGTH, diam=P.TUFT_DIAM, height=P.BUMP_HEIGHT,
                           elong=P.FUR_ELONGATION, tuft=P.TUFT)
        log("  body bumps", None if bumps is None else len(bumps["BC"]), f"{time.time()-t:.0f}s")
        m = mesh_program(body, lo, hi, vox(P, P.VOXEL_FUR), bumps=bumps,
                         noise_amp=P.FUR_NOISE_AMP * min(P.FUR_STRENGTH, 1.5) if bumps is not None else 0.0)
        log("  body raw", len(m.faces), m.is_watertight, f"{time.time()-t:.0f}s")
        pl = [((0, 0, -1), -D.Z_WAIST)] + [(D.arm_points(s)[3], float(D.arm_points(s)[3] @ D.arm_points(s)[2])) for s in D.S2]
        m = simplify(m, 0.035, pl)
        log("  body simplified", len(m.faces), m.is_watertight, f"{time.time()-t:.0f}s")
        return m
    return cached(P, "body_base", run)


def finish_body(P, base, face, head_prog, info, J):
    M = to_man(base)
    for c in FC.head_cutters(face, head_prog, P, info):
        M = M - c
    for pg in FC.head_pegs(info, P):
        M = M + pg
    M = M + pin_solid(J["torso"], P).transform(J["torso"].T())
    for nm in ("gloveL", "gloveR"):
        M = M + pin_solid(J[nm], P).transform(J[nm].T())
    return to_tri(M)


# -------------------------------------------------------------------- legs
def build_leg(P, s):
    def run():
        t = time.time()
        prog = D.leg_program(s)
        lo = (0, -34, -1) if s > 0 else (-48, -34, -1)
        hi = (48, 21, 47) if s > 0 else (0, 21, 47)
        bumps = make_bumps(prog, lo, hi, None, RG.leg_region(P.JOINT_SMOOTH_MM), RG.leg_flow, seed=P.seed + s,
                           strength=P.FUR_STRENGTH, diam=P.TUFT_DIAM, height=P.BUMP_HEIGHT, elong=P.FUR_ELONGATION,
                           tuft=P.TUFT)
        m = mesh_program(prog, lo, hi, vox(P, P.VOXEL_FUR), bumps=bumps,
                         noise_amp=P.FUR_NOISE_AMP * min(P.FUR_STRENGTH, 1.5) if bumps is not None else 0.0)
        m = simplify(m, 0.03, [((0, 0, -1), 0.0), ((0, 0, 1), D.Z_HEM)])
        log(f"  leg{s} {len(m.faces)} faces wt={m.is_watertight} {time.time()-t:.0f}s")
        return m
    return cached(P, f"leg{s}", run)


def finish_leg(P, base, J, nm, clr):
    M = to_man(base)
    M = M + pin_solid(J["leg" + nm], P).transform(J["leg" + nm].T())
    j = J["foot" + nm]
    M = M - socket_void(j, P, clr).transform(j.T())
    return to_tri(M)


# ------------------------------------------------------------------ shorts
def build_shorts(P):
    def run():
        prog = D.shorts_program()
        m = mesh_program(prog, (-48, -32, 43), (48, 32, 82), vox(P, 0.22))
        m = simplify(m, 0.01, [((0, 0, 1), D.Z_WAIST), ((0, 0, -1), -D.Z_HEM)])
        log("  shorts", len(m.faces), m.is_watertight)
        return m
    return cached(P, "shorts", run)


def finish_shorts(P, base, J, clr):
    M = to_man(base)
    for nm in ("torso", "legL", "legR"):
        M = M - socket_void(J[nm], P, clr).transform(J[nm].T())
    return to_tri(M)


# ------------------------------------------------------------------ gloves
def build_glove(P, s):
    def run():
        prog = D.glove_program(s)
        W_, M_ = D.glove_frame(s)
        c_ = W_ + M_[2] * 22.0
        lo, hi = c_ - 34.0, c_ + 34.0
        m = mesh_program(prog, lo, hi, vox(P, 0.2))
        S_, E_, W_, g_ = D.arm_points(s)
        m = simplify(m, 0.01, [(-g_, float(-g_ @ W_))])
        log(f"  glove{s}", len(m.faces), m.is_watertight)
        return m
    return cached(P, f"glove{s}", run)


def finish_glove(P, base, J, nm, clr):
    M = to_man(base)
    j = J["glove" + nm]
    M = M - socket_void(j, P, clr).transform(j.T())
    return to_tri(M)


# -------------------------------------------------------------------- base
def base_diameter(P, legs):
    """smallest circle (centred at the feet's centroid) containing both soles + margin."""
    pts = []
    for m in legs:
        v = m.vertices[m.vertices[:, 2] < 0.2][:, :2]
        pts.append(v)
    pts = np.concatenate(pts)
    c = pts.mean(0)
    # refine centre by minimising max radius (simple iterative)
    for _ in range(200):
        r = np.linalg.norm(pts - c, axis=1)
        far = pts[np.argmax(r)]
        c = c + (far - c) * 0.01
    r = np.linalg.norm(pts - c, axis=1).max()
    return max(P.BASE_DIAMETER, 2 * (r + 4.0)), c


def build_base(P, J, diam, centre):
    R = diam / 2
    t = P.BASE_THICK
    f = 2.5
    prof = [(0, -t), (R - f, -t), ]
    for th in np.linspace(-90, 0, 9)[1:]:
        a = np.radians(th)
        prof.append((R - f + f * np.cos(a), -f + f * np.sin(a)))
    prof += [(R, 0 - 0.0), (0, 0)]
    # simple: chamfered/rounded top edge, flat bottom
    prof = [(0, -t), (R - 0.4, -t), (R, -t + 0.4), (R, -f)]
    for th in np.linspace(0, 90, 10)[1:]:
        a = np.radians(th)
        prof.append((R - f + f * np.cos(a), -f + f * np.sin(a)))
    prof += [(0, 0)]
    base = revolve(prof, 128).translate((float(centre[0]), float(centre[1]), 0))
    for nm in ("footL", "footR"):
        base = base + pin_solid(J[nm], P).transform(J[nm].T())
    return to_tri(base)


# -------------------------------------------------------------- face parts
def T4(M34):
    T = np.eye(4)
    T[:3, :] = M34
    return T


def compose(A34, B34):
    return (T4(A34) @ T4(B34))[:3, :]


def build_face_parts(P, face, head_prog, info):
    """returns dict name -> dict(mesh=local trimesh, world=4x4 local->assembled)"""
    out = {}
    eye_m, einfo = FC.eye_part(P)
    out["eye"] = dict(mesh=to_tri(eye_m))
    out["highlight"] = dict(mesh=to_tri(FC.highlight_pin(einfo)))
    for s, nm in ((1, "L"), (-1, "R")):
        e = face["eyes"][s]
        Rm = FC.local_frame_eye(e)
        out["eye" + nm] = dict(world=T4(mat34(Rm, e["S"] - e["a"] * FC.DISH_DEPTH)))
        u, v = einfo["hole_uv"]
        Tl = mat34(np.eye(3), (u, v, einfo["hole_z"]))
        out["hl" + nm] = dict(world=T4(compose(mat34(Rm, e["S"] - e["a"] * FC.DISH_DEPTH), Tl)))
    # muzzle (velvet from a fine SDF)
    muz = face["muz"]
    mprog = FC.muzzle_sdf(muz, info["floor"])
    Rm, right, upv, a = FC.muz_matrix(muz)
    S = muz["S"]

    def region(Pp, N):
        d = head_prog.eval(Pp)
        w = np.clip((d - 0.15) / 0.6, 0, 1)
        return w, np.ones(len(Pp))
    lo, hi = S - 17, S + 17
    bumps = make_bumps(mprog, lo, hi, None, region, lambda Pp: np.tile(np.array([0, 0, -1.0]), (len(Pp), 1)),
                       seed=P.seed + 5, strength=0.25 * P.FUR_STRENGTH, diam=(0.6, 0.8),
                       height=(0.35, 0.6), elong=1.2, density=1.0, verbose=False)
    raw = mesh_program(mprog, lo, hi, vox(P, P.VOXEL_FACE) if not P.preview else 0.2, bumps=bumps)
    mz, fl = FC.muzzle_final(raw, info, P)
    out["muzzle"] = dict(mesh=to_tri(mz), world=T4(mat34(Rm, S + a * fl)))
    nose = to_tri(FC.nose_part(info, P))
    nf = info["nose"]
    Tn = mat34(nf["T"][:, :3], nf["T"][:, 3] + nf["T"][:, :3] @ np.array([0, 0, nf["floor"]]))
    out["nose"] = dict(mesh=nose, world=T4(compose(info["T"], Tn)))
    return out
