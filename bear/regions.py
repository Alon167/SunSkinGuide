"""Per-area fur strength / flow-direction functions."""
import numpy as np
from . import design as D
from .fur import smoothstep


def disc_dist(P, centre, axis, radius):
    """Distance from points to a flat disc (centre, axis, radius)."""
    rel = P - centre
    h = rel @ axis
    lat = np.linalg.norm(rel - np.outer(h, axis), axis=1)
    return np.sqrt(np.maximum(lat - radius, 0) ** 2 + h ** 2)


def muzzle_sdf_points(P, muz_ellipsoid):
    return muz_ellipsoid.eval(P)


def body_region(face, muz_prog, smooth_mm=2.0):
    def fn(P, N):
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        w = np.ones(len(P))
        sz = np.ones(len(P))
        # chest / abs 70 %
        wf = smoothstep(0.1, 0.5, -N[:, 1]) * (1 - smoothstep(128, 136, z)) * (1 - smoothstep(32, 40, np.abs(x)))
        wf *= (z < 140)
        w *= 1.0 - 0.3 * wf
        # inner ears 50 %, finer bumps
        for s in D.S2:
            EC, n_e = D.ear_geom(s)
            Rb = 14.0
            cb = EC + n_e * (D.EAR_R[2] + Rb - 2.7)
            r = np.linalg.norm(P - cb, axis=1) - Rb
            m = 1 - smoothstep(0.3, 1.2, np.abs(r))
            m = m * (np.linalg.norm(P - EC, axis=1) < 16)
            w *= 1 - 0.5 * m
            sz *= 1 - 0.3 * m
        # eye dishes
        for s in D.S2:
            e = face["eyes"][s]
            rel = P - e["S"]
            h = rel @ e["a"]
            rho = np.linalg.norm(rel - np.outer(h, e["a"]), axis=1)
            m = smoothstep(8.2, 10.4, rho) + (h < -5)
            w *= np.clip(m, 0, 1)
        # muzzle / nose recess area
        dm = muz_prog.eval(P)
        w *= smoothstep(smooth_mm, smooth_mm + 1.4, dm)
        # torso cut plane (z = 80)
        w *= smoothstep(smooth_mm, smooth_mm + 1.4, z - D.Z_WAIST)
        # wrist cut planes
        for s in D.S2:
            S, E, W, g = D.arm_points(s)
            dd = disc_dist(P, W, g, 10.5)
            w *= smoothstep(smooth_mm, smooth_mm + 1.4, dd)
        return w, sz
    return fn


def body_flow(P):
    d = np.tile(np.array([0.0, 0.0, -1.0]), (len(P), 1))
    face_c = np.array([0.0, -8.0, 165.0])
    rad = P - face_c
    head = (P[:, 2] > 140.0)
    d[head] = rad[head]
    return d


def leg_region(smooth_mm=2.0):
    def fn(P, N):
        z = P[:, 2]
        w = np.ones(len(P))
        w *= smoothstep(1.2, 3.0, z)                              # smooth soles
        w *= smoothstep(smooth_mm, smooth_mm + 1.4, D.Z_HEM - z)   # smooth top joint
        return w, np.ones(len(P))
    return fn


def leg_flow(P):
    return np.tile(np.array([0.0, 0.0, -1.0]), (len(P), 1))
