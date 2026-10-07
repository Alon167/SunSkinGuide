"""Plush fur: Poisson-disk bump seeds on the base surface -> sunk-sphere bumps
that are union-ed (smoothly) into the SDF before marching cubes."""
import numpy as np
import trimesh
from .sdfkit import poisson_select
from .meshing import mesh_program


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def tangent_flow(N, direction):
    """Project `direction` (n,3 or 3,) into the tangent plane of N."""
    d = np.broadcast_to(np.asarray(direction, float), N.shape).copy()
    d = d - (d * N).sum(1, keepdims=True) * N
    ln = np.linalg.norm(d, axis=1, keepdims=True)
    bad = ln[:, 0] < 1e-3
    if bad.any():
        alt = np.cross(N[bad], np.array([1.0, 0.3, 0.2]))
        alt /= np.linalg.norm(alt, axis=1, keepdims=True)
        d[bad] = alt
        ln[bad] = 1.0
    return d / np.maximum(ln, 1e-9)


def make_bumps(prog, lo, hi, P, region_fn, flow_fn, seed=7, strength=1.0,
               diam=(0.9, 1.4), height=(0.35, 0.60), elong=1.3, density=1.0, verbose=True):
    rng = np.random.default_rng(seed)
    if strength <= 0:
        return None
    base = mesh_program(prog, lo, hi, 0.4)
    area = base.area
    ncand = int(area * 14 * density)
    pts, fidx = trimesh.sample.sample_surface(base, ncand, seed=seed)
    pts = prog.project(pts, iters=3)
    g = prog.grad(pts)
    N = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    d, dn = prog.eval(pts, both=True)
    ok = np.abs(d) < 0.05
    pts, N, dn = pts[ok], N[ok], dn[ok]
    w, sz = region_fn(pts, N)
    gd = np.clip(-dn, 0.0, 1.0)
    w = w * (1.0 - 0.5 * smoothstep(0.05, 0.9, gd))
    keep = w > 0.04
    pts, N, w, sz = pts[keep], N[keep], w[keep], sz[keep]
    R = 0.5 * rng.uniform(diam[0], diam[1], len(pts)) * sz
    acc = poisson_select(pts, R, rng, fac=0.70)
    pts, N, w, R = pts[acc], N[acc], w[acc], R[acc]
    Hh = (height[0] + (height[1] - height[0]) * rng.uniform(0, 1, len(pts))) * strength * w
    ok = Hh > 0.06
    pts, N, w, R, Hh = pts[ok], N[ok], w[ok], R[ok], Hh[ok]
    Rs = (R * R + Hh * Hh) / (2 * Hh)
    Rs = np.minimum(Rs, 3.0)
    BC = pts - N * (Rs - Hh)[:, None]
    F = tangent_flow(N, flow_fn(pts))
    BE = 1.0 + (elong - 1.0) * rng.uniform(0.5, 1.0, len(pts))
    if verbose:
        print(f"   fur: area {area:.0f} mm2, candidates {ncand}, bumps {len(pts)} "
              f"({len(pts)/area:.2f}/mm2)")
    return dict(BC=BC, BR=Rs, BF=F, BE=BE, BW=w)
