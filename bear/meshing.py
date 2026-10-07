import numpy as np, trimesh
from skimage import measure
from .sdfkit import sample_field


def clean(m, min_frac=1e-3):
    """Drop floaters, merge vertices, ensure single watertight shell if possible."""
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        parts = sorted(parts, key=lambda p: -len(p.faces))
        keep = [p for p in parts if len(p.faces) >= min_frac * len(parts[0].faces)]
        m = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
    return m


def field_to_mesh(field, lo, h, level=0.0):
    # nudge exact-zero samples so marching cubes never meets a degenerate vertex
    field = np.where(field == 0.0, np.float32(1e-5), field).astype(np.float32)
    v, f, n, _ = measure.marching_cubes(field, level, spacing=(h, h, h), allow_degenerate=False)
    v = v + np.asarray(lo)
    m = trimesh.Trimesh(v, f, process=False)
    return clean(m)


def mesh_program(prog, lo, hi, h, bumps=None, noise_amp=0.0, pad=1.0):
    lo = np.asarray(lo, float) - pad
    hi = np.asarray(hi, float) + pad
    fld = sample_field(prog, lo, hi, h, bumps=bumps, noise_amp=noise_amp)
    return field_to_mesh(fld, lo, h)
