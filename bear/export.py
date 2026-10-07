"""Print orientation, plate layout, STL/3MF writers, brims, test_snap."""
import io, math, os, zipfile
import numpy as np
import trimesh
import manifold3d as mf
from . import design as D
from .config import COLORS
from .snap import Joint, pin_solid, socket_void, to_man, to_tri, box, cyl, mat34


def rot_x(deg):
    return trimesh.transformations.rotation_matrix(math.radians(deg), [1, 0, 0])


def overhang_area(m, T, limit_deg=50.0):
    """area (mm^2) of faces steeper than limit_deg overhang that are not on the bed."""
    mm = m.copy()
    mm.apply_transform(T)
    n = mm.face_normals
    zmin = mm.vertices[:, 2].min()
    cen_z = mm.triangles_center[:, 2]
    bad = (-n[:, 2] > math.sin(math.radians(limit_deg))) & (cen_z > zmin + 0.35)
    return float(mm.area_faces[bad].sum()), float(mm.area_faces[(-n[:, 2] > 0.999) & (cen_z <= zmin + 0.35)].sum())


def ground(m):
    """translate so min z = 0 and centre bbox xy at origin"""
    b = m.bounds
    T = np.eye(4)
    T[:3, 3] = [-(b[0, 0] + b[1, 0]) / 2, -(b[0, 1] + b[1, 1]) / 2, -b[0, 2]]
    return T


def pose_for(kind, m, J=None):
    """4x4 print pose for an assembled-coordinates part."""
    if kind == "body":
        best = None
        for d in range(-10, 11, 2):
            T = rot_x(-90 + d)
            a, _ = overhang_area(m, T)
            if best is None or a < best[0] - 1e-6:
                best = (a, d, T)
        T = best[2]
        mm = m.copy()
        mm.apply_transform(T)
        return ground(mm) @ T, dict(tilt=best[1], overhang=best[0])
    if kind.startswith("glove"):
        g = J[kind].a
        T = trimesh.geometry.align_vectors(g, [0, 0, 1])
        mm = m.copy()
        mm.apply_transform(T)
        return ground(mm) @ T, {}
    mm = m
    return ground(mm), {}


def brim(m, width=1.6, thick=0.24):
    """Union a thin brim (one footprint outline grown by `width`) under a tiny part."""
    M = to_man(m)
    sil = M.project()
    ring = sil.offset(width, mf.JoinType.Round, 2.0, 24)
    br = mf.Manifold.extrude(ring, thick)
    return to_tri(M + br)


# ----------------------------------------------------------------- layout
def pack(items, plate=(256.0, 256.0), gap=4.0, margin=5.0, step=2.0):
    """items: list of (name, w, d). First-fit on a grid; returns dict name -> (x0, y0) lower-left."""
    placed = []
    pos = {}
    order = sorted(items, key=lambda t: -t[1] * t[2])
    for name, w, d in order:
        found = None
        y = margin
        while y + d <= plate[1] - margin + 1e-6 and found is None:
            x = margin
            while x + w <= plate[0] - margin + 1e-6:
                ok = True
                for (px, py, pw, pd) in placed:
                    if x < px + pw + gap and px < x + w + gap and y < py + pd + gap and py < y + d + gap:
                        ok = False
                        break
                if ok:
                    found = (x, y)
                    break
                x += step
            y += step
        if found is None:
            raise RuntimeError(f"plate overflow placing {name}")
        placed.append((found[0], found[1], w, d))
        pos[name] = found
    return pos


# ------------------------------------------------------------------ writers
def write_stl(m, path):
    m.export(path)


def _fmt_mesh(m):
    V = np.asarray(m.vertices)
    F = np.asarray(m.faces)
    vs = "\n".join('<vertex x="%.4f" y="%.4f" z="%.4f"/>' % tuple(r) for r in V.tolist())
    ts = "\n".join('<triangle v1="%d" v2="%d" v3="%d"/>' % tuple(r) for r in F.tolist())
    return f"<mesh><vertices>\n{vs}\n</vertices><triangles>\n{ts}\n</triangles></mesh>"


def write_3mf(path, objects, title="Boxing teddy bear"):
    """objects: list of dict(name, mesh (plate coords), color (name))"""
    cols = []
    for o in objects:
        if o["color"] not in cols:
            cols.append(o["color"])
    bm = "\n".join('<base name="%s" displaycolor="#%02X%02X%02XFF"/>' % ((c,) + tuple(COLORS[c])) for c in cols)
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
             f'<metadata name="Title">{title}</metadata>',
             '<metadata name="Application">bear generator</metadata>',
             f'<resources><basematerials id="1">\n{bm}\n</basematerials>']
    build = []
    for i, o in enumerate(objects):
        oid = i + 2
        parts.append(f'<object id="{oid}" name="{o["name"]}" type="model" pid="1" pindex="{cols.index(o["color"])}">')
        parts.append(_fmt_mesh(o["mesh"]))
        parts.append("</object>")
        build.append(f'<item objectid="{oid}"/>')
    parts.append("</resources><build>" + "".join(build) + "</build></model>")
    xml = "\n".join(parts)
    ct = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", xml)


# ------------------------------------------------------------- test_snap
def _text_man(txt, size, height):
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    tp = TextPath((0, 0), txt, size=size, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    polys = [p for p in tp.to_polygons(closed_only=True) if len(p) > 2]
    cs = mf.CrossSection([[tuple(map(float, q)) for q in p] for p in polys], mf.FillRule.EvenOdd)
    return mf.Manifold.extrude(cs, height)


def build_test_snap(P, clearances=(0.10, 0.15, 0.20, 0.25)):
    pitch = 17.0
    n = len(clearances)
    W = pitch * n + 4
    block = box(0, W, 0, 26, 0, 9)
    H = P.GLOVE_PIN[0], P.GLOVE_PIN[1]
    for i, c in enumerate(clearances):
        x = 2 + pitch * (i + 0.5)
        j = Joint("t", np.array([x, 17.0, 9.0]), np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), *H, fillet=P.PIN_FILLET)
        block = block - socket_void(j, P, c).transform(j.T())
        txt = _text_man(f"{c:.2f}", 3.6, 0.6)
        x0, y0, z0, x1, y1, z1 = txt.bounding_box()
        block = block + txt.translate((x - (x0 + x1) / 2, 4.0, 9.0 - 0.01))
    # pin plate (vertical pins)
    plate = box(0, W, 34, 50, 0, 2.5)
    for i, c in enumerate(clearances):
        x = 2 + pitch * (i + 0.5)
        j = Joint("t", np.array([x, 42.0, 2.5]), np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), *H, fillet=P.PIN_FILLET)
        plate = plate + pin_solid(j, P).transform(j.T())
    lab = _text_man("V", 4.0, 0.6)
    plate = plate + lab.translate((1.5, 35.0, 2.5 - 0.01))
    # horizontal pin (printed lying on its anti-rotation flat)
    zc = H[0] / 2 - P.FLAT_DEPTH
    hb = box(0, 18, 56, 70, 0, 8)
    j = Joint("t", np.array([18.0, 63.0, zc]), np.array([1.0, 0, 0]), np.array([0, 0, -1.0]), *H, fillet=P.PIN_FILLET)
    hb = hb + pin_solid(j, P).transform(j.T())
    lab = _text_man("H", 4.0, 0.6)
    hb = hb + lab.translate((2.0, 58.0, 8 - 0.01))
    allm = block + plate + hb
    return to_tri(allm)
