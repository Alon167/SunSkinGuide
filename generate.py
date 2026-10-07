#!/usr/bin/env python3
"""Boxing teddy bear - parametric generator.

    python generate.py                       # full quality, all outputs
    python generate.py --fur 0               # smooth figure
    python generate.py --preview             # fast coarse build (for experiments)
    python generate.py --height 180 --clearances 0.12,0.18,0.24 --no-base

Everything is written to ./output  (see README.md).
"""
import argparse, json, os, shutil, sys, time
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from bear import design as D, build as B, face as FC, export as E, renders as RN
from bear.config import Params, COLORS
from bear.sdfkit import Program
from bear.snap import to_man, to_tri, pin_solid, socket_void, mat34

log = B.log


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--height", type=float, default=200.0, help="total figure height in mm (uniform scale of everything)")
    ap.add_argument("--fur", type=float, default=1.0, help="FUR_STRENGTH (0 = smooth)")
    ap.add_argument("--clearances", default="0.10,0.15,0.20", help="comma list of per-side snap clearances (3 folders)")
    ap.add_argument("--no-base", action="store_true", help="do not build the base")
    ap.add_argument("--preview", action="store_true", help="coarse, fast build")
    ap.add_argument("--no-renders", action="store_true")
    ap.add_argument("--out", default="output")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override any Params field, e.g. --set TORSO_PIN=12,9 --set BARB=0.35")
    return ap.parse_args()


def make_params(a):
    P = Params(HEIGHT=a.height, FUR_STRENGTH=a.fur, WITH_BASE=not a.no_base, preview=a.preview)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cur = getattr(P, k)
        if isinstance(cur, tuple):
            v = tuple(float(x) for x in v.split(","))
        elif isinstance(cur, bool):
            v = v.lower() in ("1", "true", "yes")
        else:
            v = type(cur)(v)
        setattr(P, k, v)
    return P


PARTS = {
    "body": ("01_body_brown", "brown"), "muzzle": ("02_muzzle_cream", "cream"), "nose": ("03_nose_black", "black"),
    "eye": ("04_eye_black", "black"), "highlight": ("05_highlight_white", "white"),
    "gloveL": ("06_glove_left_red", "red"), "gloveR": ("07_glove_right_red", "red"),
    "shorts": ("08_shorts_black", "black"), "legL": ("09_leg_left_brown", "brown"),
    "legR": ("10_leg_right_brown", "brown"), "base": ("11_base_black", "black"),
}


def posed(m, T):
    mm = m.copy()
    mm.apply_transform(T)
    return mm


def small_pose(m):
    """tiny parts: already in a local frame with the flat side at the lowest z."""
    return posed(m, E.ground(m))


def main():
    a = parse_args()
    P = make_params(a)
    t0 = time.time()
    out = os.path.join(ROOT, a.out)
    clrs = [float(x) for x in a.clearances.split(",")]
    k = P.HEIGHT / 200.0
    os.makedirs(out, exist_ok=True)
    report = {"params": {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in P.__dict__.items()}}

    # ---------------------------------------------------------------- design
    log("layout")
    hp = Program()
    D.add_head(hp)
    face = D.face_layout(hp)
    info = FC.build_muzzle_parts(hp, face["muz"], P, P.FACE_CLEARANCE)
    J = B.make_joints(P)

    log("body")
    body_base = B.build_body(P, face, hp, info)
    body = B.finish_body(P, body_base, face, hp, info, J)
    log(f"body final {len(body.faces)} faces, watertight={body.is_watertight}")
    log("face parts")
    fp = B.build_face_parts(P, face, hp, info)
    log("legs / shorts / gloves")
    leg_base = {nm: B.build_leg(P, s) for s, nm in ((1, "L"), (-1, "R"))}
    shorts_base = B.build_shorts(P)
    glove_base = {nm: B.build_glove(P, s) for s, nm in ((1, "L"), (-1, "R"))}
    diam, centre = B.base_diameter(P, list(leg_base.values()))
    report["base_diameter"] = diam
    base_m = B.build_base(P, J, diam, centre) if P.WITH_BASE else None

    # ------------------------------------------------- clearance-dependent
    variants = {}
    for c in clrs:
        log(f"finish parts for clearance {c:.2f}")
        v = {}
        for nm in ("L", "R"):
            v["leg" + nm] = B.finish_leg(P, leg_base[nm], J, nm, c)
            v["glove" + nm] = B.finish_glove(P, glove_base[nm], J, nm, c)
        v["shorts"] = B.finish_shorts(P, shorts_base, J, c)
        variants[c] = v

    # ----------------------------------------------------- print poses
    poses = {}
    notes = {}
    T_body, nb = E.pose_for("body", body)
    poses["body"] = T_body
    notes["body"] = nb
    ref = variants[clrs[min(1, len(clrs) - 1)]]
    for nm in ("gloveL", "gloveR"):
        poses[nm], _ = E.pose_for(nm, ref[nm], J)
    for nm in ("shorts", "legL", "legR"):
        poses[nm], _ = E.pose_for(nm, ref[nm])
    if base_m is not None:
        poses["base"], _ = E.pose_for("base", base_m)
    for nm in PARTS:
        if nm in poses:
            src = {"body": body, "base": base_m}.get(nm, ref.get(nm))
            ov, bed = E.overhang_area(src, poses[nm])
            notes.setdefault(nm, {}).update(overhang_mm2=round(ov, 1), bed_contact_mm2=round(bed, 1))
    report["print_notes"] = notes

    small = {}
    for nm in ("muzzle", "nose", "eye", "highlight"):
        small[nm] = small_pose(fp[nm]["mesh"])
    brim_parts = {nm: E.brim(small[nm]) for nm in ("nose", "eye", "highlight")}
    brim_parts["muzzle"] = E.brim(small["muzzle"], width=1.2)


    def scaled(m):
        if abs(k - 1) < 1e-9:
            return m
        mm = m.copy()
        mm.apply_scale(k)
        return mm

    # ---------------------------------------------------------- exports
    common = os.path.join(out, "common")
    os.makedirs(os.path.join(common, "with_brim"), exist_ok=True)
    for nm in ("muzzle", "nose", "eye", "highlight"):
        scaled(small[nm]).export(os.path.join(common, PARTS[nm][0] + ".stl"))
        scaled(brim_parts[nm]).export(os.path.join(common, "with_brim", PARTS[nm][0] + "_brim.stl"))

    qty = {"eye": 4, "highlight": 6}
    for c in clrs:
        folder = os.path.join(out, f"clearance_{c:.2f}")
        os.makedirs(folder, exist_ok=True)
        v = variants[c]
        files = {"body": posed(body, poses["body"])}
        for nm in ("gloveL", "gloveR", "shorts", "legL", "legR"):
            files[nm] = posed(v[nm], poses[nm])
        if base_m is not None:
            files["base"] = posed(base_m, poses["base"])
        for nm, m in files.items():
            scaled(m).export(os.path.join(folder, PARTS[nm][0] + ".stl"))
        # plate
        objs = []
        for nm, m in files.items():
            if nm == "base" and not P.WITH_BASE:
                continue
            objs.append((PARTS[nm][0], nm, m, PARTS[nm][1]))
        for nm in ("muzzle", "nose"):
            objs.append((PARTS[nm][0], nm, brim_parts[nm], PARTS[nm][1]))
        for i in range(qty["eye"]):
            objs.append((f"04_eye_black_{i+1}{'_spare' if i >= 2 else ''}", "eye", brim_parts["eye"], "black"))
        for i in range(qty["highlight"]):
            objs.append((f"05_highlight_white_{i+1}{'_spare' if i >= 2 else ''}", "highlight", brim_parts["highlight"], "white"))
        sizes = [(n, scaled(m).extents[0], scaled(m).extents[1]) for n, _, m, _ in objs]
        pos = E.pack(sizes)
        plate_objs = []
        for n, nm, m, col in objs:
            mm = scaled(m)
            b = mm.bounds
            T = np.eye(4)
            T[:3, 3] = [pos[n][0] - b[0, 0], pos[n][1] - b[0, 1], -b[0, 2]]
            plate_objs.append(dict(name=n, mesh=posed(mm, T), color=col))
        E.write_3mf(os.path.join(folder, f"plate_clearance_{c:.2f}.3mf"), plate_objs)
        log(f"wrote {folder}")

    # ------------------------------------------------------ assembled
    c_ref = clrs[min(1, len(clrs) - 1)]
    v = variants[c_ref]
    asm = []
    asm.append(dict(name="body", mesh=body, color="brown"))
    for nm in ("L", "R"):
        asm.append(dict(name="leg" + nm, mesh=v["leg" + nm], color="brown"))
        asm.append(dict(name="glove" + nm, mesh=v["glove" + nm], color="red"))
    asm.append(dict(name="shorts", mesh=v["shorts"], color="black"))
    if base_m is not None:
        asm.append(dict(name="base", mesh=base_m, color="black", gloss=0.5))
    for s, nm in ((1, "L"), (-1, "R")):
        asm.append(dict(name="eye" + nm, mesh=posed(fp["eye"]["mesh"], fp["eye" + nm]["world"]), color="black", gloss=0.95, flat=False))
        asm.append(dict(name="hl" + nm, mesh=posed(fp["highlight"]["mesh"], fp["hl" + nm]["world"]), color="white", gloss=0.6))
    asm.append(dict(name="muzzle", mesh=posed(fp["muzzle"]["mesh"], fp["muzzle"]["world"]), color="cream"))
    asm.append(dict(name="nose", mesh=posed(fp["nose"]["mesh"], fp["nose"]["world"]), color="black", gloss=0.9))
    allm = trimesh.util.concatenate([scaled(p["mesh"]) for p in asm])
    allm.export(os.path.join(out, "assembled_preview.stl"))
    log(f"assembled_preview.stl {len(allm.faces)} faces, height {allm.bounds[1,2]-(-0 if base_m is None else 0):.1f}")
    fig = trimesh.util.concatenate([scaled(p["mesh"]) for p in asm if p["name"] != "base"])
    report["figure_height_mm"] = float(fig.bounds[1, 2] - fig.bounds[0, 2])
    report["figure_bounds"] = fig.bounds.round(2).tolist()

    # -------------------------------------------------------- test_snap
    ts = E.build_test_snap(P)
    scaled(ts).export(os.path.join(out, "test_snap.stl"))

    # -------------------------------------------------------- validation
    val = validate(P, asm, J, v, body, fp, info, face)
    report["validation"] = val

    # ---------------------------------------------------------- renders
    if not a.no_renders:
        log("renders")
        rdir = os.path.join(out, "renders")
        RN.make_renders([dict(p, mesh=scaled(p["mesh"])) for p in asm], rdir)
        ex = explode_vectors(J, face, info)
        for p in asm:
            p["explode"] = ex.get(p["name"], np.zeros(3))
        RN.make_exploded([dict(p, mesh=scaled(p["mesh"]), explode=p["explode"] * k) for p in asm], rdir)

    with open(os.path.join(out, "build_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    log(f"done in {time.time()-t0:.0f}s")


def explode_vectors(J, face, info):
    ex = {}
    up = np.array([0, 0, 30.0])
    ex["body"] = up
    ex["shorts"] = np.zeros(3)
    for s, nm in ((1, "L"), (-1, "R")):
        ex["leg" + nm] = np.array([s * 14.0, 0, -30.0])
        S, E_, W, g = D.arm_points(s)
        ex["glove" + nm] = g * 32 + np.array([s * 6, 0, 0])
        e = face["eyes"][s]
        ex["eye" + nm] = up + e["a"] * 24
        ex["hl" + nm] = up + e["a"] * 24 + np.array([-8.0, -34.0, 14.0]) * 0.0 + e["a"] * 22
    a = face["muz"]["a"]
    ex["muzzle"] = up + a * 30
    ex["nose"] = up + a * 56
    ex["base"] = np.array([0, 0, -58.0])
    return ex


def validate(P, asm, J, v, body, fp, info, face):
    res = {}
    mans = {p["name"]: to_man(p["mesh"]) for p in asm if p["name"] in ("body", "shorts", "legL", "legR", "gloveL", "gloveR", "base")}
    pairs = [("body", "shorts"), ("shorts", "legL"), ("shorts", "legR"), ("body", "gloveL"), ("body", "gloveR"),
             ("gloveL", "shorts"), ("gloveR", "shorts"), ("legL", "legR"), ("gloveL", "legL"), ("gloveR", "legR")]
    if "base" in mans:
        pairs += [("base", "legL"), ("base", "legR")]
    ov = {}
    for a, b in pairs:
        vol = (mans[a] ^ mans[b]).volume()
        ov[f"{a}~{b}"] = round(float(vol), 4)
    res["interference_mm3"] = ov
    # pins inside sockets: pin solid vs the receiving (socketed) part
    pins = {}
    recv = {"torso": "shorts", "legL": "shorts", "legR": "shorts", "gloveL": "gloveL", "gloveR": "gloveR",
            "footL": "legL", "footR": "legR"}
    for jn, rn in recv.items():
        j = J[jn]
        pin = pin_solid(j, P).transform(j.T())
        pins[jn] = round(float((pin ^ mans[rn]).volume()), 4)
    res["pin_vs_socket_interference_mm3"] = pins
    # stability: centre of mass vs support polygon
    parts = [p for p in asm if p["name"] not in ("base",)]
    vol = [(p["mesh"].volume, p["mesh"].center_mass) for p in parts]
    tot = sum(v_ for v_, _ in vol)
    com = sum(v_ * c for v_, c in vol) / tot
    soles = np.concatenate([p["mesh"].vertices[p["mesh"].vertices[:, 2] < 0.05][:, :2] for p in parts if p["name"].startswith("leg")])
    from scipy.spatial import ConvexHull
    hull = ConvexHull(soles)
    pts = soles[hull.vertices]
    # point in polygon
    from matplotlib.path import Path
    inside = Path(pts).contains_point(com[:2])
    # distance to hull edge
    dmin = min(abs(eq[0] * com[0] + eq[1] * com[1] + eq[2]) for eq in hull.equations)
    res["centre_of_mass_xyz"] = [round(float(x), 2) for x in com]
    res["com_inside_support_polygon"] = bool(inside)
    res["com_margin_to_edge_mm"] = round(float(dmin), 2)
    res["watertight"] = {p["name"]: bool(p["mesh"].is_watertight) for p in asm}
    log("validation:", json.dumps({k_: v_ for k_, v_ in res.items() if k_ != "watertight"}))
    return res


if __name__ == "__main__":
    main()
