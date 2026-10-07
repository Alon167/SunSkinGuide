#!/usr/bin/env python3
"""Builds ONE Bambu Studio project (.3mf) with every part on ordered plates, one filament slot per colour.
usage: python make_bambu.py [--out output/BEAR_all_plates_bambu.3mf]"""
import argparse, math, os, sys, zipfile, json
import numpy as np, trimesh
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from bear import export as E
from bear.config import COLORS

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="output/BEAR_all_plates_bambu.3mf")
ap.add_argument("--clearances", default="0.15,0.10,0.20")
a = ap.parse_args()
OUT = os.path.join(ROOT, "output")
clrs = [f"{float(c):.2f}" for c in a.clearances.split(",")]

SLOT = {"brown": 1, "black": 2, "red": 3, "cream": 4, "white": 5}
NAMES = {"brown": "Brown", "black": "Black", "red": "Red", "cream": "Cream", "white": "White"}


def load(path, ground=False):
    m = trimesh.load(path)
    from bear.snap import to_man, to_tri
    m = to_tri(to_man(m))          # splits the few pinched vertices STL float32 merging creates -> 2-manifold
    if ground:
        m.apply_transform(E.ground(m))
    return m


plates = []   # (plate name, [(object name, mesh, colour, extra)])
body = load(f"{OUT}/clearance_{clrs[0]}/01_body_brown.stl")
plates.append(("01 Body (brown)", [("01_body_brown", body, "brown", dict(support=True))]))
cm = lambda n: load(f"{OUT}/common/with_brim/{n}_brim.stl")
face = [("02_muzzle_cream", cm("02_muzzle_cream"), "cream", dict(lh="0.08")),
        ("03_nose_black", cm("03_nose_black"), "black", dict(lh="0.08"))]
face += [(f"04_eye_black_{i+1}" + ("_spare" if i >= 2 else ""), cm("04_eye_black"), "black", dict(lh="0.08")) for i in range(4)]
face += [(f"05_highlight_white_{i+1}" + ("_spare" if i >= 2 else ""), cm("05_highlight_white"), "white", dict(lh="0.08")) for i in range(6)]
plates.append(("02 Face parts (cream/black/white)", face))
plates.append(("03 Snap test (any colour)", [("test_snap", load(f"{OUT}/test_snap.stl", True), "brown", {})]))
for c in clrs:
    d = f"{OUT}/clearance_{c}"
    plates.append((f"Legs clr {c} (brown)", [(f"09_leg_left_brown_c{c}", load(f"{d}/09_leg_left_brown.stl"), "brown", {}),
                                             (f"10_leg_right_brown_c{c}", load(f"{d}/10_leg_right_brown.stl"), "brown", {})]))
    blk = [(f"08_shorts_black_c{c}", load(f"{d}/08_shorts_black.stl"), "black", {})]
    if c == clrs[0] and os.path.exists(f"{d}/11_base_black.stl"):
        blk.append(("11_base_black", load(f"{d}/11_base_black.stl"), "black", {}))
    plates.append((f"Shorts{' + base' if len(blk) > 1 else ''} clr {c} (black)", blk))
    plates.append((f"Gloves clr {c} (red)", [(f"06_glove_left_red_c{c}", load(f"{d}/06_glove_left_red.stl"), "red", {}),
                                             (f"07_glove_right_red_c{c}", load(f"{d}/07_glove_right_red.stl"), "red", {})]))

BED = 256.0
cols = max(1, math.ceil(math.sqrt(len(plates))))
stride = BED * 1.2
objs = []          # dict(id,name,mesh(world),colour,extra,plate)
for pi, (pname, items) in enumerate(plates):
    ox = (pi % cols) * stride
    oy = -(pi // cols) * stride
    sizes = [(n, m.extents[0], m.extents[1]) for n, m, _, _ in items]
    pos = E.pack(sizes)
    for n, m, col, extra in items:
        mm = m.copy()
        b = mm.bounds
        mm.apply_translation([ox + pos[n][0] - b[0, 0], oy + pos[n][1] - b[0, 1], -b[0, 2]])
        objs.append(dict(id=len(objs) + 2, name=n, mesh=mm, col=col, extra=extra, plate=pi + 1))
    print(f"plate {pi+1}: {pname}: {len(items)} objects")

used = sorted({o["col"] for o in objs}, key=lambda c: SLOT[c])
fcol = ["#%02X%02X%02X" % tuple(COLORS[c]) for c in sorted(SLOT, key=SLOT.get)]

model = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
         'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">',
         ' <metadata name="BambuStudio:3mfVersion">1</metadata>',
         ' <metadata name="Application">BambuStudio-01.09.00.00</metadata>',
         ' <metadata name="Title">Boxing teddy bear - all parts</metadata>',
         ' <resources>']
for o in objs:
    model.append(f'  <object id="{o["id"]}" name="{o["name"]}" type="model">')
    model.append(E._fmt_mesh(o["mesh"]))
    model.append("  </object>")
model.append(" </resources>\n <build>")
for o in objs:
    model.append(f'  <item objectid="{o["id"]}" transform="1 0 0 0 1 0 0 0 1 0 0 0" printable="1"/>')
model.append(" </build>\n</model>")

cfg = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>"]
for o in objs:
    cfg.append(f'  <object id="{o["id"]}">')
    cfg.append(f'    <metadata key="name" value="{o["name"]}"/>')
    cfg.append(f'    <metadata key="extruder" value="{SLOT[o["col"]]}"/>')
    if o["extra"].get("lh"):
        cfg.append(f'    <metadata key="layer_height" value="{o["extra"]["lh"]}"/>')
    if o["extra"].get("support"):
        cfg.append('    <metadata key="enable_support" value="1"/>')
        cfg.append('    <metadata key="support_type" value="tree(auto)"/>')
    cfg.append('    <part id="1" subtype="normal_part">')
    cfg.append(f'      <metadata key="name" value="{o["name"]}"/>')
    cfg.append(f'      <metadata key="extruder" value="{SLOT[o["col"]]}"/>')
    cfg.append("    </part>")
    cfg.append("  </object>")
for pi, (pname, items) in enumerate(plates):
    cfg.append("  <plate>")
    cfg.append(f'    <metadata key="plater_id" value="{pi+1}"/>')
    cfg.append(f'    <metadata key="plater_name" value="{pname}"/>')
    cfg.append('    <metadata key="locked" value="false"/>')
    for o in [x for x in objs if x["plate"] == pi + 1]:
        cfg.append("    <model_instance>")
        cfg.append(f'      <metadata key="object_id" value="{o["id"]}"/>')
        cfg.append('      <metadata key="instance_id" value="0"/>')
        cfg.append(f'      <metadata key="identify_id" value="{100 + o["id"]}"/>')
        cfg.append("    </model_instance>")
    cfg.append("  </plate>")
cfg.append("</config>")

proj = {
    "printer_model": "Bambu Lab P1S", "printer_settings_id": "Bambu Lab P1S 0.4 nozzle",
    "nozzle_diameter": ["0.4"], "layer_height": "0.12",
    "filament_colour": fcol, "filament_type": ["PLA"] * 5,
    "filament_settings_id": ["Bambu PLA Basic"] * 5,
    "enable_prime_tower": "1", "different_settings_to_system": [""] * 3,
}
ct = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
      '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
      '<Default Extension="config" ContentType="text/xml"/></Types>')
rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
path = os.path.join(ROOT, a.out)
with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.writestr("[Content_Types].xml", ct)
    z.writestr("_rels/.rels", rels)
    z.writestr("3D/3dmodel.model", "\n".join(model))
    z.writestr("Metadata/model_settings.config", "\n".join(cfg))
    z.writestr("Metadata/project_settings.config", json.dumps(proj, indent=1))
print("wrote", path, os.path.getsize(path) // 1e6, "MB")
