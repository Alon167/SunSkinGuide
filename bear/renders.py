"""Preview renders of the assembled figure (own z-buffer renderer)."""
import os
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from .render import render, label

RGB = {
    "brown": (0.78, 0.40, 0.13), "cream": (0.93, 0.80, 0.62), "black": (0.07, 0.07, 0.08),
    "red": (0.80, 0.07, 0.08), "white": (0.95, 0.95, 0.95), "gray": (0.6, 0.6, 0.62),
}
GLOSS = {"brown": 0.12, "cream": 0.12, "black": 0.7, "red": 0.8, "white": 0.5, "gray": 0.3}


def items_from(parts, explode=False, mats=None):
    items = []
    for p in parts:
        m = p["mesh"].copy()
        if explode and p.get("explode") is not None:
            m.apply_translation(p["explode"])
        g = p.get("gloss", GLOSS[p["color"]])
        items.append(dict(V=m.vertices, F=m.faces, color=RGB[p["color"]], gloss=g, flat=p.get("flat", False)))
    return items


def cam(az_deg, dist, z, tz=None):
    a = np.radians(az_deg)
    # az 0 = front (camera at -y); 90 = camera on +x side (figure faces image-left); 180 back; 270 = -x
    eye = np.array([np.sin(a) * dist, -np.cos(a) * dist, z])
    return eye


def make_renders(parts, outdir, size=(900, 1200), ss=2):
    os.makedirs(outdir, exist_ok=True)
    items = items_from(parts)
    paths = []
    ims = []
    for az in (0, 90, 180, 270):
        eye = cam(az, 760, 104)
        im = render(items, eye, (0, -6, 101), fov_deg=21, size=size, ss=ss)
        label(im, f"{az}°  " + {0: "(FRONT)", 90: "(RIGHT)", 180: "(BACK)", 270: "(LEFT)"}[az])
        p = os.path.join(outdir, f"view_{az:03d}.png")
        im.save(p)
        paths.append(p)
        ims.append(im)
    for az in (45, 135, 225, 315):
        eye = cam(az, 760, 130)
        im = render(items, eye, (0, -6, 101), fov_deg=21, size=size, ss=1)
        label(im, f"{az}°")
        im.save(os.path.join(outdir, f"view_{az:03d}.png"))
    # face close-up
    im = render(items, (0, -520, 168), (0, -30, 168), fov_deg=11, size=(1100, 1100), ss=2)
    label(im, "FACE CLOSE-UP")
    im.save(os.path.join(outdir, "face_closeup.png"))
    im3 = render(items, (160, -480, 172), (0, -30, 172), fov_deg=11, size=(1100, 1100), ss=2)
    label(im3, "FACE 3/4")
    im3.save(os.path.join(outdir, "face_closeup_34.png"))
    # fur texture close-up (shoulder / upper arm)
    im2 = render(items, (-330, -300, 140), (-52, -8, 124), fov_deg=2.6, size=(1100, 1100), ss=2)
    label(im2, "FUR TEXTURE CLOSE-UP")
    im2.save(os.path.join(outdir, "fur_closeup.png"))
    # overview sheet
    W = sum(i.size[0] for i in ims)
    sheet = Image.new("RGB", (W, size[1]))
    x = 0
    for i in ims:
        sheet.paste(i, (x, 0))
        x += i.size[0]
    sheet.save(os.path.join(outdir, "overview_4views.png"))
    return paths


def make_exploded(parts, outdir, size=(1500, 1250), ss=2):
    items = items_from(parts, explode=True)
    im = render(items, (260, -900, 140), (0, -10, 112), fov_deg=26, size=size, ss=ss)
    label(im, "EXPLODED VIEW (SNAP CONNECTIONS)")
    im.save(os.path.join(outdir, "exploded.png"))
    im = render(items, (0, -900, 112), (0, -10, 112), fov_deg=26, size=size, ss=ss)
    label(im, "EXPLODED VIEW - FRONT")
    im.save(os.path.join(outdir, "exploded_front.png"))
    # cut-away of joints: section view through x = 20 / 0 plane to show pins in sockets
    return im


def make_glove_closeups(parts, outdir, size=(1100, 1100), ss=2):
    items = items_from(parts)
    im = render(items, (260, -480, 118), (50, -40, 84), fov_deg=14, size=size, ss=ss)
    label(im, "GLOVE CLOSE-UP (3/4)")
    im.save(os.path.join(outdir, "gloves_closeup.png"))
    im = render(items, (0, -520, 100), (0, -40, 84), fov_deg=17, size=size, ss=ss)
    label(im, "GLOVES CLOSE-UP (FRONT)")
    im.save(os.path.join(outdir, "gloves_closeup_front.png"))
    im = render(items, (620, -40, 90), (40, -40, 86), fov_deg=14, size=size, ss=ss)
    label(im, "GLOVE CLOSE-UP (SIDE)")
    im.save(os.path.join(outdir, "gloves_closeup_side.png"))


# reference.png panels (x0, x1) of the top row (y 0..482)
REF_PANELS = {0: (0, 362), 90: (362, 632), 180: (632, 930), 270: (930, 1180)}


def make_comparisons(ref_path, outdir, views=(0, 90, 180, 270), h=900):
    ref = Image.open(ref_path).convert("RGB")
    for az in views:
        x0, x1 = REF_PANELS[az]
        r = ref.crop((x0, 0, x1, 482))
        r = r.resize((int(r.size[0] * h / r.size[1]), h), Image.LANCZOS)
        mine = Image.open(os.path.join(outdir, f"view_{az:03d}.png")).convert("RGB")
        mine = mine.resize((int(mine.size[0] * h / mine.size[1]), h), Image.LANCZOS)
        W = r.size[0] + mine.size[0]
        sheet = Image.new("RGB", (W, h), (20, 20, 22))
        sheet.paste(r, (0, 0))
        sheet.paste(mine, (r.size[0], 0))
        label(sheet, "REFERENCE", (10, h - 40), 24)
        label(sheet, "THIS BUILD", (r.size[0] + 10, h - 40), 24)
        sheet.save(os.path.join(outdir, f"compare_{az:03d}.png"))
