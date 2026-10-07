"""Tiny numba z-buffer renderer (no OpenGL needed) with deferred Blinn-Phong
shading, screen-space ambient occlusion and 2x supersampling."""
import math
import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFont


@njit(cache=True)
def _raster(sv, F, vn, vc, vg, zbuf, nbuf, cbuf, gbuf):
    H, W = zbuf.shape
    for t in range(F.shape[0]):
        a, b, c = F[t, 0], F[t, 1], F[t, 2]
        x0, y0, z0 = sv[a, 0], sv[a, 1], sv[a, 2]
        x1, y1, z1 = sv[b, 0], sv[b, 1], sv[b, 2]
        x2, y2, z2 = sv[c, 0], sv[c, 1], sv[c, 2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        # back-face cull (screen space winding, y down)
        if den > 0:
            continue
        minx = max(int(math.floor(min(x0, min(x1, x2)))), 0)
        maxx = min(int(math.ceil(max(x0, max(x1, x2)))), W - 1)
        miny = max(int(math.floor(min(y0, min(y1, y2)))), 0)
        maxy = min(int(math.ceil(max(y0, max(y1, y2)))), H - 1)
        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                px = x + 0.5
                py = y + 0.5
                l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
                l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
                l2 = 1.0 - l0 - l1
                if l0 < -1e-6 or l1 < -1e-6 or l2 < -1e-6:
                    continue
                z = l0 * z0 + l1 * z1 + l2 * z2
                if z < zbuf[y, x]:
                    zbuf[y, x] = z
                    for k in range(3):
                        nbuf[y, x, k] = l0 * vn[a, k] + l1 * vn[b, k] + l2 * vn[c, k]
                        cbuf[y, x, k] = l0 * vc[a, k] + l1 * vc[b, k] + l2 * vc[c, k]
                    gbuf[y, x] = l0 * vg[a] + l1 * vg[b] + l2 * vg[c]


def look_at(eye, target, up=(0, 0, 1)):
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f])  # world->cam rows


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(ln, 1e-12)


def render(items, eye, target, fov_deg=28.0, size=(900, 1200), ss=2, bg=((18, 18, 20), (46, 46, 50)),
           ao=True, up=(0, 0, 1)):
    """items: list of dict(V,F,color(rgb 0-1),gloss(0-1)). Returns PIL image."""
    W, H = size[0] * ss, size[1] * ss
    Rm = look_at(eye, target, up)
    f = 0.5 * H / math.tan(math.radians(fov_deg) / 2)
    Vall, Fall, Nall, Call, Gall = [], [], [], [], []
    off = 0
    for it in items:
        V = np.asarray(it["V"], float)
        F = np.asarray(it["F"], np.int64)
        N = it.get("N")
        if it.get("flat"):
            fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
            fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
            V = V[F].reshape(-1, 3)
            N = np.repeat(fn, 3, axis=0)
            F = np.arange(len(V)).reshape(-1, 3)
        if N is None:
            N = vertex_normals(V, F)
        Vall.append(V)
        Fall.append(F + off)
        Nall.append(N)
        Call.append(np.tile(np.asarray(it["color"], float), (len(V), 1)))
        Gall.append(np.full(len(V), it.get("gloss", 0.2)))
        off += len(V)
    V = np.concatenate(Vall)
    F = np.concatenate(Fall)
    N = np.concatenate(Nall)
    C = np.concatenate(Call)
    G = np.concatenate(Gall)
    cam = (V - np.asarray(eye, float)) @ Rm.T
    sx = W / 2 + f * cam[:, 0] / cam[:, 2]
    sy = H / 2 - f * cam[:, 1] / cam[:, 2]
    sv = np.stack([sx, sy, cam[:, 2]], 1)
    ncam = N @ Rm.T
    zbuf = np.full((H, W), 1e9)
    nbuf = np.zeros((H, W, 3))
    cbuf = np.zeros((H, W, 3))
    gbuf = np.zeros((H, W))
    _raster(sv, F, ncam, C, G, zbuf, nbuf, cbuf, gbuf)
    hit = zbuf < 1e8
    nl = np.linalg.norm(nbuf, axis=2, keepdims=True)
    n = nbuf / np.maximum(nl, 1e-9)
    # view-space lights (x right, y up, z into screen)
    def L(v):
        v = np.array(v, float)
        return v / np.linalg.norm(v)
    key = L((-0.55, 0.65, -0.55))
    fill = L((0.8, 0.15, -0.5))
    rim = L((0.3, 0.6, 0.75))
    view = np.array([0, 0, -1.0])
    col = np.zeros((H, W, 3))
    amb = 0.22 + 0.10 * n[..., 1]
    occ = np.ones((H, W))
    if ao:
        z = np.where(hit, zbuf, np.nan)
        zf = np.where(hit, zbuf, 0)
        zmed = np.nanmedian(z) if hit.any() else 1
        acc = np.zeros((H, W))
        cnt = 0
        for r in (4 * ss, 9 * ss, 16 * ss):
            for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r), (r, r), (-r, -r), (r, -r), (-r, r)):
                zs = np.roll(np.roll(zf, dy, 0), dx, 1)
                hs = np.roll(np.roll(hit, dy, 0), dx, 1)
                diff = zbuf - zs  # >0 : neighbour closer than me
                diff = np.where(hs, diff, 0)
                scale = 0.5 * r / ss * 0.5
                acc += np.clip(diff / max(scale, 1e-6), 0, 1) * (diff < 40)
                cnt += 1
        occ = 1.0 - 0.55 * np.clip(acc / cnt, 0, 1)
    for lv, strength, spec_col in ((key, 0.85, 1.0), (fill, 0.30, 0.4), (rim, 0.22, 0.8)):
        ndl = np.clip(-(n @ lv), 0, 1)
        hv = lv + view
        hv = hv / np.linalg.norm(hv)
        ndh = np.clip((n * hv).sum(2), 0, 1)
        shin = 12 + 90 * gbuf
        spec = (ndh ** shin) * (0.04 + 0.9 * gbuf) * spec_col
        col += cbuf * ndl[..., None] * strength + spec[..., None] * strength
    col += cbuf * amb[..., None]
    col *= occ[..., None]
    col = np.clip(col, 0, 1) ** (1 / 2.2)
    # background gradient
    yy = np.linspace(0, 1, H)[:, None, None]
    b0 = np.array(bg[0]) / 255.0
    b1 = np.array(bg[1]) / 255.0
    bgi = b1 * (1 - yy) + b0 * yy
    bgi = np.broadcast_to(bgi, (H, W, 3))
    img = np.where(hit[..., None], col, bgi)
    img = (img * 255).astype(np.uint8)
    im = Image.fromarray(img)
    if ss > 1:
        im = im.resize(size, Image.LANCZOS)
    return im


def label(im, text, xy=(14, 10), size=26):
    d = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        fnt = ImageFont.load_default()
    d.text((xy[0] + 1, xy[1] + 1), text, fill=(0, 0, 0), font=fnt)
    d.text(xy, text, fill=(255, 255, 255), font=fnt)
    return im
