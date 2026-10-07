"""Signed-distance-function kit (numba).

A `Program` is an ordered list of primitives combined with smooth union /
smooth subtraction / intersection.  It is evaluated by a numba kernel, either
directly (`Program.eval`) or on a dense grid with a narrow band around the
surface (`sample_field`), where optional fur bumps are added.
"""
import math
import numpy as np
from numba import njit, prange

ELL, CAP, RBOX, TORUS, PLANE, ECYL, ERING = range(7)
UNION, SUB, INTER = range(3)


# ----------------------------------------------------------------- helpers
def frame(z_axis, hint=(0.0, -1.0, 0.0)):
    """Orthonormal frame; returns 3x3 matrix whose ROWS are local x,y,z axes."""
    z = np.asarray(z_axis, float)
    z = z / np.linalg.norm(z)
    h = np.asarray(hint, float)
    x = h - z * np.dot(h, z)
    if np.linalg.norm(x) < 1e-6:
        h = np.array([1.0, 0.0, 0.0])
        x = h - z * np.dot(h, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def rot_axes(rx=0.0, ry=0.0, rz=0.0):
    """Rows = local axes after rotating by rx,ry,rz degrees (applied x,y,z)."""
    a, b, c = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx          # local->world columns
    return R.T                # world->local rows


class Program:
    def __init__(self):
        self.rows = []

    def _add(self, t, op, k, c=(0, 0, 0), M=None, r=(0, 0, 0, 0), a=(0, 0, 0), b=(0, 0, 0)):
        M = np.eye(3) if M is None else np.asarray(M, float)
        r = list(r) + [0.0] * (4 - len(r))
        self.rows.append((t, op, k, np.asarray(c, float), M, np.asarray(r, float),
                          np.asarray(a, float), np.asarray(b, float)))
        return self

    # --- primitives
    def ell(self, c, r, M=None, op=UNION, k=3.0):
        return self._add(ELL, op, k, c=c, M=M, r=r)

    def cap(self, a, b, ra, rb=None, op=UNION, k=3.0):
        return self._add(CAP, op, k, a=a, b=b, r=(ra, ra if rb is None else rb))

    def rbox(self, c, half, rr, M=None, op=UNION, k=3.0):
        return self._add(RBOX, op, k, c=c, M=M, r=(*half, rr))

    def torus(self, c, R, r, M=None, op=UNION, k=0.5):
        return self._add(TORUS, op, k, c=c, M=M, r=(R, r))

    def plane(self, n, off, op=INTER, k=0.0):
        n = np.asarray(n, float)
        n = n / np.linalg.norm(n)
        return self._add(PLANE, op, k, a=n, r=(off,))

    def ecyl(self, c, a, b, hh, rr, op=UNION, k=3.0, M=None):
        return self._add(ECYL, op, k, c=c, M=M, r=(a, b, hh, rr))

    def ering(self, c, a, b, tube, op=UNION, k=0.5, M=None):
        return self._add(ERING, op, k, c=c, M=M, r=(a, b, tube))

    # --- packing
    def pack(self):
        n = len(self.rows)
        T = np.array([r[0] for r in self.rows], np.int32)
        OP = np.array([r[1] for r in self.rows], np.int32)
        K = np.array([r[2] for r in self.rows], np.float64)
        C = np.array([r[3] for r in self.rows], np.float64).reshape(n, 3)
        M = np.array([r[4] for r in self.rows], np.float64).reshape(n, 3, 3)
        R = np.array([r[5] for r in self.rows], np.float64).reshape(n, 4)
        A = np.array([r[6] for r in self.rows], np.float64).reshape(n, 3)
        B = np.array([r[7] for r in self.rows], np.float64).reshape(n, 3)
        return T, OP, K, C, M, R, A, B

    def eval(self, P, both=False):
        """Evaluate at points P (N,3).  both=True also returns the field
        without subtractions (used to measure groove depth)."""
        P = np.ascontiguousarray(P, np.float64).reshape(-1, 3)
        d, dn = _eval_many(P, *self.pack())
        return (d, dn) if both else d

    def grad(self, P, eps=0.05):
        P = np.ascontiguousarray(P, np.float64).reshape(-1, 3)
        g = np.zeros_like(P)
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            g[:, i] = (self.eval(P + e) - self.eval(P - e)) / (2 * eps)
        return g

    def project(self, P, iters=4):
        """Project points onto the zero level set (Newton)."""
        P = np.array(P, float)
        for _ in range(iters):
            d = self.eval(P)
            g = self.grad(P)
            gn = np.maximum((g * g).sum(1), 1e-9)
            P = P - (d / gn)[:, None] * g
        return P

    def raymarch(self, o, d, tmax=300.0, step=0.5):
        """First point where the field becomes <=0 along ray o + t d."""
        o = np.asarray(o, float)
        d = np.asarray(d, float)
        d = d / np.linalg.norm(d)
        ts = np.arange(0, tmax, step)
        vals = self.eval(o[None, :] + ts[:, None] * d[None, :])
        idx = np.nonzero(vals <= 0)[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        lo, hi = ts[max(i - 1, 0)], ts[i]
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.eval((o + mid * d)[None, :])[0] <= 0:
                hi = mid
            else:
                lo = mid
        return o + hi * d


# ------------------------------------------------------------ numba kernels
@njit(inline="always")
def _smin(a, b, k):
    if k <= 0.0:
        return a if a < b else b
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25


@njit(inline="always")
def _smax(a, b, k):
    return -_smin(-a, -b, k)


@njit(inline="always")
def _prim(i, px, py, pz, T, C, M, R, A, B):
    t = T[i]
    if t == 4:  # plane
        return A[i, 0] * px + A[i, 1] * py + A[i, 2] * pz - R[i, 0]
    if t == 1:  # capsule / round cone
        bax = B[i, 0] - A[i, 0]
        bay = B[i, 1] - A[i, 1]
        baz = B[i, 2] - A[i, 2]
        pax = px - A[i, 0]
        pay = py - A[i, 1]
        paz = pz - A[i, 2]
        bb = bax * bax + bay * bay + baz * baz
        h = (pax * bax + pay * bay + paz * baz) / bb
        h = min(max(h, 0.0), 1.0)
        dx = pax - bax * h
        dy = pay - bay * h
        dz = paz - baz * h
        return math.sqrt(dx * dx + dy * dy + dz * dz) - (R[i, 0] + (R[i, 1] - R[i, 0]) * h)
    qx = px - C[i, 0]
    qy = py - C[i, 1]
    qz = pz - C[i, 2]
    lx = M[i, 0, 0] * qx + M[i, 0, 1] * qy + M[i, 0, 2] * qz
    ly = M[i, 1, 0] * qx + M[i, 1, 1] * qy + M[i, 1, 2] * qz
    lz = M[i, 2, 0] * qx + M[i, 2, 1] * qy + M[i, 2, 2] * qz
    if t == 0:  # ellipsoid
        rx, ry, rz = R[i, 0], R[i, 1], R[i, 2]
        k0 = math.sqrt((lx / rx) ** 2 + (ly / ry) ** 2 + (lz / rz) ** 2)
        k1 = math.sqrt((lx / (rx * rx)) ** 2 + (ly / (ry * ry)) ** 2 + (lz / (rz * rz)) ** 2)
        if k1 < 1e-12:
            return -min(rx, min(ry, rz))
        return k0 * (k0 - 1.0) / k1
    if t == 2:  # rounded box
        rr = R[i, 3]
        ax = abs(lx) - (R[i, 0] - rr)
        ay = abs(ly) - (R[i, 1] - rr)
        az = abs(lz) - (R[i, 2] - rr)
        out = math.sqrt(max(ax, 0.0) ** 2 + max(ay, 0.0) ** 2 + max(az, 0.0) ** 2)
        return out + min(max(ax, max(ay, az)), 0.0) - rr
    if t == 3:  # torus (axis = local z)
        q = math.sqrt(lx * lx + ly * ly) - R[i, 0]
        return math.sqrt(q * q + lz * lz) - R[i, 1]
    if t == 5:  # elliptic cylinder, rounded
        a, b, hh, rr = R[i, 0], R[i, 1], R[i, 2], R[i, 3]
        d2 = (math.sqrt((lx / a) ** 2 + (ly / b) ** 2) - 1.0) * min(a, b)
        qx2 = d2 + rr
        qy2 = abs(lz) - hh + rr
        out = math.sqrt(max(qx2, 0.0) ** 2 + max(qy2, 0.0) ** 2)
        return out + min(max(qx2, qy2), 0.0) - rr
    # t == 6 : elliptic torus
    a, b, tube = R[i, 0], R[i, 1], R[i, 2]
    d2 = (math.sqrt((lx / a) ** 2 + (ly / b) ** 2) - 1.0) * min(a, b)
    return math.sqrt(d2 * d2 + lz * lz) - tube


@njit
def _eval_one(px, py, pz, T, OP, K, C, M, R, A, B):
    d = 1e9
    dn = 1e9
    for i in range(T.shape[0]):
        di = _prim(i, px, py, pz, T, C, M, R, A, B)
        op = OP[i]
        k = K[i]
        if op == 0:
            d = _smin(d, di, k)
            dn = _smin(dn, di, k)
        elif op == 1:
            d = _smax(d, -di, k)
        else:
            d = _smax(d, di, k)
            dn = _smax(dn, di, k)
    return d, dn


@njit(parallel=True)
def _eval_many(P, T, OP, K, C, M, R, A, B):
    n = P.shape[0]
    d = np.empty(n)
    dn = np.empty(n)
    for i in prange(n):
        a, b = _eval_one(P[i, 0], P[i, 1], P[i, 2], T, OP, K, C, M, R, A, B)
        d[i] = a
        dn[i] = b
    return d, dn


# --------------------------------------------------------------- fur noise
@njit(inline="always")
def _hash3(ix, iy, iz):
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647) & 0x7FFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0x7FFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFF) / 65535.0


@njit(inline="always")
def _vnoise(x, y, z):
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    iz = int(math.floor(z))
    fx = x - ix
    fy = y - iy
    fz = z - iz
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    fz = fz * fz * (3 - 2 * fz)
    v = 0.0
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                v += w * _hash3(ix + dx, iy + dy, iz + dz)
    return v


@njit(inline="always")
def _fbm(x, y, z):
    return 0.6 * _vnoise(x * 0.9, y * 0.9, z * 0.9) + 0.4 * _vnoise(x * 2.1 + 17.0, y * 2.1, z * 2.1 + 5.0) - 0.5


# ----------------------------------------------------------- grid sampling
@njit(parallel=True)
def _field_kernel(out, o, h, Dc, oc, hc, band, T, OP, K, C, M, R, A, B,
                  nb, BC, BR, BF, BE, BW, cell_start, cell_items, cell_o, cell_s, cdim,
                  noise_amp, bump_k):
    nx, ny, nz = out.shape
    cx, cy, cz = Dc.shape
    for i in prange(nx):
        px = o[0] + i * h
        for j in range(ny):
            py = o[1] + j * h
            for k in range(nz):
                pz = o[2] + k * h
                # trilinear lookup of the coarse field
                fx = (px - oc[0]) / hc
                fy = (py - oc[1]) / hc
                fz = (pz - oc[2]) / hc
                ix = min(max(int(math.floor(fx)), 0), cx - 2)
                iy = min(max(int(math.floor(fy)), 0), cy - 2)
                iz = min(max(int(math.floor(fz)), 0), cz - 2)
                tx = min(max(fx - ix, 0.0), 1.0)
                ty = min(max(fy - iy, 0.0), 1.0)
                tz = min(max(fz - iz, 0.0), 1.0)
                c00 = Dc[ix, iy, iz] * (1 - tx) + Dc[ix + 1, iy, iz] * tx
                c10 = Dc[ix, iy + 1, iz] * (1 - tx) + Dc[ix + 1, iy + 1, iz] * tx
                c01 = Dc[ix, iy, iz + 1] * (1 - tx) + Dc[ix + 1, iy, iz + 1] * tx
                c11 = Dc[ix, iy + 1, iz + 1] * (1 - tx) + Dc[ix + 1, iy + 1, iz + 1] * tx
                dc = (c00 * (1 - ty) + c10 * ty) * (1 - tz) + (c01 * (1 - ty) + c11 * ty) * tz
                if abs(dc) > band:
                    out[i, j, k] = dc
                    continue
                d, dn = _eval_one(px, py, pz, T, OP, K, C, M, R, A, B)
                if nb > 0 and d < 1.6 and d > -1.2:
                    # fur bumps from hashed grid
                    gx = int((px - cell_o[0]) / cell_s)
                    gy = int((py - cell_o[1]) / cell_s)
                    gz = int((pz - cell_o[2]) / cell_s)
                    best = 1e9
                    wsum = 0.0
                    for ax in range(max(gx - 1, 0), min(gx + 2, cdim[0])):
                        for ay in range(max(gy - 1, 0), min(gy + 2, cdim[1])):
                            for az in range(max(gz - 1, 0), min(gz + 2, cdim[2])):
                                cid = (ax * cdim[1] + ay) * cdim[2] + az
                                for s in range(cell_start[cid], cell_start[cid + 1]):
                                    b = cell_items[s]
                                    dx = px - BC[b, 0]
                                    dy = py - BC[b, 1]
                                    dz = pz - BC[b, 2]
                                    al = dx * BF[b, 0] + dy * BF[b, 1] + dz * BF[b, 2]
                                    rest = dx * dx + dy * dy + dz * dz - al * al
                                    q = math.sqrt((al / BE[b]) ** 2 + max(rest, 0.0))
                                    db = (q - BR[b]) * min(1.0, 1.0 / BE[b] + 0.15)
                                    if db < best:
                                        best = db
                                    if db < 0.8:
                                        wsum = max(wsum, BW[b])
                    if best < 1e8:
                        d = _smin(d, best, bump_k)
                    if noise_amp > 0.0 and wsum > 0.0:
                        d += noise_amp * wsum * _fbm(px, py, pz)
                out[i, j, k] = d


def build_bump_grid(BC, cell=2.0):
    lo = BC.min(0) - 1.0
    hi = BC.max(0) + 1.0
    dim = np.maximum(np.ceil((hi - lo) / cell).astype(np.int64), 1)
    idx = np.floor((BC - lo) / cell).astype(np.int64)
    idx = np.minimum(np.maximum(idx, 0), dim - 1)
    cid = (idx[:, 0] * dim[1] + idx[:, 1]) * dim[2] + idx[:, 2]
    order = np.argsort(cid, kind="stable")
    counts = np.bincount(cid, minlength=int(dim.prod()))
    start = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(counts, out=start[1:])
    return lo, dim, start, order.astype(np.int64)


def sample_field(prog, lo, hi, h, bumps=None, band=2.6, noise_amp=0.0, bump_k=0.14, hc=0.9):
    """Dense float32 field on grid lo + h*(i,j,k) covering [lo,hi]."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    n = np.ceil((hi - lo) / h).astype(int) + 1
    # coarse grid
    nc = np.ceil((hi - lo) / hc).astype(int) + 3
    oc = lo - hc
    gx, gy, gz = [oc[a] + hc * np.arange(nc[a]) for a in range(3)]
    G = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)
    Dc = prog.eval(G).reshape(nc[0], nc[1], nc[2])
    T, OP, K, C, M, R, A, B = prog.pack()
    out = np.empty(n, np.float32)
    if bumps is not None and len(bumps["BC"]):
        BC = np.ascontiguousarray(bumps["BC"], np.float64)
        cell = 2.0
        cell_o, cdim, cell_start, cell_items = build_bump_grid(BC, cell)
        nb = len(BC)
        args = (nb, BC, np.ascontiguousarray(bumps["BR"], np.float64),
                np.ascontiguousarray(bumps["BF"], np.float64),
                np.ascontiguousarray(bumps["BE"], np.float64),
                np.ascontiguousarray(bumps["BW"], np.float64),
                cell_start, cell_items, cell_o, cell, cdim)
    else:
        z3 = np.zeros((1, 3))
        z1 = np.zeros(1)
        args = (0, z3, z1, z3, np.ones(1), z1, np.zeros(2, np.int64), np.zeros(1, np.int64),
                np.zeros(3), 2.0, np.ones(3, np.int64))
    _field_kernel(out, lo, h, Dc, oc, hc, band, T, OP, K, C, M, R, A, B, *args,
                  noise_amp, bump_k)
    return out


# ----------------------------------------------------------- bump seeding
@njit
def _dart(P, R, order, cell, lo, dim, fac):
    n = P.shape[0]
    head = -np.ones(int(dim[0] * dim[1] * dim[2]), np.int64)
    nxt = -np.ones(n, np.int64)
    acc = np.zeros(n, np.bool_)
    for t in range(order.shape[0]):
        i = order[t]
        gx = int((P[i, 0] - lo[0]) / cell)
        gy = int((P[i, 1] - lo[1]) / cell)
        gz = int((P[i, 2] - lo[2]) / cell)
        ok = True
        for ax in range(max(gx - 1, 0), min(gx + 2, dim[0])):
            if not ok:
                break
            for ay in range(max(gy - 1, 0), min(gy + 2, dim[1])):
                if not ok:
                    break
                for az in range(max(gz - 1, 0), min(gz + 2, dim[2])):
                    j = head[(ax * dim[1] + ay) * dim[2] + az]
                    while j >= 0:
                        dx = P[i, 0] - P[j, 0]
                        dy = P[i, 1] - P[j, 1]
                        dz = P[i, 2] - P[j, 2]
                        m = fac * (R[i] + R[j])
                        if dx * dx + dy * dy + dz * dz < m * m:
                            ok = False
                            break
                        j = nxt[j]
                    if not ok:
                        break
        if ok:
            acc[i] = True
            cid = (gx * dim[1] + gy) * dim[2] + gz
            nxt[i] = head[cid]
            head[cid] = i
    return acc


def poisson_select(P, R, rng, fac=0.78, cell=1.6):
    lo = P.min(0) - 0.5
    hi = P.max(0) + 0.5
    dim = np.maximum(np.ceil((hi - lo) / cell).astype(np.int64), 1)
    order = rng.permutation(len(P)).astype(np.int64)
    return _dart(np.ascontiguousarray(P), np.ascontiguousarray(R), order, cell, lo, dim, fac)
