"""Render a multi-angle product shot of an STL with a small numpy z-buffer.

Run:  .venv/bin/python render_views.py SmallPartsTray.stl tray_views.png
"""
import sys

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = 900                                   # pixels per view (square)
SS = 2                                      # supersampling for smooth edges
COLOR = np.array([0.95, 0.55, 0.20])        # PLA orange
BG_TOP, BG_BOT = np.array([0.97, 0.97, 0.98]), np.array([0.86, 0.88, 0.91])
VIEWS = [  # (azimuth deg, elevation deg, title)
    (-60, 35, "Front right"),
    (130, 40, "Back left"),
    (-90, 89.9, "Top"),
    (-90, 22, "Front (bag spout)"),
]


def camera(az, el):
    az, el = np.radians(az), np.radians(el)
    fwd = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fwd, right, up


def render(mesh, az, el):
    fwd, right, up = camera(az, el)
    v = mesh.vertices - mesh.bounds.mean(axis=0)
    # orthographic projection, fit to frame with a margin
    px, py, depth = v @ right, v @ up, v @ fwd
    px, py = px - (px.min() + px.max()) / 2, py - (py.min() + py.max()) / 2
    span = max(np.ptp(px), np.ptp(py)) * 1.08
    R = RES * SS
    sx = (px / span + 0.5) * R
    sy = (0.5 - py / span) * R

    # light: key from upper left, fill from the camera, rim from behind
    key = -fwd * 0.3 + up * 0.8 - right * 0.5
    key /= np.linalg.norm(key)
    n = np.nan_to_num(mesh.vertex_normals)
    lam = 0.28 + 0.62 * np.clip(n @ key, 0, 1) + 0.22 * np.clip(n @ -fwd, 0, 1)
    spec = np.clip(n @ ((key - fwd) / np.linalg.norm(key - fwd)), 0, 1) ** 40 * 0.35

    zbuf = np.full((R, R), np.inf)
    shade = np.zeros((R, R))
    hi = np.zeros((R, R))
    for f in mesh.faces:
        x, y, z = sx[f], sy[f], depth[f]
        x0, x1 = int(max(x.min(), 0)), int(min(x.max() + 1, R))
        y0, y1 = int(max(y.min(), 0)), int(min(y.max() + 1, R))
        if x0 >= x1 or y0 >= y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        d = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(d) < 1e-12:
            continue
        w0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / d
        w1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        zz = w0 * z[0] + w1 * z[1] + w2 * z[2]
        sub = zbuf[y0:y1, x0:x1]
        win = inside & (zz < sub)
        sub[win] = zz[win]
        shade[y0:y1, x0:x1][win] = (w0 * lam[f[0]] + w1 * lam[f[1]] + w2 * lam[f[2]])[win]
        hi[y0:y1, x0:x1][win] = (w0 * spec[f[0]] + w1 * spec[f[1]] + w2 * spec[f[2]])[win]

    hit = np.isfinite(zbuf)
    t = np.linspace(0, 1, R)[:, None, None]
    img = BG_TOP * (1 - t) + BG_BOT * t
    img = np.broadcast_to(img, (R, R, 3)).copy()
    obj = np.clip(COLOR * shade[..., None] + hi[..., None], 0, 1)
    img[hit] = obj[hit]

    # dark outline where depth jumps (silhouettes and creases between parts)
    zf = np.where(hit, zbuf, zbuf[hit].max() + span)
    edge = np.zeros_like(hit)
    thresh = span * 0.012
    edge[1:, :] |= np.abs(np.diff(zf, axis=0)) > thresh
    edge[:, 1:] |= np.abs(np.diff(zf, axis=1)) > thresh
    img[edge] *= 0.35
    return img.reshape(RES, SS, RES, SS, 3).mean(axis=(1, 3))


def main(src, dst):
    mesh = trimesh.load(src)
    mesh.merge_vertices(digits_vertex=4)
    # smooth the curved surfaces (facets are ~6 deg apart) but keep creases
    # like the ramp's edges crisp
    mesh = trimesh.graph.smooth_shade(mesh, angle=np.radians(15))
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), facecolor="white")
    for ax, (az, el, title) in zip(axs.ravel(), VIEWS):
        ax.imshow(render(mesh, az, el))
        ax.set_title(title, fontsize=13, color="#333")
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(dst, dpi=110, facecolor="white")


if __name__ == "__main__":
    main(*sys.argv[1:3])
