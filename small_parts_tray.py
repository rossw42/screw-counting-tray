"""Small-parts counting tray.

Dump parts in the main well, count them and push them up the ramp and over the
ridge into the side gutter. Tilt toward the gutter spout to pour the counted
parts into a bag, then tilt toward the corner spout to pour the rest back.

Run:  .venv/bin/python small_parts_tray.py   ->  SmallPartsTray.stl
All dimensions in mm.
"""
import math
import numpy as np
import trimesh
from manifold3d import CrossSection, JoinType, Manifold, set_circular_segments

set_circular_segments(64)

# ---- parameters -----------------------------------------------------------
L = 93.0            # overall length (X)
W = 115.0           # overall width (Y)
H = 18.0            # wall height
T = 1.6             # wall thickness
FL = 2.0            # floor thickness
R_OUT = 10.0        # outer corner radius
R_FILLET = 4.0      # inside floor fillet (keeps screws out of corners)

GUTTER_W = 20.0     # inner width of the counted-parts gutter
RIDGE_W = 3.0       # ridge thickness
RIDGE_H = 5.0       # ridge height above the well floor (small screws: 4-6)
RAMP_L = 16.0       # length of the ramp up to the ridge
WALL_FRAC = 0.5     # fraction of the ridge (bag-spout end) that is a full-height
                    # wall, so uncounted parts can't follow when pouring the gutter
RAMP_TAPER = 6.25  # length over which the ramp fades into the wall corner

BAG_SPOUT_EXIT = 12.0    # inner exit width of the bag spout
BAG_SPOUT_LEN = 32.0
BAG_SPOUT_H = 9.0        # spout wall height just past the front wall
BAG_SPOUT_EXIT_H = 7.0   # spout wall height at the tip
BAG_SPOUT_DROP = 6.0     # length over which the wall drops from H to BAG_SPOUT_H

BACK_SPOUT_ENTRY = 40.0  # return spout for the leftover parts
BACK_SPOUT_EXIT = 20.0
BACK_SPOUT_LEN = 22.0
BACK_SPOUT_H = 9.0
BACK_SPOUT_EXIT_H = 7.0
BACK_SPOUT_DROP = 6.0

BOTTOM_CHAMFER = 0.6     # 45 deg chamfer on the bottom outside edge

LEAN_DEG = 35.0     # well's front wall leans inward this much from vertical, so
                    # parts collect under it instead of flying out when pouring
LEAN_BLEND = 4.0    # height over which the lean eases in above the floor fillet

# ---------------------------------------------------------------------------

X_RIDGE = L - T - GUTTER_W - RIDGE_W   # well side face of the ridge
Z0 = FL                                # floor surface
Y_BUMP = T + 5 + WALL_FRAC * (W - 2 * (T + 5))   # where the wall ends and the bump starts
BIG = 1000.0


def rrect(x0, y0, x1, y1, radii, inset=0.0):
    """Convex rounded rect via hull of corner circles; radii = (bl, br, tr, tl)."""
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    sx = [1, -1, -1, 1]
    sy = [1, 1, -1, -1]
    circles = []
    for (px, py), r, a, b in zip(pts, radii, sx, sy):
        rr = max(r - inset, 0.01)
        circles.append(CrossSection.circle(rr).translate((px + a * r, py + b * r)))
    return CrossSection.batch_hull(circles)


def filleted_cavity(x0, y0, x1, y1, radii, z_floor, z_top, rf=R_FILLET, steps=10,
                    front_lean=None):
    """Pocket with a rounded floor-to-wall fillet, built as a hull of thin slabs.

    front_lean(z) optionally moves the front (y0) wall back with height.
    """
    lean = front_lean or (lambda z: 0.0)
    slabs = []
    for i in range(steps + 1):
        a = (math.pi / 2) * i / steps
        dz = rf * (1 - math.cos(a))
        inset = rf * (1 - math.sin(a))
        sec = rrect(x0, y0, x1, y1, radii, inset)
        slabs.append(sec.extrude(0.01).translate((0, 0, z_floor + dz)))
    z_lean = [z_floor + rf + k for k in np.linspace(0, z_top - z_floor - rf, 13)]
    for z in (z_lean if front_lean else [z_top]):
        sec = rrect(x0, y0 + lean(z), x1, y1, radii)
        slabs.append(sec.extrude(0.01).translate((0, 0, z)))
    return Manifold.batch_hull(slabs)


def front_lean(z):
    """How far the well's front wall has leaned back at height z.

    Starts above the floor fillet and eases in over LEAN_BLEND so there's no
    crease, then leans at LEAN_DEG from vertical.
    """
    z0 = Z0 + R_FILLET
    k = math.tan(math.radians(LEAN_DEG))
    t = z - z0
    if t <= 0:
        return 0.0
    if t <= LEAN_BLEND:
        return k * t * t / (2 * LEAN_BLEND)
    return k * (t - LEAN_BLEND / 2)


def box(x0, y0, z0, x1, y1, z1):
    return Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def _arc(cx, cz, r, a0, a1, n=10):
    return [(cx + r * math.cos(a), cz + r * math.sin(a))
            for a in np.linspace(math.radians(a0), math.radians(a1), n)]


def _inner_section(hw, e, z_top, rf=R_FILLET):
    """Channel cross-section (x, z): filleted U grown outward by e."""
    c = hw - rf
    return (_arc(-c, Z0 + rf, rf + e, 180, 270) + _arc(c, Z0 + rf, rf + e, 270, 360)
            + [(hw + e, z_top), (-hw - e, z_top)])


def _outer_section(hw, e, z_top, rf=R_FILLET):
    """Spout shell cross-section (x, z), shrunk inward by e.

    Concentric with the channel fillet so the wall is an even thickness, but
    the curve stops at 45 deg and runs straight to the bed so it prints
    without supports.
    """
    c = hw - rf
    ro = rf + T - e
    a = _arc(-c, Z0 + rf, ro, 180, 225, 6)
    x45, z45 = a[-1]
    left = [(-hw - T + e, z_top)] + a + [(x45 + (z45 - e), e)]
    right = [(-x, z) for x, z in reversed(left)]
    return left + right


def spout(entry_w, exit_w, length, h_low, h_exit, drop, back, rim_from=0.0, tip_r=1.0):
    """Rounded, tapered, open-top spout pointing +Y, centred on x=0.

    y=0 is the inside face of the tray wall. The walls drop from H to h_low on
    a smooth S-curve over `drop`, then slope to h_exit at the tip. Wall tops
    get a half-round rim and the tip a rounded nose. `back` extends the shell
    and channel into the tray so they blend in.

    Returns (shell, channel_cut, rims); add rims after cutting the channel.
    """
    def hw(y):
        y = max(y, 0.0)
        return entry_w / 2 - (entry_w - exit_w) / 2 * y / length

    def top(y):
        if y <= 0:
            return H
        if y <= drop:
            return H - (H - h_low) * (1 - math.cos(math.pi * y / drop)) / 2
        return h_low - (h_low - h_exit) * (y - drop) / (length - drop)

    def pinch(y):
        # rounds the tip: how far the surfaces close in over the last tip_r
        s = (y - (length - tip_r)) / tip_r
        return 0.0 if s <= 0 else tip_r * (1 - math.sqrt(max(1 - s * s, 0)))

    ys = sorted(set([-back, 0.0]
                    + list(np.linspace(0, drop, 13))
                    + list(np.linspace(length - tip_r, length, 7))))

    def at(sec, y):
        return [(x, y, z) for x, z in sec]

    shell_parts, cut_parts, rim_parts = [], [], []
    for y0, y1 in zip(ys, ys[1:]):
        shell_parts.append(Manifold.hull_points(
            at(_outer_section(hw(y0), pinch(y0), top(y0) - T / 2), y0)
            + at(_outer_section(hw(y1), pinch(y1), top(y1) - T / 2), y1)))
    # channel runs a little past the tip so the end is fully open
    cut_ys = [-back - 1] + ys + [length + 3]
    for y0, y1 in zip(cut_ys, cut_ys[1:]):
        cut_parts.append(Manifold.hull_points(
            at(_inner_section(hw(y0), pinch(min(y0, length)), H + 5), y0)
            + at(_inner_section(hw(y1), pinch(min(y1, length)), H + 5), y1)))

    rim_ys = [y for y in ys if y >= rim_from]
    if rim_from not in rim_ys:
        rim_ys = [rim_from] + rim_ys
    for side in (-1, 1):
        balls = [Manifold.sphere(max(T / 2 - pinch(y), 0.05), 24)
                 .translate((side * (hw(y) + T / 2), y, top(y) - T / 2))
                 for y in rim_ys]
        for b0, b1 in zip(balls, balls[1:]):
            rim_parts.append(Manifold.batch_hull([b0, b1]))

    add = Manifold.batch_boolean
    return add(shell_parts, _op("Add")), add(cut_parts, _op("Add")), add(rim_parts, _op("Add"))


def build():
    r_in = R_OUT - T
    body = rrect(0, 0, L, W, (R_OUT,) * 4).extrude(H)

    # main well: big radii on the outside corners, tight ones at the ridge
    # back corners at the ridge are kept tight so the bump runs flush to the
    # back wall
    tight = R_FILLET + 0.1
    well = filleted_cavity(T, T, X_RIDGE, W - T, (r_in, 5, tight, r_in), Z0, H + 1,
                           front_lean=front_lean)
    gutter = filleted_cavity(X_RIDGE + RIDGE_W, T, L - T, W - T,
                             (5, r_in, r_in, tight), Z0, H + 1)

    # lower the back part of the well/gutter wall to form the ridge, all the
    # way to the back wall; the front part (toward the bag spout) stays full
    # height. The cut is wide enough to clear the corner rounding too.
    nose_r = RIDGE_W / 2
    z_nose = Z0 + RIDGE_H - nose_r
    slot = box(X_RIDGE - tight - 1, Y_BUMP, z_nose,
               X_RIDGE + RIDGE_W + tight + 1, W - T, H + 1)

    # bag spout off the front of the gutter (points -Y). Its channel has the
    # gutter's exact cross-section where they meet and runs back into the
    # gutter, so there's no lip between them.
    g_cx = X_RIDGE + RIDGE_W + GUTTER_W / 2
    bo, bi, br = spout(GUTTER_W - 0.002, BAG_SPOUT_EXIT, BAG_SPOUT_LEN, BAG_SPOUT_H,
                       BAG_SPOUT_EXIT_H, BAG_SPOUT_DROP, back=10)
    bo, bi, br = (m.rotate((0, 0, 180)).translate((g_cx, T, 0)) for m in (bo, bi, br))

    # return spout off the back-left corner of the well (points -X,+Y at 45 deg)
    c = np.array([T + r_in, W - T - r_in])
    d = np.array([-1.0, 1.0]) / math.sqrt(2)
    p = c + d * r_in
    ro, ri, rr = spout(BACK_SPOUT_ENTRY, BACK_SPOUT_EXIT, BACK_SPOUT_LEN, BACK_SPOUT_H,
                       BACK_SPOUT_EXIT_H, BACK_SPOUT_DROP, back=14)
    ro, ri, rr = (m.rotate((0, 0, 45)).translate((p[0], p[1], 0)) for m in (ro, ri, rr))

    solid = Manifold.batch_boolean([body, bo, ro], _op("Add"))
    # keep the spout walls from poking into the open cavities
    solid = solid - Manifold.batch_boolean([well, gutter, slot, bi, ri], _op("Add"))
    solid = solid + br + rr


    # ramp up to the ridge, plus a rounded nose on the ridge
    # ramp profile (x, z): foot -> tangent point on the nose -> nose centre,
    # so the slope runs smoothly onto the rounded ridge top
    cx = X_RIDGE + nose_r
    foot = np.array([X_RIDGE - RAMP_L, Z0])
    to_foot = foot - (cx, z_nose)
    phi = math.atan2(to_foot[1], to_foot[0]) - math.acos(nose_r / np.linalg.norm(to_foot))
    tan_pt = (cx + nose_r * math.cos(phi), z_nose + nose_r * math.sin(phi))
    prof = [tuple(foot), (cx, Z0), (cx, z_nose), tan_pt]

    def ramp_slice(y, s):
        # profile scaled by s about the ridge foot (cx, Z0)
        return [(cx + s * (px - cx), y, Z0 + s * (pz - Z0)) for px, pz in prof]

    ramp = Manifold.hull_points(ramp_slice(Y_BUMP - 1, 1) + ramp_slice(W, 1))
    # past the bump the ramp shrinks smoothly into the wall corner instead of
    # ending in a cliff; sin() keeps it tangent to the full ramp at Y_BUMP
    taper_pts = []
    for i in range(25):
        u = i / 24
        taper_pts += ramp_slice(Y_BUMP - RAMP_TAPER * (1 - u), math.sin(math.pi / 2 * u))
    # let the ramp sink into the solid ridge rather than stopping exactly at
    # its face; coincident faces there leave a hairline tunnel
    ridge_core = box(X_RIDGE - 0.01, T, 0, cx, W, H)
    # (the taper sits entirely inside the well/ridge, so it isn't clipped;
    # clipping it to the filleted well face leaves slivers)
    ramp = ramp ^ (well + slot + ridge_core) + Manifold.hull_points(taper_pts)
    # start the nose inside the wall so the faces don't coincide
    nose = (Manifold.cylinder(W - T - Y_BUMP + 1, nose_r)
            .rotate((-90, 0, 0))
            .translate((cx, Y_BUMP - 0.5, z_nose)))
    solid = soften_edges(solid + ramp + nose)
    # drop zero-volume slivers left where the ramp meets the well's corner
    return max(solid.decompose(), key=lambda p: p.volume())


def soften_edges(m, rim_r=T / 2, chamfer=BOTTOM_CHAMFER, step=0.1):
    """Round the top edges of the full-height walls and chamfer the bottom edge.

    Works on horizontal slices: in the top rim_r of the walls (and the bottom
    `chamfer` of the floor) each slice is eroded by the rounding profile, in
    `step`-thick layers (finer than a print layer).
    """
    z_rim = H - rim_r
    keep = m ^ box(-BIG, -BIG, chamfer, BIG, BIG, z_rim)
    layers = []
    for z in np.arange(z_rim, H - 1e-6, step):
        dz = min(z + step, H) - z_rim
        d = rim_r - math.sqrt(max(rim_r ** 2 - dz ** 2, 0))
        sec = m.slice(z + step / 2).offset(-d, JoinType.Round)
        layers.append(sec.extrude(step).translate((0, 0, z)))
    for z in np.arange(0, chamfer - 1e-6, step):
        sec = m.slice(z + step / 2).offset(-(chamfer - z), JoinType.Round)
        layers.append(sec.extrude(step).translate((0, 0, z)))
    return Manifold.batch_boolean([keep] + layers, _op("Add"))


def _op(name):
    from manifold3d import OpType
    return getattr(OpType, name)


def save(m, path):
    mesh = m.to_mesh()
    tm = trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                         faces=np.asarray(mesh.tri_verts), process=False)
    tm.export(path)
    return tm


if __name__ == "__main__":
    tm = save(build(), "SmallPartsTray.stl")
    print("bounds", tm.bounds.round(2), "watertight", tm.is_watertight,
          "volume cm3", round(tm.volume / 1000, 1))
