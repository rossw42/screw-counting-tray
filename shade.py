"""Ray-traced orthographic views: python shade.py in.stl out.png"""
import sys, numpy as np, trimesh, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
m = trimesh.load(sys.argv[1]); m.merge_vertices(digits_vertex=4)
def view(ax, az, el, center, half, res=260):
    az, el = np.radians(az), np.radians(el)
    d = -np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    up = np.array([0,0,1.0]); r = np.cross(d, up); r/=np.linalg.norm(r); u = np.cross(r, d)
    g = np.linspace(-half, half, res); U,V = np.meshgrid(g, g[::-1])
    o = np.asarray(center) - d*300 + U.reshape(-1,1)*r + V.reshape(-1,1)*u
    tri, ray = [], []
    for k in range(0, len(o), 8000):
        t_, r_ = m.ray.intersects_id(o[k:k+8000], np.tile(d,(len(o[k:k+8000]),1)), multiple_hits=False)[:2]
        tri.append(t_); ray.append(r_+k)
    tri, ray = np.concatenate(tri), np.concatenate(ray)
    img = np.ones(len(o))
    n = m.face_normals[tri]; n[(n@d)>0] *= -1
    L1 = np.array([0.4,-0.6,0.7]); L1/=np.linalg.norm(L1)
    img[ray] = 0.25 + 0.55*np.clip(n@L1,0,1) + 0.2*np.clip(-(n@d),0,1)
    ax.imshow(img.reshape(res,res), cmap='gray', vmin=0, vmax=1); ax.set_axis_off()
views = [(-60,30,(82,-10,6),28), (200,35,(82,-10,6),28), (-90,10,(82,-10,6),25),
         (120,30,(5,115,8),35), (-150,35,(46,57,6),75), (60,55,(46,57,6),75)]
fig, axs = plt.subplots(2,3, figsize=(18,12))
for ax, v in zip(axs.ravel(), views): view(ax, *v)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=60)
