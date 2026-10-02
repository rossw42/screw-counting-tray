import trimesh, numpy as np, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
m = trimesh.load(sys.argv[1], process=False); b=m.bounds
xs=np.arange(b[0,0],b[1,0],0.5); ys=np.arange(b[0,1],b[1,1],0.5)
X,Y=np.meshgrid(xs,ys); o=np.c_[X.ravel(),Y.ravel(),np.full(X.size,50)]
loc,ri,_=m.ray.intersects_location(o,np.tile([0,0,-1],(len(o),1)),multiple_hits=False)
Z=np.full(X.size,np.nan); Z[ri]=loc[:,2]; Z=Z.reshape(X.shape)
fig,ax=plt.subplots(2,3,figsize=(20,11)); ax=ax.ravel()
im=ax[0].imshow(Z,origin='lower',extent=[xs[0],xs[-1],ys[0],ys[-1]],cmap='terrain'); plt.colorbar(im,ax=ax[0]); ax[0].set_title('top height map (mm)')
ls=matplotlib.colors.LightSource(315,45); ax[1].imshow(ls.shade(np.nan_to_num(Z,nan=-2),cmap=plt.cm.Blues,vert_exag=3),origin='lower',extent=[xs[0],xs[-1],ys[0],ys[-1]]); ax[1].set_title('hillshade')
for a,(o_,n_,ix,t) in zip(ax[2:],[((0,57,0),(0,1,0),0,"y=57"),((0,-5,0),(0,1,0),0,"y=-5 (in spout)"),((81.4,0,0),(1,0,0),1,"x=81.4 (gutter+bag spout)"),((35,0,0),(1,0,0),1,"x=35 (leaning front wall)")]):
    s=m.section(plane_origin=o_,plane_normal=n_)
    for e in s.discrete: a.plot(e[:,ix],e[:,2],'k-',lw=0.8)
    a.set_aspect('equal'); a.grid(True); a.set_title(t)
plt.tight_layout(); plt.savefig(sys.argv[2],dpi=65)
