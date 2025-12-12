
# -*- coding: utf-8 -*-
# Full-range morphospace shapes (see README in code comments)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

def rd_antagonistic_1d(L=1.0, Nx=121, T=3.5, dt=None,
                       D1=0.02, D2=0.02, k1=0.08, k2=0.08, k12=2.0,
                       J1=1.0, J2=1.0, source_width=3):
    x = np.linspace(0, L, Nx); dx = x[1]-x[0]
    if dt is None: dt = 0.45*(dx**2)/(2.0*max(D1, D2))
    steps = int(T/dt)
    m1 = np.zeros(Nx); m2 = np.zeros(Nx)
    left = np.arange(source_width); right = np.arange(Nx-source_width, Nx)
    for _ in range(steps):
        m1p = np.concatenate(([m1[1]], m1, [m1[-2]]))
        m2p = np.concatenate(([m2[1]], m2, [m2[-2]]))
        lap1 = (m1p[2:] - 2*m1p[1:-1] + m1p[:-2])/(dx**2)
        lap2 = (m2p[2:] - 2*m2p[1:-1] + m2p[:-2])/(dx**2)
        annih = k12*m1*m2
        m1 += 0.0
        m1 += dt*(D1*lap1 - k1*m1 - annih)
        m2 += dt*(D2*lap2 - k2*m2 - annih)
        m1[left]  += dt*(J1/dx)/source_width
        m2[right] += dt*(J2/dx)/source_width
        m1 = np.clip(m1, 0.0, None); m2 = np.clip(m2, 0.0, None)
    return x, m1, m2

def rd_linear_1d(L=1.0, Nz=121, T=3.5, dt=None,
                 D=0.02, k=0.08, J_left=0.6, J_right=0.6, source_width=3):
    z = np.linspace(0, L, Nz); dz = z[1]-z[0]
    if dt is None: dt = 0.45*(dz**2)/(2.0*D)
    steps = int(T/dt)
    m = np.zeros(Nz)
    left = np.arange(source_width); right = np.arange(Nz-source_width, Nz)
    for _ in range(steps):
        mp = np.concatenate(([m[1]], m, [m[-2]]))
        lap = (mp[2:] - 2*mp[1:-1] + mp[:-2])/(dz**2)
        m += dt*(D*lap - k*m)
        m[left]  += dt*(J_left/dz)/source_width
        m[right] += dt*(J_right/dz)/source_width
        m = np.clip(m, 0.0, None)
    return z, m

def wavefront(x, m1, m2):
    d = m1-m2; s = np.sign(d); crossing = np.where(np.diff(s)!=0)[0]
    if crossing.size==0: return x[np.argmin(np.abs(d))]
    i = crossing[0]; x0,x1=x[i],x[i+1]; y0,y1=d[i],d[i+1]
    if y1==y0: return 0.5*(x0+x1)
    return x0 - y0*(x1-x0)/(y1-y0)

def build_shapes_grid(f2_values, f3_values,
                      D1=0.02, D2_base=0.02, D3_base=0.02,
                      k1=0.08, k2=0.08, k12=2.0, k3=0.08,
                      J1=1.0, J2=1.0, J3L=0.6, J3R=0.6):
    stats = []; cache = {}
    for f2 in f2_values:
        for f3 in f3_values:
            x,m1,m2 = rd_antagonistic_1d(D1=D1, D2=max(1e-4, D2_base*f2),
                                         k1=k1, k2=k2, k12=k12, J1=J1, J2=J2)
            z,m3    = rd_linear_1d(D=max(1e-4, D3_base*f3), k=k3, J_left=J3L, J_right=J3R)
            M1 = np.trapz(m1, x); M2 = np.trapz(m2, x); M3 = np.trapz(m3, z)
            xs = wavefront(x, m1, m2); xstar = xs/(x[-1]-x[0])
            stats.append((f2,f3,M1,M2,M3,xstar))
            cache[(f2,f3)] = (f2,f3,M1,M2,M3,xstar)
    stats = np.array(stats)
    def norm(v):
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        return np.clip((v-lo)/(hi-lo+1e-12), 0, 1)
    M1n = norm(stats[:,2]); M2n = norm(stats[:,3]); M3n = norm(stats[:,4]); Xn = norm(stats[:,5])
    statsN = {(f2,f3):(M1n[k],M2n[k],M3n[k],Xn[k])
              for k,(f2,f3,_,_,_,_) in enumerate(stats)}
    return cache, statsN

def poly_from_stats_norm(M1n,M2n,M3n,Xn, npts=120):
    h = 0.75 + 0.7*(0.5*(M1n+M2n))
    top_w = 0.35 + 0.35*(Xn-0.5)
    cheek_w = 0.45 + 0.4*M3n
    jaw_w   = 0.25 + 0.35*(1.0-M3n)
    chin    = 0.03 + 0.17*abs(M1n-M2n)

    t = np.linspace(0,np.pi,npts//2)
    x_top = top_w*np.cos(t); y_top = h*(0.6 + 0.4*np.sin(t)**1.8)
    t2 = np.linspace(0,1,npts//2)
    x_side = cheek_w*(1-0.6*t2) + jaw_w*(0.6*t2)
    y_side = 0.55*h*(1-t2) + 0.16*h*(t2)
    xr = np.concatenate([x_top, x_side]); yr = np.concatenate([y_top, y_side])
    xl = -xr[::-1]; yl = yr[::-1]
    xs = np.concatenate([xl, xr, [0.0]]); ys = np.concatenate([yl, yr, [-chin]])
    return np.vstack([xs,ys]).T

def demo():
    f2_values = np.linspace(0.2, 1.6, 10)
    f3_values = np.linspace(0.2, 1.6, 10)
    raw, normed = build_shapes_grid(f2_values, f3_values)
    fig, axes = plt.subplots(len(f2_values), len(f3_values), figsize=(14,14))
    for i,f2 in enumerate(f2_values):
        for j,f3 in enumerate(f3_values):
            M1n,M2n,M3n,Xn = normed[(f2,f3)]
            poly = poly_from_stats_norm(M1n,M2n,M3n,Xn)
            ax = axes[i,j]
            ax.add_patch(Polygon(poly, closed=True, fill=False))
            eye_sep = 0.18 + 0.30*(Xn-0.5)
            ax.add_patch(Circle((-eye_sep, 0.25), 0.028, fill=True))
            ax.add_patch(Circle(( eye_sep, 0.25), 0.028, fill=True))
            ax.set_aspect('equal', 'box')
            ax.set_xlim(-1.1,1.1); ax.set_ylim(-0.35,1.35)
            ax.set_xticks([]); ax.set_yticks([])
            if j==0: ax.set_ylabel(f"f2={f2:.2f}")
            if i==0: ax.set_title(f"f3={f3:.2f}")
    plt.suptitle("Full-range shape morphospace (f2=↓/↑D2, f3=↓/↑D3)", y=0.92)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo()
