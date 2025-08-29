# Classic planarian outline driven by morphogen ranges
# ---------------------------------------------------
# - Shows morphogen ranges clearly (RGB background: m1=head, m3=trunk, m2=tail)
# - Silhouette matches the classic “spade head + slim tail” look
# - Eyes and U-shaped pharynx are included
# - Set SAVE_GIF=True to export an animated GIF (requires Pillow)
#
# Usage: run this file. Tweak schedule() to explore different shapes.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation

# ------------ PDE core (two morphogens along x ∈ [0, L]) --------------------
class RD1D:
    def __init__(self, L=1.0, Nx=500,
                 D1=0.002, D2=0.002,
                 k1=0.10, k2=0.10, k12=1.60,
                 h1=1.4, h2=1.4, seed=3):
        self.L, self.Nx = L, Nx
        self.x = np.linspace(0.0, L, Nx)
        self.dx = self.x[1] - self.x[0]
        self.D1, self.D2 = D1, D2
        self.k1, self.k2, self.k12 = k1, k2, k12
        self.h1, self.h2 = float(h1), float(h2)

        rng = np.random.default_rng(seed)
        self.m1 = 1e-4 + 1e-4*rng.standard_normal(Nx)
        self.m2 = 1e-4 + 1e-4*rng.standard_normal(Nx)

        dt_diff = 0.22*(self.dx**2)/(2.0*max(D1, D2) + 1e-12)
        dt_reac = 0.35/max(k1, k2, k12)
        self.dt = min(dt_diff, dt_reac)

    @staticmethod
    def _norm(v, eps=1e-9):
        return (v - v.min())/(v.max() - v.min() + eps)

    def _laplacian_neumann(self, u, left_slope, right_slope):
        dx = self.dx
        lap = np.zeros_like(u)
        uL = u[1] - 2*dx*left_slope
        uR = u[-2] + 2*dx*right_slope
        lap[0]    = (u[1]   - 2*u[0]   + uL   )/(dx*dx)
        lap[1:-1] = (u[2:]  - 2*u[1:-1]+ u[:-2])/(dx*dx)
        lap[-1]   = (uR     - 2*u[-1]  + u[-2])/(dx*dx)
        return lap

    def step(self, n=1, h1=None, h2=None):
        if h1 is not None: self.h1 = float(h1)
        if h2 is not None: self.h2 = float(h2)
        for _ in range(n):
            m1_Ls, m1_Rs = -self.h1/(self.D1+1e-12), 0.0
            m2_Ls, m2_Rs = 0.0,  self.h2/(self.D2+1e-12)
            lap1 = self._laplacian_neumann(self.m1, m1_Ls, m1_Rs)
            lap2 = self._laplacian_neumann(self.m2, m2_Ls, m2_Rs)
            R1 = -self.k1*self.m1 - self.k12*self.m1*self.m2
            R2 = -self.k2*self.m2 - self.k12*self.m1*self.m2
            self.m1 += self.dt*(self.D1*lap1 + R1)
            self.m2 += self.dt*(self.D2*lap2 + R2)
            self.m1 = np.clip(self.m1, 0.0, None)
            self.m2 = np.clip(self.m2, 0.0, None)

# ------------ Smooth curve helper (Cubic Hermite/Catmull–Rom) ---------------
def catmull_rom(xk, yk, x):
    xk = np.asarray(xk); yk = np.asarray(yk); x = np.asarray(x)
    n = len(xk)
    m = np.zeros_like(yk)
    m[0]  = (yk[1]-yk[0])/(xk[1]-xk[0])
    m[-1] = (yk[-1]-yk[-2])/(xk[-1]-xk[-2])
    for i in range(1, n-1):
        m[i] = (yk[i+1]-yk[i-1])/(xk[i+1]-xk[i-1])
    idx = np.searchsorted(xk, x, side='right') - 1
    idx = np.clip(idx, 0, n-2)
    x0, x1 = xk[idx], xk[idx+1]
    y0, y1 = yk[idx], yk[idx+1]
    m0, m1 = m[idx],  m[idx+1]
    h = (x1 - x0)
    t = (x - x0)/h
    h00 =  2*t**3 - 3*t**2 + 1
    h10 =      t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 =      t**3 -   t**2
    return h00*y0 + h10*h*m0 + h01*y1 + h11*h*m1

# ------------ Morphogens → classic planarian outline ------------------------
def shape_from_morphogens(x, m1, m2, L):
    s = x/L
    n1 = RD1D._norm(m1)
    n2 = RD1D._norm(m2)
    m3 = 1.0 - RD1D._norm(n1 + n2)  # trunk proxy

    # Regional averages (control the template)
    H = float(np.mean(n1[s < 0.18]))                       # head
    T = float(np.mean(n2[s > 0.82]))                       # tail
    U = float(np.mean(m3[(s > 0.30) & (s < 0.70)]))        # trunk

    # Baseline template (empirically tuned to match the classic cartoon)
    xk = np.array([0.00, 0.07, 0.13, 0.18, 0.55, 0.75, 0.92, 1.00])
    yk = np.array([0.024, 0.20, 0.34, 0.21, 0.33, 0.26, 0.10, 0.020])

    # Morphogen-driven tweaks (small but perceptible)
    yk[2] *= 1.00 + 0.45*(H-0.5)      # head width (auricles)
    yk[3] *= 1.00 - 0.40*(H-0.5)      # neck pinch
    yk[4] *= 1.00 + 0.30*(U-0.5)      # mid-body bulge
    yk[5] *= 1.00 + 0.20*(U-0.5)      # posterior trunk
    yk[6] *= 1.00 - 0.60*(T-0.5)      # pre-tail narrowing
    yk[7]  = max(0.005, yk[7]*(1.00 - 0.75*(T-0.5)))  # tip sharpness

    r = catmull_rom(xk, yk, s)

    # Soft “spade” flare & nose rounding for the cartoon look
    flare = 1.0 + (0.08 + 0.10*H)*np.exp(-((s-0.12)/0.05)**2)
    nose  = 1.0 - (0.30 + 0.30*H)*np.exp(-(s/0.045)**2)
    r = np.clip(r*flare*nose, 0.004, None)

    head_zone = (s < 0.26) & (n1 > 0.35)
    return r, head_zone, n1, n2, m3

def eyes_from_head(x, r, head_mask, L):
    s = x/L
    if np.any(head_mask):
        idxs = np.where(head_mask)[0]
        idx = int(0.60*idxs[0] + 0.40*idxs[-1])
        xe = x[idx]
    else:
        xe = 0.14*L
    re = np.interp(xe, x, r)
    ax = 0.11*re; ay = 0.17*re; dx = 0.32*re
    pr = 0.50*min(ax, ay)  # pupil radius
    return (xe/L,  ay, ax, ay, pr), (xe/L, -ay, ax, ay, pr), dx/L

def pharynx_U(x, r):
    s = x/x[-1]
    s0 = 0.58
    r0 = np.interp(s0*x[-1], x, r)
    width = 0.46*r0
    depth = 0.60*r0
    th = np.linspace(np.pi, 2*np.pi, 64)
    cx, cy = s0, -0.02*r0
    xb = cx + (width)*np.cos(th)
    yb = cy + (depth)*np.sin(th)
    xl = np.array([cx - width, cx - width])
    xr = np.array([cx + width, cx + width])
    yl = np.array([0.12*r0, yb[0]])
    yr = np.array([yb[-1], 0.12*r0])
    return np.r_[xl, xb, xr], np.r_[yl, yb, yr]

def morphogen_rgb_strip(n1, n2, m3, height=80, gamma=0.75):
    r = np.clip(n1, 0, 1)**gamma
    g = np.clip(m3, 0, 1)**gamma
    b = np.clip(n2, 0, 1)**gamma
    nx = len(n1)
    img = np.zeros((height, nx, 3))
    img[:, :, 0] = r[None, :]
    img[:, :, 1] = g[None, :]
    img[:, :, 2] = b[None, :]
    return img

# ------------ Influx schedule (clear changes across the run) -----------------
def schedule(frame, total_frames):
    q = frame/total_frames
    if q < 0.25:   return 1.4, 1.4   # balanced
    elif q < 0.50: return 2.3, 0.6   # head-dominant
    elif q < 0.75: return 0.6, 2.3   # tail-dominant
    else:          return 2.2, 2.2   # both strong

# ------------ Animate (with optional GIF export) -----------------------------
def animate(frames=180, steps_per_frame=24, SAVE_GIF=False, GIF_NAME="planarian_classic.gif", seed=5):
    L, Nx = 1.0, 500
    rd = RD1D(L=L, Nx=Nx, D1=0.002, D2=0.002,
              k1=0.10, k2=0.10, k12=1.60,
              h1=1.4, h2=1.4, seed=seed)

    rd.step(n=2600)  # relax to a clear gradient

    x = rd.x; s = x/L
    fig = plt.figure(figsize=(10.5, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.4, 1.4], hspace=0.28)
    ax  = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[1, 0])

    ax.set_xlim(0, 1); ax.set_ylim(-0.42, 0.42); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x/L"); ax.set_ylabel("half-width")
    ax.set_title("Planarian from Morphogen Ranges  (RGB background: m1=head, m3=trunk, m2=tail)")

    r, head_mask, n1, n2, m3 = shape_from_morphogens(x, rd.m1, rd.m2, L)
    bg_img = morphogen_rgb_strip(n1, n2, m3, height=90, gamma=0.75)
    bg = ax.imshow(bg_img, extent=(0, 1, -0.42, 0.42), origin="lower", interpolation="nearest", alpha=0.33, zorder=0)

    xy = np.column_stack([np.r_[s, s[::-1]], np.r_[r, -r[::-1]]])
    body = patches.Polygon(xy, closed=True, facecolor="#8b5a2b", edgecolor="k", lw=1.0, zorder=2)
    ax.add_patch(body)

    # eyes
    (cx,  ey, axw, ayw, pr), (cx2, ey2, axw2, ayw2, pr2), dx = eyes_from_head(x, r, head_mask, L)
    eyeW_L = patches.Ellipse((cx-dx,  ey), 2*axw, 2*ayw, facecolor="white", edgecolor="k", lw=0.8, zorder=3)
    eyeW_R = patches.Ellipse((cx+dx,  ey2), 2*axw2, 2*ayw2, facecolor="white", edgecolor="k", lw=0.8, zorder=3)
    eyeP_L = patches.Circle((cx-dx+0.25*axw,  ey+0.10*ayw), pr, color="k", zorder=4)
    eyeP_R = patches.Circle((cx+dx+0.25*axw2, ey2+0.10*ayw2), pr2, color="k", zorder=4)
    ax.add_patch(eyeW_L); ax.add_patch(eyeW_R); ax.add_patch(eyeP_L); ax.add_patch(eyeP_R)

    # pharynx
    XU, YU = pharynx_U(x, r)
    (lineU,) = ax.plot(XU, YU, color="k", lw=1.25, alpha=0.75, zorder=5)

    # morphogen profiles
    axp.set_xlim(0, 1); axp.set_ylim(0, 1.05); axp.set_xlabel("x/L"); axp.set_ylabel("normalized level")
    (lm1,) = axp.plot(s, n1, color='r', label="m1 (head)")
    (lm3,) = axp.plot(s, m3, color='g', label="m3 (trunk)")
    (lm2,) = axp.plot(s, n2, color='b', label="m2 (tail)")
    axp.legend(loc="upper right")

    txt = ax.text(0.02, 0.94, "", transform=ax.transAxes)

    def update(f):
        h1, h2 = schedule(f, frames)
        rd.step(n=steps_per_frame, h1=h1, h2=h2)

        r, head_mask, n1, n2, m3 = shape_from_morphogens(x, rd.m1, rd.m2, L)

        bg.set_data(morphogen_rgb_strip(n1, n2, m3, height=90, gamma=0.75))
        xy[:, 0] = np.r_[s, s[::-1]]
        xy[:, 1] = np.r_[r, -r[::-1]]
        body.set_xy(xy)

        (cx,  ey, axw, ayw, pr), (cx2, ey2, axw2, ayw2, pr2), dx = eyes_from_head(x, r, head_mask, L)
        eyeW_L.center = (cx-dx,  ey);   eyeW_L.width, eyeW_L.height = 2*axw,  2*ayw
        eyeW_R.center = (cx+dx,  ey2);  eyeW_R.width, eyeW_R.height = 2*axw2, 2*ayw2
        eyeP_L.center = (cx-dx+0.25*axw,  ey+0.10*ayw);  eyeP_L.radius = pr
        eyeP_R.center = (cx+dx+0.25*axw2, ey2+0.10*ayw2); eyeP_R.radius = pr2

        XU, YU = pharynx_U(x, r)
        lineU.set_data(XU, YU)

        lm1.set_ydata(n1); lm2.set_ydata(n2); lm3.set_ydata(m3)
        txt.set_text(f"h1={h1:.2f}, h2={h2:.2f}")
        return (bg, body, eyeW_L, eyeW_R, eyeP_L, eyeP_R, lineU, lm1, lm2, lm3, txt)

    anim = FuncAnimation(fig, update, frames=frames, interval=30, blit=False)

    if SAVE_GIF:
        from matplotlib.animation import PillowWriter
        anim.save(GIF_NAME, writer=PillowWriter(fps=30))
        print(f"Saved GIF → {GIF_NAME}")

    plt.show()
    return anim

# Run (set SAVE_GIF=True to export)
if __name__ == "__main__":
    animate(SAVE_GIF=False, frames=180, steps_per_frame=24, seed=6)
