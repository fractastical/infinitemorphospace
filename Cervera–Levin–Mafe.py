"""
Morphospace reaction–diffusion demo inspired by Cervera–Levin–Mafe (2021).

- m1, m2: antagonistic morphogens on an antero–posterior (AP) axis with
  diffusion, degradation, and mutual annihilation (A + B -> ∅).
  Boundary fluxes inject m1 at the left and m2 at the right.
- m3: independent morphogen on a lateral axis with diffusion + degradation,
  injected from both sides.
- Gap-junction blocking is modeled by reducing D2 and D3 with factors f2, f3.
- We sweep (f2, f3) and plot:
  (1) example AP profiles with wavefront x* where m1 = m2,
  (2) a heatmap of x* across (f2, f3),
  (3) a 3D morphospace of mean expressions (<m1>, <m2>, <m3>) colored by x*.

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def rd_antagonistic_1d(
    L=1.0, Nx=121, T=3.0, dt=None,
    D1=0.02, D2=0.02, k1=0.08, k2=0.08, k12=2.0,
    J1=1.0, J2=1.0, source_width=3, seed=None
):
    """Two-species RD with mutual annihilation on [0, L] and flux at opposite ends."""
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, L, Nx)
    dx = x[1] - x[0]
    if dt is None:
        dt = 0.45 * (dx**2) / (2.0 * max(D1, D2))  # explicit stability
    steps = int(T / dt)

    m1 = np.zeros(Nx)
    m2 = np.zeros(Nx)
    left_idx = np.arange(source_width)
    right_idx = np.arange(Nx - source_width, Nx)

    for _ in range(steps):
        # Neumann (zero-flux) BCs by reflection
        m1_pad = np.concatenate(([m1[1]], m1, [m1[-2]]))
        m2_pad = np.concatenate(([m2[1]], m2, [m2[-2]]))
        lap1 = (m1_pad[2:] - 2*m1_pad[1:-1] + m1_pad[:-2]) / (dx**2)
        lap2 = (m2_pad[2:] - 2*m2_pad[1:-1] + m2_pad[:-2]) / (dx**2)

        annih = k12 * m1 * m2
        m1 += dt * (D1*lap1 - k1*m1 - annih)
        m2 += dt * (D2*lap2 - k2*m2 - annih)

        # flux sources near edges
        m1[left_idx] += dt * (J1 / dx) / source_width
        m2[right_idx] += dt * (J2 / dx) / source_width

        m1 = np.clip(m1, 0.0, None)
        m2 = np.clip(m2, 0.0, None)

    return x, m1, m2


def rd_linear_1d(
    L=1.0, Nz=121, T=3.0, dt=None,
    D=0.02, k=0.08, J_left=0.5, J_right=0.5, source_width=3
):
    """Single-species RD with bilateral flux sources on [0, L]."""
    z = np.linspace(0, L, Nz)
    dz = z[1] - z[0]
    if dt is None:
        dt = 0.45 * (dz**2) / (2.0 * D)
    steps = int(T / dt)

    m = np.zeros(Nz)
    left_idx = np.arange(source_width)
    right_idx = np.arange(Nz - source_width, Nz)

    for _ in range(steps):
        m_pad = np.concatenate(([m[1]], m, [m[-2]]))
        lap = (m_pad[2:] - 2*m_pad[1:-1] + m_pad[:-2]) / (dz**2)
        m += dt * (D*lap - k*m)

        m[left_idx] += dt * (J_left / dz) / source_width
        m[right_idx] += dt * (J_right / dz) / source_width

        m = np.clip(m, 0.0, None)

    return z, m


def compute_wavefront_position(x, m1, m2):
    """Return position where m1 == m2 (linear interp around first zero of m1-m2)."""
    diff = m1 - m2
    sign = np.sign(diff)
    crossings = np.where(np.diff(sign) != 0)[0]
    if len(crossings) == 0:
        return x[np.argmin(np.abs(diff))]
    i = crossings[0]
    x0, x1 = x[i], x[i+1]
    y0, y1 = diff[i], diff[i+1]
    if y1 == y0:
        return 0.5*(x0 + x1)
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def demo():
    # --- base params (dimensionless) ---
    L = 1.0
    Nx = 121
    Nz = 121
    k1 = k2 = 0.08
    k12 = 2.0
    k3 = 0.08
    D1 = 0.02
    D2_base = 0.02
    D3_base = 0.02
    J1 = 1.0
    J2 = 1.0
    J3_left = 0.6
    J3_right = 0.6

    # Gap-junction blocking factors for D2 and D3
    f2_values = np.linspace(1.0, 0.4, 8)
    f3_values = np.linspace(1.0, 0.4, 8)

    records = []
    profiles_example = {}

    for i, f2 in enumerate(f2_values):
        for j, f3 in enumerate(f3_values):
            D2 = D2_base * f2
            D3 = D3_base * f3

            x, m1, m2 = rd_antagonistic_1d(
                L=L, Nx=Nx, D1=D1, D2=D2,
                k1=k1, k2=k2, k12=k12, J1=J1, J2=J2, T=3.5
            )
            z, m3 = rd_linear_1d(
                L=L, Nz=Nz, D=D3, k=k3, J_left=J3_left, J_right=J3_right, T=3.5
            )

            M1 = np.trapz(m1, x) / L  # mean expressions
            M2 = np.trapz(m2, x) / L
            M3 = np.trapz(m3, z) / L
            x_star = compute_wavefront_position(x, m1, m2) / L  # normalize

            records.append([f2, f3, M1, M2, M3, x_star])

            if (i == len(f2_values)//2) and (j == len(f3_values)//2):
                profiles_example = {
                    "x": x, "m1": m1, "m2": m2, "z": z, "m3": m3,
                    "f2": f2, "f3": f3, "x_star": x_star
                }

    records = np.array(records)

    # 1) Example AP profiles
    plt.figure(figsize=(8, 5))
    plt.plot(profiles_example["x"], profiles_example["m1"], label="m1 (AP)")
    plt.plot(profiles_example["x"], profiles_example["m2"], label="m2 (AP)")
    plt.axvline(profiles_example["x_star"], linestyle="--", label="wavefront x*")
    plt.xlabel("position along AP axis (x)")
    plt.ylabel("concentration")
    plt.title(f"Example profiles (f2={profiles_example['f2']:.2f}, f3={profiles_example['f3']:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Heatmap of wavefront position over (f2, f3)
    F2 = np.unique(records[:, 0])
    F3 = np.unique(records[:, 1])
    Zheat = records[:, 5].reshape(len(F2), len(F3))
    plt.figure(figsize=(6, 5))
    plt.imshow(Zheat, origin="lower", aspect="auto",
               extent=[F3.min(), F3.max(), F2.min(), F2.max()])
    plt.colorbar(label="normalized wavefront position x*")
    plt.xlabel("blocking factor f3 (↓D3)")
    plt.ylabel("blocking factor f2 (↓D2)")
    plt.title("AP wavefront position across GJ-block morphospace")
    plt.tight_layout()
    plt.show()

    # 3) 3D morphospace of mean expressions
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig3 = plt.figure(figsize=(7, 6))
    ax = fig3.add_subplot(111, projection="3d")
    sc = ax.scatter(records[:, 2], records[:, 3], records[:, 4], c=records[:, 5])
    ax.set_xlabel("⟨m1⟩ (mean)")
    ax.set_ylabel("⟨m2⟩ (mean)")
    ax.set_zlabel("⟨m3⟩ (mean)")
    ax.set_title("Morphospace of mean expressions\n(color = AP wavefront position x*)")
    cb = fig3.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("x*")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()
