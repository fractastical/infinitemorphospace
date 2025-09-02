# planarian_outline.py
# Minimal, dependency-free (numpy + matplotlib) generator of a planarian-like 2D outline
# featuring: tapered tail, flared head “auricles,” mid-body bulge, and eye spots.

import numpy as np
import matplotlib.pyplot as plt

def planarian_outline(
    L=1.0,                 # half-length (x ranges from -L to +L)
    W=0.35,                # half-width scale (overall “thickness”)
    p=1.6,                 # overall body roundness (higher => squarer mid-body)
    head_taper=2.2,        # taper exponent toward the head (x > 0)
    tail_taper=3.2,        # taper exponent toward the tail (x < 0)
    belly_bulge=0.22,      # additional width at mid-body (Gaussian)
    belly_pos=0.00,        # center of belly bulge (x)
    belly_sigma=0.45,      # spread of belly bulge (relative to L)
    head_flare=0.32,       # extra width near head to mimic auricles
    head_flare_pos=0.72,   # where the head flare is centered (as fraction of +L)
    head_flare_sigma=0.18, # spread of the head flare (relative to L)
    samples=800            # resolution of the outline
):
    """Return closed outline points (X, Y) for a planarian-like shape."""
    x = np.linspace(-L, L, samples)

    # Asymmetric taper: head vs tail get different exponents
    alpha = np.where(x >= 0.0, head_taper, tail_taper)
    core = np.clip(1.0 - (np.abs(x) / L) ** alpha, 0.0, 1.0) ** p

    # Mid-body "belly" bulge (wider trunk)
    belly = belly_bulge * np.exp(-0.5 * ((x - belly_pos) / (belly_sigma * L)) ** 2)

    # Head flare to mimic auricles (wider anterior)
    hf_center = head_flare_pos * L
    headf = head_flare * np.exp(-0.5 * ((x - hf_center) / (head_flare_sigma * L)) ** 2)

    # Final half-width profile
    w = W * np.clip(core * (1.0 + belly + headf), 0.0, None)

    # Build closed outline: go forward along +w (top edge), return along -w (bottom edge)
    X = np.concatenate([x, x[::-1]])
    Y = np.concatenate([w, -w[::-1]])
    return X, Y, x, w

def default_params():
    """Parameter set tuned to look like a classic planarian (Schmidtea-like) silhouette."""
    return dict(
        L=1.0,
        W=0.34,
        p=1.7,
        head_taper=2.1,
        tail_taper=3.4,
        belly_bulge=0.24,
        belly_pos=-0.05,
        belly_sigma=0.48,
        head_flare=0.35,
        head_flare_pos=0.78,
        head_flare_sigma=0.20,
        samples=900,
    )

def plot_planarian(params=None, add_eyes=True, ax=None, facecolor="#7ac", edgecolor="k"):
    if params is None:
        params = default_params()
    X, Y, x, w = planarian_outline(**params)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    ax.fill(X, Y, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)

    # Eyes: small dark spots on the head margin, slightly lateral (dorsal view stylization)
    if add_eyes:
        # choose an x-position near the head where width is appreciable
        x_eye = params["L"] * 0.72
        # find width there to place eyes a fraction of the half-width
        idx = np.argmin(np.abs(x - x_eye))
        y_half = w[idx]
        eye_offset = 0.45 * y_half  # lateral placement
        eye_r = 0.03 * params["L"]
        for y_eye in (+eye_offset, -eye_offset):
            circle = plt.Circle((x_eye, y_eye), eye_r, color="black", zorder=5)
            ax.add_patch(circle)

    # aesthetics
    ax.set_aspect("equal", adjustable="box")
    pad = 0.12 * params["L"]
    ax.set_xlim(-params["L"] - pad, params["L"] + pad)
    ax.set_ylim(-params["W"] * 2.2, params["W"] * 2.2)
    ax.axis("off")
    return ax

if __name__ == "__main__":
    # --- Single static planarian ---
    ax = plot_planarian()
    plt.show()

    # --- Parameter sweep demo (optional): uncomment to preview variation ---
    # fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    # for i, flare in enumerate([0.15, 0.30, 0.45]):
    #     p = default_params()
    #     p["head_flare"] = flare
    #     plot_planarian(params=p, ax=axs[i])
    #     axs[i].set_title(f"Head flare = {flare:.2f}")
    # plt.show()

    # --- Save to SVG/PNG (optional) ---
    # ax = plot_planarian()
    # plt.savefig("planarian.svg", bbox_inches="tight", transparent=True)
    # plt.savefig("planarian.png", dpi=300, bbox_inches="tight")
