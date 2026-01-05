import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path

def raup_shell(W=3, D=0.2, T=0.5, S=1, turns=6, res=100):
    # theta: Array of angles for coiling along the spiral path (from 0 to full turns)
    # This defines the parametric position along the helicospiral.
    theta = np.linspace(0, 2 * np.pi * turns, res * turns)
    
    # phi: Angles around the generating curve (aperture), creating the cross-section at each point.
    # This meshes with theta to form a 2D grid for the full surface.
    phi = np.linspace(0, 2 * np.pi, res)
    theta, phi = np.meshgrid(theta, phi)
    
    # b: Expansion coefficient derived from W (whorl expansion rate).
    # W controls how much the radius grows per full revolution (2π radians).
    # Higher W means faster expansion, leading to wider shells.
    b = np.log(W) / (2 * np.pi)
    
    # r: Radius at each theta, following logarithmic spiral growth.
    # This ensures isometric (self-similar) expansion, central to Raup's model.
    r = np.exp(b * theta)
    
    # offset: Adjusts the position relative to the coiling axis based on D.
    # D (distance from coiling axis) determines how far the inner edge is from the axis.
    # D=0: Tight coiling with no umbilicus; higher D: More open or disc-like forms.
    offset = D / (1 - D) if D < 1 else 0
    
    # xc, yc: X and Y coordinates of the spiral path center.
    # Scaled by (1 + offset) to incorporate D, creating the offset from the axis.
    xc = (1 + offset) * r * np.cos(theta)
    yc = (1 + offset) * r * np.sin(theta)
    
    # zc: Z coordinate for translation along the axis.
    # T (translation rate) controls vertical rise per radian; relates to turret height.
    # T=0: Planispiral (flat); higher T: Elongated, helical forms.
    zc = T * theta
    
    # a: Semi-major axis of the generating curve (ellipse), scaled by r for growth.
    # This makes the aperture expand proportionally with the spiral.
    a = 0.5 * r
    
    # b_val: Semi-minor axis, adjusted by S (shape of generating curve).
    # S=1: Circular aperture; S>1: Wider than tall; S<1: Taller than wide.
    b_val = a / S
    
    # dx, dy: Offsets for the generating curve in local coordinates.
    # These define the elliptical cross-section around the path center.
    dx = a * np.cos(phi)
    dy = b_val * np.sin(phi)
    
    # X, Y: Final coordinates by rotating and adding the generating curve to the path.
    # Rotation aligns the curve perpendicular to the tangent (approximation here).
    # This sweeps the aperture along the helicospiral to form the shell surface.
    X = xc + dx * np.cos(theta) - dy * np.sin(theta)
    Y = yc + dx * np.sin(theta) + dy * np.cos(theta)
    
    # Z: Simply the translated zc, as the generating curve is in the XY plane.
    Z = zc
    
    return X, Y, Z

# Load parameter ranges from raup_dataset/variables.json
def load_raup_parameters():
    """Load Raup model parameters from data file."""
    script_dir = Path(__file__).parent
    data_file = script_dir / 'datasets' / 'raup' / 'variables.json'
    
    if data_file.exists():
        with open(data_file, 'r') as f:
            data = json.load(f)
        return data
    else:
        # Fallback to default values if file not found
        print(f"Warning: {data_file} not found, using default parameters")
        return None

# Load parameters from data file
raup_data = load_raup_parameters()

if raup_data:
    # Use parameter ranges from data file
    W_range = raup_data['core_variables']['W']['typical_range']
    D_range = raup_data['core_variables']['D']['typical_range']
    T_range = raup_data['core_variables']['T']['typical_range']
    
    # Grid parameters: Vary W and T to sample a slice of the morphospace.
    # Ws: Range of whorl expansion rates (rows in the plot grid).
    # Ts: Range of translation rates (columns in the plot grid).
    # D_fixed and S_fixed: Held constant for this 2D slice; vary them for fuller exploration.
    Ws = [W_range['min'], (W_range['min'] + W_range['max']) / 2, W_range['max']]
    Ts = [T_range['min'], (T_range['min'] + T_range['max']) / 2, T_range['max']]
    D_fixed = (D_range['min'] + D_range['max']) / 2
    S_fixed = 1.0
    
    print(f"Loaded parameters from datasets/raup/variables.json")
    print(f"  W range: {W_range['min']:.2f} - {W_range['max']:.2f}")
    print(f"  D range: {D_range['min']:.2f} - {D_range['max']:.2f}")
    print(f"  T range: {T_range['min']:.2f} - {T_range['max']:.2f}")
else:
    # Fallback defaults
    Ws = [1.5, 2.5, 4.0]
    Ts = [0.1, 0.5, 1.0]
    D_fixed = 0.1
    S_fixed = 1

# Plot grid: Create subplots to visualize multiple parameter combinations.
# This illustrates how changing W and T affects shell morphology,
# mimicking Raup's approach to mapping theoretical vs. observed forms.
fig, axs = plt.subplots(3, 3, subplot_kw={'projection': '3d'}, figsize=(12, 12))
for i, W in enumerate(Ws):
    for j, T in enumerate(Ts):
        X, Y, Z = raup_shell(W=W, D=D_fixed, T=T, S=S_fixed)
        axs[i, j].plot_wireframe(X, Y, Z, rstride=5, cstride=5)
        axs[i, j].set_title(f'W={W}, T={T}')
        axs[i, j].set_axis_off()  # Cleaner view
plt.tight_layout()
plt.show()