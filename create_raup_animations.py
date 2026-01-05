#!/usr/bin/env python3
"""
Create animated visualizations of Raup's shell coiling model.

Generates:
1. Animated GIF showing shell morphogenesis through parameter space
2. Parameter space visualization (W, D, T)
3. Comparison with empirical shell forms (if data available)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import json
from pathlib import Path


def raup_shell(W=3, D=0.2, T=0.5, S=1, turns=6, res=100):
    """
    Generate a 3D shell using Raup's parametric model.
    
    Parameters:
        W: Whorl expansion rate (how much radius grows per revolution)
        D: Distance from coiling axis (0=tight, 1=open)
        T: Translation rate (vertical rise per radian)
        S: Shape of generating curve (1=circular, >1=wider, <1=taller)
        turns: Number of full rotations
        res: Resolution (points per turn)
    
    Returns:
        X, Y, Z: 3D coordinates of shell surface
    """
    theta = np.linspace(0, 2 * np.pi * turns, res * turns)
    phi = np.linspace(0, 2 * np.pi, res)
    theta, phi = np.meshgrid(theta, phi)
    
    b = np.log(W) / (2 * np.pi)
    r = np.exp(b * theta)
    offset = D / (1 - D) if D < 1 else 0
    
    xc = (1 + offset) * r * np.cos(theta)
    yc = (1 + offset) * r * np.sin(theta)
    zc = T * theta
    
    a = 0.5 * r
    b_val = a / S
    
    dx = a * np.cos(phi)
    dy = b_val * np.sin(phi)
    
    X = xc + dx * np.cos(theta) - dy * np.sin(theta)
    Y = yc + dx * np.sin(theta) + dy * np.cos(theta)
    Z = zc
    
    return X, Y, Z


def load_raup_data():
    """Load Raup model variables and empirical shell data from JSON file."""
    script_dir = Path(__file__).parent
    data_file = script_dir / 'datasets' / 'raup' / 'variables.json'
    
    if not data_file.exists():
        raise FileNotFoundError(f"Raup data file not found: {data_file}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    return data


def create_animated_gif(output_path='raup_shell_animation.gif', 
                        num_frames=240, fps=15, turns=5, res=60):
    """
    Create an animated GIF showing six realistic shell forms side by side.
    Each shell is shown from two angles: side view and overhead view.
    Based on Raup's original paper, showing how parameter variation produces
    different shell morphologies observed in nature.
    """
    print(f"Creating animated GIF: {output_path}")
    
    # Load shell forms from data file
    raup_data = load_raup_data()
    empirical_shells = raup_data['empirical_shells']
    
    # Convert to format expected by animation function
    shell_forms = []
    for shell in empirical_shells:
        shell_forms.append({
            'name': f"{shell['name']}\n({shell['common_name']})",
            'W': shell['W'],
            'D': shell['D'],
            'T': shell['T'],
            'S': shell.get('S', 1.0),
            'turns': shell['turns'],
            'color': shell['color']
        })
    
    print(f"  Loaded {len(shell_forms)} shell forms from datasets/raup/variables.json")
    
    # Create figure with 2x3 grid of 3D subplots (6 shells)
    fig = plt.figure(figsize=(18, 12))
    axes = []
    for i in range(6):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        axes.append(ax)
    
    # Two view modes: side angle and overhead
    # Alternate between them: first half of frames = side view, second half = overhead
    frames_per_view = num_frames // 2
    
    def animate(frame):
        """Update function for each frame."""
        # Determine which view mode (side or overhead)
        view_mode = 'side' if frame < frames_per_view else 'overhead'
        
        # Rotate view smoothly within each view mode
        frame_in_view = frame % frames_per_view
        azim = (frame_in_view / frames_per_view) * 360
        
        # Set elevation based on view mode
        if view_mode == 'side':
            elev = 32  # Side angle view
        else:
            elev = 90  # Overhead view (looking straight down)
        
        for idx, (ax, shell) in enumerate(zip(axes, shell_forms)):
            ax.clear()
            
            # Generate shell
            X, Y, Z = raup_shell(W=shell['W'], D=shell['D'], T=shell['T'], 
                                S=shell.get('S', 1.0), turns=shell['turns'], res=res)
            
            # Plot wireframe with shell-specific color (finer mesh)
            ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, 
                             color=shell['color'], linewidth=0.6, alpha=0.8)
            
            # Calculate dynamic limits based on actual shell size (same as comparison script)
            x_range = X.max() - X.min()
            y_range = Y.max() - Y.min()
            z_range = Z.max() - Z.min()
            max_range = max(x_range, y_range, z_range)
            
            # Set limits with generous padding (zoom out 50% more)
            padding = max_range * 0.5
            center_x = (X.max() + X.min()) / 2
            center_y = (Y.max() + Y.min()) / 2
            center_z = (Z.max() + Z.min()) / 2
            
            ax.set_xlim([center_x - max_range/2 - padding, center_x + max_range/2 + padding])
            ax.set_ylim([center_y - max_range/2 - padding, center_y + max_range/2 + padding])
            ax.set_zlim([center_z - max_range/2 - padding, center_z + max_range/2 + padding])
            
            # Apply view based on mode
            ax.view_init(elev=elev, azim=azim)
            
            # Set title with parameters and view mode
            view_label = "Side View" if view_mode == 'side' else "Overhead View"
            ax.set_title('{} [{}]\nW={:.2f}, D={:.2f}, T={:.2f}'.format(
                shell['name'], view_label, shell['W'], shell['D'], shell['T']),
                fontsize=10, fontweight='bold', pad=10)
            
            # Remove axis labels for cleaner look
            ax.set_axis_off()
        
        # Overall title with view mode indicator
        view_title = "Side Angle View" if view_mode == 'side' else "Overhead View"
        fig.suptitle(f'Raup Shell Coiling Model: Parameter Variation Produces Diverse Forms - {view_title}',
                    fontsize=14, fontweight='bold', y=0.98)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000/fps, blit=False)
    
    # Save as GIF
    print(f"  Saving {num_frames} frames (6 shells × 2 views: side + overhead) at {fps} fps...")
    anim.save(output_path, writer=PillowWriter(fps=fps))
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_parameter_space_visualization(output_path='raup_parameter_space.png'):
    """
    Create a visualization showing the three-dimensional parameter space.
    Shows how W, D, and T affect shell morphology.
    """
    print(f"Creating parameter space visualization: {output_path}")
    
    # Load variable ranges from data file
    raup_data = load_raup_data()
    W_range = raup_data['core_variables']['W']['typical_range']
    D_range = raup_data['core_variables']['D']['typical_range']
    T_range = raup_data['core_variables']['T']['typical_range']
    
    # Define parameter grid based on loaded ranges
    W_values = np.linspace(W_range['min'], W_range['max'], 6).tolist()
    D_values = np.linspace(D_range['min'], D_range['max'], 5).tolist()
    T_values = np.linspace(T_range['min'], T_range['max'], 6).tolist()
    
    print(f"  Using parameter ranges from datasets/raup/variables.json:")
    print(f"    W: {W_range['min']:.2f} - {W_range['max']:.2f}")
    print(f"    D: {D_range['min']:.2f} - {D_range['max']:.2f}")
    print(f"    T: {T_range['min']:.2f} - {T_range['max']:.2f}")
    
    # Create figure with subplots showing different slices
    fig = plt.figure(figsize=(16, 12))
    
    # Top row: Vary W and D (T fixed)
    T_fixed = 0.5
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    
    # Bottom row: Vary W and T (D fixed)
    D_fixed = 0.2
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    
    axes_top = [ax1, ax2, ax3]
    axes_bottom = [ax4, ax5, ax6]
    
    # Top row: Show effect of W and D
    for idx, (W, D) in enumerate([(1.5, 0.1), (2.5, 0.2), (3.5, 0.3)]):
        ax = axes_top[idx]
        X, Y, Z = raup_shell(W=W, D=D, T=T_fixed, turns=4, res=40)
        ax.plot_wireframe(X, Y, Z, rstride=6, cstride=6, 
                         color='steelblue', linewidth=0.5, alpha=0.8)
        ax.set_title(f'W={W}, D={D}, T={T_fixed}\n(Whorl expansion vs Distance)', 
                    fontsize=10)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_zlim([0, 8])
    
    # Bottom row: Show effect of W and T
    for idx, (W, T) in enumerate([(1.5, 0.2), (2.5, 0.5), (3.5, 1.0)]):
        ax = axes_bottom[idx]
        X, Y, Z = raup_shell(W=W, D=D_fixed, T=T, turns=4, res=40)
        ax.plot_wireframe(X, Y, Z, rstride=6, cstride=6, 
                         color='coral', linewidth=0.5, alpha=0.8)
        ax.set_title(f'W={W}, D={D_fixed}, T={T}\n(Whorl expansion vs Translation)', 
                    fontsize=10)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_zlim([0, 8])
    
    plt.suptitle("Raup Shell Coiling: Parameter Space Exploration\n"
                 "Three Core Variables: W (Whorl Expansion), D (Distance from Axis), T (Translation Rate)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_empirical_comparison(output_path='raup_empirical_comparison.png'):
    """
    Create a comparison showing theoretical Raup shells vs. empirical shell forms.
    Uses typical parameter values observed in nature.
    """
    print(f"Creating empirical comparison: {output_path}")
    
    # Load empirical shell data from JSON file
    raup_data = load_raup_data()
    empirical_shells_raw = raup_data['empirical_shells']
    
    # Convert to format expected by visualization function
    empirical_shells = []
    for shell in empirical_shells_raw:
        empirical_shells.append({
            'name': f"{shell['name']}\n({shell['common_name']}-like)",
            'W': shell['W'],
            'D': shell['D'],
            'T': shell['T'],
            'S': shell.get('S', 1.0),
            'turns': shell['turns'],
            'color': shell['color']
        })
    
    print(f"  Loaded {len(empirical_shells)} empirical shell forms from datasets/raup/variables.json")
    
    fig = plt.figure(figsize=(18, 10))
    
    for idx, shell in enumerate(empirical_shells):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        X, Y, Z = raup_shell(W=shell['W'], D=shell['D'], T=shell['T'], 
                            S=shell.get('S', 1.0), turns=shell.get('turns', 4), res=50)
        
        ax.plot_wireframe(X, Y, Z, rstride=6, cstride=6, 
                         color=shell['color'], linewidth=0.8, alpha=0.9)
        
        ax.set_title(shell['name'] + f'\nW={shell["W"]}, D={shell["D"]:.2f}, T={shell["T"]}', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        
        # Adjust limits based on shell type
        if shell['T'] > 0.7:  # High-spired
            ax.set_zlim([0, 12])
        else:
            ax.set_zlim([0, 6])
        
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
    
    plt.suptitle("Raup Model: Theoretical Shells vs. Empirical Forms\n"
                 "Parameter combinations matching observed mollusk shell morphologies",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("="*60)
    print("Raup Shell Coiling Model Visualizations")
    print("="*60)
    print()
    
    # Create animated GIF (all 6 shells side by side, showing side view then overhead view)
    create_animated_gif('raup_shell_animation.gif', num_frames=240, fps=15)
    print()
    
    # Create parameter space visualization
    create_parameter_space_visualization('raup_parameter_space.png')
    print()
    
    # Create empirical comparison
    create_empirical_comparison('raup_empirical_comparison.png')
    print()
    
    print("="*60)
    print("✓ All visualizations complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - raup_shell_animation.gif (animated morphing through parameter space)")
    print("  - raup_parameter_space.png (3D parameter space exploration)")
    print("  - raup_empirical_comparison.png (theoretical vs. empirical forms)")


if __name__ == '__main__':
    main()
