#!/usr/bin/env python3
"""
Generate side-by-side comparisons of our implementation vs. original Raup paper figures.

Creates comparison images showing:
- Our generated shell using extracted parameters
- Reference to original paper figure (placeholder for scanned image)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
import sys

# Import the raup_shell function
sys.path.insert(0, str(Path(__file__).parent.parent))
from create_raup_animations import raup_shell


def load_raup_data():
    """Load Raup model variables from JSON file."""
    script_dir = Path(__file__).parent
    data_file = script_dir / 'variables.json'
    
    if not data_file.exists():
        raise FileNotFoundError(f"Raup data file not found: {data_file}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    return data


def create_variable_comparison(variable_name, variable_data, output_dir):
    """
    Create comparison for a single variable showing its effect.
    
    Args:
        variable_name: Name of variable (W, D, T, S)
        variable_data: Variable data from JSON
        output_dir: Output directory for images
    """
    print(f"  Creating comparison for {variable_name}...")
    
    # Get typical range
    v_min = variable_data['typical_range']['min']
    v_max = variable_data['typical_range']['max']
    v_mid = (v_min + v_max) / 2
    
    # Default values for other parameters
    defaults = {'W': 2.5, 'D': 0.2, 'T': 0.5, 'S': 1.0}
    
    # Create figure with 3 subplots: low, mid, high values
    fig = plt.figure(figsize=(18, 6))
    
    values = [v_min, v_mid, v_max]
    titles = [f'{variable_name} = {v_min:.2f} (Low)', 
              f'{variable_name} = {v_mid:.2f} (Mid)',
              f'{variable_name} = {v_max:.2f} (High)']
    
    for idx, val in enumerate(values):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        # Set parameters
        params = defaults.copy()
        params[variable_name] = val
        
        # Generate shell
        X, Y, Z = raup_shell(W=params['W'], D=params['D'], 
                            T=params['T'], S=params['S'], 
                            turns=4, res=50)
        
        # Plot wireframe
        ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4, 
                         color='steelblue', linewidth=0.6, alpha=0.8)
        
        # Calculate appropriate limits based on shell size
        x_range = X.max() - X.min()
        y_range = Y.max() - Y.min()
        z_range = Z.max() - Z.min()
        max_range = max(x_range, y_range, z_range)
        
        # Set limits with padding (zoom out more)
        padding = max_range * 0.3
        ax.set_xlim([X.min() - padding, X.max() + padding])
        ax.set_ylim([Y.min() - padding, Y.max() + padding])
        ax.set_zlim([Z.min() - padding, Z.max() + padding])
        
        # Better viewing angle for shell recognition
        ax.view_init(elev=20, azim=45)
        ax.set_title(titles[idx], fontsize=11, fontweight='bold')
        ax.set_axis_off()
    
    fig.suptitle(f'Variable {variable_name}: {variable_data["name"]}\n'
                f'{variable_data["definition"]}\n'
                f'Effect: {variable_data["effect"]}',
                fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = output_dir / f'variable_{variable_name}_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {output_path.name}")
    return output_path


def create_empirical_shell_comparison(shell_data, output_dir, paper_figure_ref):
    """
    Create comparison for an empirical shell form.
    
    Args:
        shell_data: Shell data from JSON
        output_dir: Output directory
        paper_figure_ref: Reference to paper figure (e.g., "Figure 5")
    """
    shell_name = shell_data['name']
    print(f"  Creating comparison for {shell_name}...")
    
    fig = plt.figure(figsize=(16, 8))
    
    # Left: Our generated shell
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    X, Y, Z = raup_shell(W=shell_data['W'], D=shell_data['D'], 
                        T=shell_data['T'], S=shell_data.get('S', 1.0),
                        turns=shell_data['turns'], res=60)
    
    ax1.plot_wireframe(X, Y, Z, rstride=4, cstride=4,
                      color=shell_data['color'], linewidth=0.7, alpha=0.85)
    
    # Calculate appropriate limits based on actual shell dimensions
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    z_range = Z.max() - Z.min()
    max_range = max(x_range, y_range, z_range)
    
    # Set limits with generous padding (zoom out significantly)
    padding = max_range * 0.4
    center_x = (X.max() + X.min()) / 2
    center_y = (Y.max() + Y.min()) / 2
    
    ax1.set_xlim([center_x - max_range/2 - padding, center_x + max_range/2 + padding])
    ax1.set_ylim([center_y - max_range/2 - padding, center_y + max_range/2 + padding])
    ax1.set_zlim([Z.min() - padding, Z.max() + padding])
    
    # Better viewing angle - lower elevation shows shell shape better
    ax1.view_init(elev=15, azim=45)
    ax1.set_title(f'Our Implementation\n{shell_name} ({shell_data["common_name"]})\n'
                 f'W={shell_data["W"]}, D={shell_data["D"]:.2f}, T={shell_data["T"]:.2f}',
                 fontsize=12, fontweight='bold')
    ax1.set_axis_off()
    
    # Right: Paper figure (if available) or placeholder
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Try to load scanned paper image
    paper_img_path = output_dir / f'paper_{paper_figure_ref.lower().replace(" ", "_")}.png'
    
    if paper_img_path.exists():
        # Load and display scanned image
        import matplotlib.image as mpimg
        img = mpimg.imread(str(paper_img_path))
        ax2.imshow(img)
        ax2.set_title(f'Raup (1966) {paper_figure_ref}\n(Scanned from paper)', 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
    else:
        # Placeholder
        ax2.text(0.5, 0.5, 
                f'Original Paper\n{paper_figure_ref}\n\n'
                f'[Placeholder for scanned image\nfrom Raup (1966)]\n\n'
                f'To add: Scan {paper_figure_ref} from paper\n'
                f'and save as: {paper_img_path.name}',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_title(f'Raup (1966) {paper_figure_ref}', fontsize=12, fontweight='bold')
        ax2.axis('off')
    
    fig.suptitle(f'Empirical Shell Comparison: {shell_name}',
                fontsize=14, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    output_path = output_dir / f'empirical_{shell_name.lower().replace(" ", "_").replace("-", "_")}_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {output_path.name}")
    return output_path


def create_paper_figure_comparison(figure_num, description, params_list, output_dir):
    """
    Create comparison for a specific paper figure showing multiple parameter combinations.
    
    Args:
        figure_num: Figure number from paper (e.g., 5)
        description: Description of what the figure shows
        params_list: List of parameter dicts to generate
        output_dir: Output directory
    """
    print(f"  Creating comparison for Figure {figure_num}...")
    
    n_shells = len(params_list)
    cols = min(3, n_shells)
    rows = (n_shells + cols - 1) // cols
    
    fig = plt.figure(figsize=(6 * cols, 6 * rows))
    
    for idx, params in enumerate(params_list):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        
        X, Y, Z = raup_shell(W=params['W'], D=params['D'], 
                            T=params['T'], S=params.get('S', 1.0),
                            turns=params.get('turns', 4), res=50)
        
        ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4,
                         color='steelblue', linewidth=0.6, alpha=0.8)
        
        # Calculate appropriate limits based on shell size
        x_range = X.max() - X.min()
        y_range = Y.max() - Y.min()
        z_range = Z.max() - Z.min()
        max_range = max(x_range, y_range, z_range)
        
        # Set limits with padding (zoom out more)
        padding = max_range * 0.3
        center_x = (X.max() + X.min()) / 2
        center_y = (Y.max() + Y.min()) / 2
        
        ax.set_xlim([center_x - max_range/2 - padding, center_x + max_range/2 + padding])
        ax.set_ylim([center_y - max_range/2 - padding, center_y + max_range/2 + padding])
        ax.set_zlim([Z.min() - padding, Z.max() + padding])
        
        # Better viewing angle
        ax.view_init(elev=15, azim=45)
        ax.set_title(f"W={params['W']}, D={params['D']:.2f}, T={params['T']:.2f}",
                    fontsize=10, fontweight='bold')
        ax.set_axis_off()
    
    fig.suptitle(f'Our Implementation: Raup (1966) Figure {figure_num}\n{description}',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = output_dir / f'paper_figure_{figure_num}_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {output_path.name}")
    return output_path


def main():
    """Generate all comparison images."""
    print("="*60)
    print("Generating Paper Comparison Images")
    print("="*60)
    print()
    
    # Load data
    raup_data = load_raup_data()
    output_dir = Path(__file__).parent / 'comparisons'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate variable comparisons
    print("Generating variable comparisons...")
    core_vars = raup_data['core_variables']
    for var_name, var_data in core_vars.items():
        create_variable_comparison(var_name, var_data, output_dir)
    print()
    
    # Generate empirical shell comparisons
    print("Generating empirical shell comparisons...")
    figure_refs = ['Figure 5', 'Figure 6', 'Figure 7', 'Figure 8', 'Figure 9', 'Figure 10']
    for shell, fig_ref in zip(raup_data['empirical_shells'], figure_refs):
        create_empirical_shell_comparison(shell, output_dir, fig_ref)
    print()
    
    # Generate paper figure comparisons based on descriptions
    print("Generating paper figure reconstructions...")
    
    # Figure 1: Effect of W (whorl expansion)
    create_paper_figure_comparison(
        1, "Effect of Whorl Expansion Rate (W)",
        [
            {'W': 1.5, 'D': 0.2, 'T': 0.5},
            {'W': 2.5, 'D': 0.2, 'T': 0.5},
            {'W': 3.5, 'D': 0.2, 'T': 0.5},
        ],
        output_dir
    )
    
    # Figure 2: Effect of D (distance from axis)
    create_paper_figure_comparison(
        2, "Effect of Distance from Coiling Axis (D)",
        [
            {'W': 2.5, 'D': 0.05, 'T': 0.5},
            {'W': 2.5, 'D': 0.2, 'T': 0.5},
            {'W': 2.5, 'D': 0.4, 'T': 0.5},
        ],
        output_dir
    )
    
    # Figure 3: Effect of T (translation)
    create_paper_figure_comparison(
        3, "Effect of Translation Rate (T)",
        [
            {'W': 2.5, 'D': 0.2, 'T': 0.1},
            {'W': 2.5, 'D': 0.2, 'T': 0.5},
            {'W': 2.5, 'D': 0.2, 'T': 1.0},
        ],
        output_dir
    )
    
    print()
    print("="*60)
    print("✓ All comparison images generated!")
    print("="*60)
    print(f"\nGenerated {len(list(output_dir.glob('*.png')))} comparison images")
    print(f"\nNext steps:")
    print("  1. Scan original figures from Raup (1966) paper")
    print("  2. Save as: paper_figure_X.png in comparisons/ folder")
    print("  3. Update generate_paper_comparisons.py to load and display scanned images")
    print(f"\nComparison images saved to: {output_dir}")


if __name__ == '__main__':
    main()
