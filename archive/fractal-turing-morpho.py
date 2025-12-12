import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
from scipy.ndimage import zoom as scipy_zoom
from scipy.interpolate import RegularGridInterpolator
import time

class FractalTuringMorpho:
    """
    A fractal version of Turing morphogenesis that exhibits self-similarity across scales.
    Uses hierarchical pattern generation where coarse patterns seed finer patterns.
    """
    
    def __init__(self, base_size=256, max_levels=4):
        self.base_size = base_size
        self.max_levels = max_levels
        self.current_level = 0
        self.zoom_factor = 1.0
        self.center_x, self.center_y = 0.5, 0.5
        
        # Multi-scale parameters - each level has different characteristics
        self.scales = []
        for level in range(max_levels):
            scale_factor = 2 ** level
            self.scales.append({
                'size': base_size // scale_factor,
                'du': 0.2 / (1 + 0.1 * level),  # Slightly different diffusion at each scale
                'dv': 0.1 / (1 + 0.05 * level),
                'f': 0.055 + 0.005 * level,     # Varying feed rates create different patterns
                'k': 0.062 + 0.003 * level,
                'dt': 1.0 / (1 + 0.2 * level),  # Smaller time steps for finer scales
                'coupling': 0.1 * (1 - level / max_levels)  # Coupling strength decreases with level
            })
        
        # Initialize multi-scale grids
        self.U_levels = []
        self.V_levels = []
        self.initialize_grids()
        
        # Setup visualization
        self.setup_visualization()
        
    def initialize_grids(self):
        """Initialize reaction-diffusion grids at all scales with fractal seeding."""
        self.U_levels = []
        self.V_levels = []
        
        for level, scale in enumerate(self.scales):
            size = scale['size']
            
            # Initialize with uniform concentrations
            U = np.ones((size, size))
            V = np.zeros((size, size))
            
            # Add fractal seeding pattern
            self.add_fractal_seeds(U, V, level, size)
            
            # Add noise with scale-dependent amplitude
            noise_amp = 0.05 / (1 + 0.5 * level)
            U += noise_amp * np.random.random((size, size))
            V += noise_amp * np.random.random((size, size))
            
            self.U_levels.append(U)
            self.V_levels.append(V)
    
    def add_fractal_seeds(self, U, V, level, size):
        """Add self-similar seed patterns with true fractal structure."""
        # Create hierarchical seed pattern based on level
        if level == 0:
            # Base level: few large seeds
            num_seeds = 3
            seed_positions = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]
        else:
            # Higher levels: more seeds in fractal arrangement
            num_seeds = 2 ** (level + 1)
            seed_positions = []
            
            # Generate fractal seed positions using recursive subdivision
            for i in range(num_seeds):
                for j in range(num_seeds):
                    # Base position
                    base_x = (i + 0.5) / num_seeds
                    base_y = (j + 0.5) / num_seeds
                    
                    # Add fractal offset based on level
                    fractal_offset = 0.1 / (level + 1)
                    offset_x = fractal_offset * np.sin(2 * np.pi * i * j / num_seeds)
                    offset_y = fractal_offset * np.cos(2 * np.pi * i * j / num_seeds)
                    
                    seed_positions.append((base_x + offset_x, base_y + offset_y))
        
        # Place seeds with level-dependent characteristics
        for pos_x, pos_y in seed_positions:
            x = int(pos_x * size)
            y = int(pos_y * size)
            
            # Ensure seeds are within bounds
            x = np.clip(x, 5, size - 5)
            y = np.clip(y, 5, size - 5)
            
            # Seed size and strength vary with level
            seed_size = max(2, int(15 / (level + 1)))
            strength = 0.8 / (level * 0.3 + 1)
            
            # Create main seed
            U[x-seed_size:x+seed_size, y-seed_size:y+seed_size] *= (1 - strength)
            V[x-seed_size:x+seed_size, y-seed_size:y+seed_size] += strength * 0.4
            
            # Add fractal sub-structure
            if level > 0:
                sub_seeds = 4 if level < 3 else 2
                sub_size = max(1, seed_size // 3)
                sub_strength = strength * 0.6
                
                angles = np.linspace(0, 2*np.pi, sub_seeds, endpoint=False)
                radius = seed_size * 0.7
                
                for angle in angles:
                    sx = x + int(radius * np.cos(angle))
                    sy = y + int(radius * np.sin(angle))
                    
                    if sub_size <= sx < size - sub_size and sub_size <= sy < size - sub_size:
                        U[sx-sub_size:sx+sub_size, sy-sub_size:sy+sub_size] *= (1 - sub_strength)
                        V[sx-sub_size:sx+sub_size, sy-sub_size:sy+sub_size] += sub_strength * 0.3
    
    def laplacian(self, Z):
        """Compute Laplacian for diffusion with periodic boundary conditions."""
        return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
                np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z)
    
    def update_level(self, level):
        """Update reaction-diffusion at a specific scale level."""
        if level >= len(self.U_levels):
            return
            
        U = self.U_levels[level]
        V = self.V_levels[level]
        scale = self.scales[level]
        
        # Compute diffusion
        Lu = self.laplacian(U)
        Lv = self.laplacian(V)
        
        # Reaction terms
        Uvv = U * V * V
        
        # Cross-scale coupling - influence from coarser scales
        coupling_term_U = 0
        coupling_term_V = 0
        
        if level > 0 and scale['coupling'] > 0:
            # Upscale the coarser level to influence this level
            coarse_U = self.U_levels[level - 1]
            coarse_V = self.V_levels[level - 1]
            
            # Interpolate coarse pattern to current resolution
            upscaled_U = scipy_zoom(coarse_U, U.shape[0] / coarse_U.shape[0], order=1)
            upscaled_V = scipy_zoom(coarse_V, V.shape[0] / coarse_V.shape[0], order=1)
            
            # Coupling influences the reaction
            coupling_term_U = scale['coupling'] * (upscaled_U - U)
            coupling_term_V = scale['coupling'] * (upscaled_V - V)
        
        # Update equations with cross-scale coupling
        dU = scale['dt'] * (scale['du'] * Lu - Uvv + scale['f'] * (1 - U) + coupling_term_U)
        dV = scale['dt'] * (scale['dv'] * Lv + Uvv - (scale['f'] + scale['k']) * V + coupling_term_V)
        
        self.U_levels[level] += dU
        self.V_levels[level] += dV
        
        # Clamp values to prevent instability
        self.U_levels[level] = np.clip(self.U_levels[level], 0, 2)
        self.V_levels[level] = np.clip(self.V_levels[level], 0, 1)
    
    def get_zoomed_view(self):
        """Extract the current zoomed view with proper fractal detail generation."""
        # Use multiple levels and blend them for true fractal behavior
        base_level = min(int(np.log2(max(1, self.zoom_factor))), self.max_levels - 1)
        
        # Start with the base level pattern
        V_base = self.V_levels[base_level]
        size = V_base.shape[0]
        
        # Calculate view window
        view_size = max(10, int(size / self.zoom_factor))
        
        # Center coordinates in grid space
        cx = int(self.center_x * size)
        cy = int(self.center_y * size)
        
        # Calculate bounds with proper clamping
        half_view = view_size // 2
        x1 = max(0, cx - half_view)
        x2 = min(size, cx + half_view)
        y1 = max(0, cy - half_view)
        y2 = min(size, cy + half_view)
        
        # Adjust center if we hit boundaries
        if x2 - x1 < view_size:
            if x1 == 0:
                x2 = min(size, x1 + view_size)
            else:
                x1 = max(0, x2 - view_size)
        if y2 - y1 < view_size:
            if y1 == 0:
                y2 = min(size, y1 + view_size)
            else:
                y1 = max(0, y2 - view_size)
        
        # Extract base view
        view = V_base[x1:x2, y1:y2].copy()
        
        # Add detail from finer levels if zoomed in
        if self.zoom_factor > 1.5 and base_level < self.max_levels - 1:
            for detail_level in range(base_level + 1, self.max_levels):
                if detail_level < len(self.V_levels):
                    V_detail = self.V_levels[detail_level]
                    detail_size = V_detail.shape[0]
                    
                    # Map the view coordinates to the detail level
                    scale_ratio = detail_size / size
                    dx1 = int(x1 * scale_ratio)
                    dx2 = int(x2 * scale_ratio)
                    dy1 = int(y1 * scale_ratio)
                    dy2 = int(y2 * scale_ratio)
                    
                    # Ensure bounds are valid
                    dx1 = max(0, min(dx1, detail_size - 1))
                    dx2 = max(dx1 + 1, min(dx2, detail_size))
                    dy1 = max(0, min(dy1, detail_size - 1))
                    dy2 = max(dy1 + 1, min(dy2, detail_size))
                    
                    if dx2 > dx1 and dy2 > dy1:
                        detail_view = V_detail[dx1:dx2, dy1:dy2]
                        
                        # Resize detail to match current view
                        if detail_view.shape[0] > 0 and detail_view.shape[1] > 0:
                            detail_resized = scipy_zoom(detail_view, 
                                                      (view.shape[0] / detail_view.shape[0],
                                                       view.shape[1] / detail_view.shape[1]), 
                                                      order=1)
                            
                            # Blend detail with base (higher zoom = more detail influence)
                            detail_weight = min(0.6, (self.zoom_factor - 1.5) / 10)
                            view = (1 - detail_weight) * view + detail_weight * detail_resized
        
        # Upscale for high zoom levels to show crisp detail
        if self.zoom_factor > 4:
            upscale_factor = min(3, self.zoom_factor / 4)
            target_size = int(view.shape[0] * upscale_factor)
            view = scipy_zoom(view, target_size / view.shape[0], order=3)  # Cubic interpolation for smoothness
        
        return view
    
    def setup_visualization(self):
        """Setup the interactive matplotlib visualization."""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.25, right=0.85)
        
        # Initial display
        initial_view = self.get_zoomed_view()
        self.im = self.ax.imshow(initial_view, cmap='viridis', interpolation='bilinear')
        self.ax.set_title(f'Fractal Turing Morphogenesis - Level {self.current_level}, Zoom: {self.zoom_factor:.1f}x')
        
        # Add colorbar
        plt.colorbar(self.im, ax=self.ax, shrink=0.8)
        
        # Zoom controls
        ax_zoom = plt.axes([0.2, 0.1, 0.5, 0.03])
        self.zoom_slider = Slider(ax_zoom, 'Zoom', 1.0, 16.0, valinit=1.0, valfmt='%.1fx')
        self.zoom_slider.on_changed(self.update_zoom)
        
        # Center X control
        ax_cx = plt.axes([0.2, 0.05, 0.5, 0.03])
        self.cx_slider = Slider(ax_cx, 'Center X', 0.0, 1.0, valinit=0.5, valfmt='%.2f')
        self.cx_slider.on_changed(self.update_center)
        
        # Center Y control
        ax_cy = plt.axes([0.2, 0.01, 0.5, 0.03])
        self.cy_slider = Slider(ax_cy, 'Center Y', 0.0, 1.0, valinit=0.5, valfmt='%.2f')
        self.cy_slider.on_changed(self.update_center)
        
        # Reset button
        ax_reset = plt.axes([0.8, 0.1, 0.1, 0.04])
        self.reset_button = Button(ax_reset, 'Reset')
        self.reset_button.on_clicked(self.reset_view)
        
        # Regenerate button
        ax_regen = plt.axes([0.8, 0.05, 0.1, 0.04])
        self.regen_button = Button(ax_regen, 'Regenerate')
        self.regen_button.on_clicked(self.regenerate)
        
        # Mouse interaction for panning
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    
    def update_zoom(self, val):
        """Update zoom level."""
        self.zoom_factor = val
        self.update_display()
    
    def update_center(self, val):
        """Update center position."""
        self.center_x = self.cx_slider.val
        self.center_y = self.cy_slider.val
        self.update_display()
    
    def reset_view(self, event):
        """Reset to default view."""
        self.zoom_factor = 1.0
        self.center_x, self.center_y = 0.5, 0.5
        self.zoom_slider.set_val(1.0)
        self.cx_slider.set_val(0.5)
        self.cy_slider.set_val(0.5)
        self.update_display()
    
    def regenerate(self, event):
        """Regenerate the fractal patterns."""
        self.initialize_grids()
        self.update_display()
    
    def on_click(self, event):
        """Handle mouse clicks for panning with proper coordinate mapping."""
        if event.inaxes == self.ax and event.button == 1:  # Left click
            if event.xdata is not None and event.ydata is not None:
                # Get current view dimensions
                view = self.get_zoomed_view()
                view_height, view_width = view.shape
                
                # Convert click coordinates to relative position in view (0 to 1)
                rel_x = event.xdata / view_width
                rel_y = event.ydata / view_height
                
                # Calculate the current view extent in normalized coordinates
                view_extent = 1.0 / self.zoom_factor
                
                # Current view bounds in normalized space
                view_left = self.center_x - view_extent / 2
                view_top = self.center_y - view_extent / 2
                
                # Convert click to absolute normalized coordinates
                click_x = view_left + rel_x * view_extent
                click_y = view_top + rel_y * view_extent
                
                # Set new center to clicked position
                self.center_x = np.clip(click_x, view_extent/2, 1 - view_extent/2)
                self.center_y = np.clip(click_y, view_extent/2, 1 - view_extent/2)
                
                # Update sliders
                self.cx_slider.set_val(self.center_x)
                self.cy_slider.set_val(self.center_y)
                self.update_display()
    
    def update_display(self):
        """Update the visualization display."""
        view = self.get_zoomed_view()
        self.im.set_array(view)
        self.im.set_extent([0, view.shape[1], view.shape[0], 0])
        
        # Update title with current parameters
        level = min(int(np.log2(self.zoom_factor)), self.max_levels - 1)
        self.ax.set_title(f'Fractal Turing Morphogenesis - Level {level}, Zoom: {self.zoom_factor:.1f}x\n'
                         f'Center: ({self.center_x:.2f}, {self.center_y:.2f})')
        
        # Auto-scale colormap
        self.im.autoscale()
        self.fig.canvas.draw()
    
    def update_simulation(self, frame):
        """Animation update function."""
        # Update all levels
        for level in range(self.max_levels):
            self.update_level(level)
        
        # Update display
        self.update_display()
        return [self.im]
    
    def run_interactive(self):
        """Run the interactive fractal Turing morphogenesis."""
        # Create animation
        self.ani = FuncAnimation(self.fig, self.update_simulation, 
                                frames=1000, interval=100, blit=False, repeat=True)
        
        print("Interactive Fractal Turing Morphogenesis")
        print("Controls:")
        print("- Use zoom slider to zoom in/out")
        print("- Use center sliders to pan around")
        print("- Click on the image to center view at that point")
        print("- Reset button returns to default view")
        print("- Regenerate button creates new fractal patterns")
        print("\nFractal properties:")
        print("- Patterns show self-similarity across scales")
        print("- Zooming reveals finer detail with similar structures")
        print("- Each level influences the next through coupling")
        
        plt.show()
    
    def save_fractal_sequence(self, filename_base="fractal_turing", num_zooms=5):
        """Save a sequence of images showing fractal zoom."""
        zoom_levels = np.logspace(0, np.log10(8), num_zooms)
        
        for i, zoom in enumerate(zoom_levels):
            self.zoom_factor = zoom
            self.center_x, self.center_y = 0.3 + 0.2 * np.sin(i), 0.4 + 0.1 * np.cos(i)
            
            # Run simulation for a bit to develop patterns
            for _ in range(50):
                for level in range(self.max_levels):
                    self.update_level(level)
            
            # Save current view
            view = self.get_zoomed_view()
            plt.figure(figsize=(8, 8))
            plt.imshow(view, cmap='viridis', interpolation='bilinear')
            plt.title(f'Fractal Turing - Zoom {zoom:.1f}x')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{filename_base}_zoom_{i+1:02d}.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Saved {num_zooms} fractal zoom images")

# Example usage and demonstration
if __name__ == "__main__":
    # Create fractal Turing morphogenesis system
    fractal_turing = FractalTuringMorpho(base_size=256, max_levels=4)
    
    # Run interactive version
    fractal_turing.run_interactive()
    
    # Uncomment to save fractal sequence
    # fractal_turing.save_fractal_sequence()
