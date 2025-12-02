#!/usr/bin/env python3
"""
Generate simulated Ca²⁺ wave data for testing hypotheses.

This script creates synthetic spark tracking data that mimics real experimental
conditions with configurable:
- Number of embryos
- Poke locations
- Wave propagation patterns
- Timing and dynamics
- Spatial distributions

Output format matches spark_tracks.csv for direct comparison.
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import math


@dataclass
class Embryo:
    """Configuration for a simulated embryo."""
    id: str
    center_x: float
    center_y: float
    length: float  # Length of embryo (pixels)
    width: float   # Width of embryo (pixels)
    angle: float   # Orientation angle (degrees, 0 = horizontal)
    head_x: float  # Head position
    head_y: float
    tail_x: float  # Tail position
    tail_y: float


@dataclass
class PokeConfig:
    """Configuration for a poke event."""
    x: float
    y: float
    embryo_id: Optional[str] = None  # Which embryo was poked
    time: float = 0.0  # Time of poke (relative to experiment start)


@dataclass
class WaveConfig:
    """Configuration for wave propagation."""
    speed_px_per_s: float = 5.0  # Propagation speed
    duration_s: float = 10.0  # How long waves persist
    decay_rate: float = 0.1  # Activity decay over time
    radial: bool = True  # Radial vs directional propagation
    direction_deg: Optional[float] = None  # Direction if not radial


class SimulationGenerator:
    """Generate simulated Ca²⁺ wave data."""
    
    def __init__(self, fps=1.0, image_width=2048, image_height=2048):
        self.fps = fps
        self.width = image_width
        self.height = image_height
        self.embryos: List[Embryo] = []
        self.pokes: List[PokeConfig] = []
        self.wave_config = WaveConfig()
        
    def add_embryo(self, embryo: Embryo):
        """Add an embryo to the simulation."""
        self.embryos.append(embryo)
        
    def add_poke(self, poke: PokeConfig):
        """Add a poke event."""
        self.pokes.append(poke)
        
    def set_wave_config(self, config: WaveConfig):
        """Set wave propagation parameters."""
        self.wave_config = config
        
    def _point_in_embryo(self, x: float, y: float, embryo: Embryo) -> bool:
        """Check if a point is within an embryo boundary."""
        # Transform to embryo-relative coordinates
        dx = x - embryo.center_x
        dy = y - embryo.center_y
        
        # Rotate to align with embryo axis
        angle_rad = math.radians(embryo.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a
        
        # Check if within ellipse (approximating embryo shape)
        a = embryo.length / 2
        b = embryo.width / 2
        return (rx / a) ** 2 + (ry / b) ** 2 <= 1.0
        
    def _calculate_ap_position(self, x: float, y: float, embryo: Embryo) -> float:
        """Calculate AP position (0=head, 1=tail) for a point in an embryo."""
        # Vector from head to point
        dx = x - embryo.head_x
        dy = y - embryo.head_y
        
        # Vector along embryo axis
        axis_dx = embryo.tail_x - embryo.head_x
        axis_dy = embryo.tail_y - embryo.head_y
        axis_len = math.hypot(axis_dx, axis_dy)
        
        if axis_len < 1e-6:
            return 0.5  # Default to middle
        
        # Project point onto axis
        proj = (dx * axis_dx + dy * axis_dy) / axis_len
        ap_norm = max(0.0, min(1.0, proj / axis_len))
        
        return ap_norm
        
    def _generate_wave_from_poke(self, poke: PokeConfig, max_time_s: float, 
                                 track_id_start: int) -> pd.DataFrame:
        """Generate spark tracks from a single poke event."""
        tracks = []
        track_id = track_id_start
        
        # Determine which embryos are affected
        affected_embryos = []
        poke_embryo = None
        
        if poke.embryo_id:
            # Find the poked embryo
            for emb in self.embryos:
                if emb.id == poke.embryo_id:
                    poke_embryo = emb
                    affected_embryos.append(emb)
                    break
        
        # Check if poke affects other embryos (inter-embryo signaling)
        for emb in self.embryos:
            if emb == poke_embryo:
                continue
            # Check distance between poke and embryo center
            dist = math.hypot(poke.x - emb.center_x, poke.y - emb.center_y)
            # If within reasonable distance, trigger wave
            if dist < emb.length * 2:
                affected_embryos.append(emb)
        
        # Generate waves for each affected embryo
        for embryo in affected_embryos:
            # Initial spark at poke location (if in embryo) or near embryo surface
            if embryo == poke_embryo and self._point_in_embryo(poke.x, poke.y, embryo):
                start_x, start_y = poke.x, poke.y
            else:
                # Start near embryo center or at closest point to poke
                start_x = embryo.center_x
                start_y = embryo.center_y
            
            # Generate multiple spark clusters propagating from start
            n_clusters = np.random.poisson(3) + 1  # 1-6 clusters typically
            
            for cluster_idx in range(n_clusters):
                cluster_tracks = self._generate_cluster_track(
                    start_x, start_y, embryo, poke.time, max_time_s, track_id
                )
                tracks.extend(cluster_tracks)
                track_id += 1
        
        return pd.DataFrame(tracks)
    
    def _generate_cluster_track(self, start_x: float, start_y: float, 
                                embryo: Embryo, poke_time: float, 
                                max_time_s: float, track_id: int) -> List[dict]:
        """Generate a single cluster track propagating from a starting point."""
        track = []
        current_x = start_x
        current_y = start_y
        
        # Determine propagation direction
        if self.wave_config.radial:
            # Radial: random direction from poke point
            angle = np.random.uniform(0, 2 * math.pi)
        else:
            # Directional: use specified direction or random
            if self.wave_config.direction_deg is not None:
                angle = math.radians(self.wave_config.direction_deg)
            else:
                angle = np.random.uniform(0, 2 * math.pi)
        
        speed = np.random.normal(self.wave_config.speed_px_per_s, 
                                self.wave_config.speed_px_per_s * 0.2)
        speed = max(0.5, speed)  # Minimum speed
        
        # Duration of this track
        duration = np.random.exponential(self.wave_config.duration_s * 0.5)
        duration = min(duration, max_time_s - poke_time)
        
        dt = 1.0 / self.fps
        time = poke_time
        frame_idx = int(poke_time * self.fps)
        
        prev_x, prev_y = current_x, current_y
        
        while time < poke_time + duration and time < max_time_s:
            # Calculate position with some randomness
            dx = math.cos(angle) * speed * dt
            dy = math.sin(angle) * speed * dt
            
            # Add some directional variation (not perfectly straight)
            angle_variation = np.random.normal(0, 0.1)
            angle += angle_variation
            
            current_x += dx
            current_y += dy
            
            # Bounce back if outside embryo (simplified boundary)
            if not self._point_in_embryo(current_x, current_y, embryo):
                # Reflect direction
                angle += math.pi / 2
                # Move back inside
                current_x = prev_x
                current_y = prev_y
                continue
            
            # Calculate speed for this frame
            vx = (current_x - prev_x) / dt if dt > 0 else 0
            vy = (current_y - prev_y) / dt if dt > 0 else 0
            speed_frame = math.hypot(vx, vy)
            angle_deg = math.degrees(math.atan2(-vy, vx))
            
            # Activity decays over time
            time_since_poke = time - poke_time
            activity_factor = math.exp(-self.wave_config.decay_rate * time_since_poke)
            area = max(10, int(100 * activity_factor * np.random.uniform(0.5, 1.5)))
            
            # Calculate AP position
            ap_norm = self._calculate_ap_position(current_x, current_y, embryo)
            
            # Distance from poke
            dist_from_poke = math.hypot(current_x - self.pokes[0].x, 
                                       current_y - self.pokes[0].y)
            
            track.append({
                'track_id': track_id,
                'frame_idx': frame_idx,
                'time_s': time - poke_time,  # Relative to poke
                'x': current_x,
                'y': current_y,
                'vx': vx,
                'vy': vy,
                'speed': speed_frame,
                'angle_deg': angle_deg,
                'area': area,
                'embryo_id': embryo.id,
                'ap_norm': ap_norm,
                'dv_px': 0.0,  # Could calculate if needed
                'dist_from_poke_px': dist_from_poke,
                'filename': f'simulated_{embryo.id}.tif'
            })
            
            prev_x, prev_y = current_x, current_y
            time += dt
            frame_idx += 1
        
        return track
    
    def generate(self, duration_s: float = 30.0, output_path: str = "simulated_spark_tracks.csv") -> pd.DataFrame:
        """Generate complete simulated dataset."""
        all_tracks = []
        track_id = 0
        
        # Sort pokes by time
        sorted_pokes = sorted(self.pokes, key=lambda p: p.time)
        
        for poke in sorted_pokes:
            poke_tracks = self._generate_wave_from_poke(poke, duration_s, track_id)
            if len(poke_tracks) > 0:
                all_tracks.append(poke_tracks)
                track_id = poke_tracks['track_id'].max() + 1
        
        if len(all_tracks) == 0:
            raise ValueError("No tracks generated. Check poke locations and embryo configurations.")
        
        df = pd.concat(all_tracks, ignore_index=True)
        df = df.sort_values(['frame_idx', 'track_id'])
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"✓ Generated {len(df)} track states from {track_id} unique tracks")
        print(f"✓ Saved to {output_path}")
        
        return df


def create_experiment_scenarios():
    """Create predefined experimental scenarios for testing."""
    scenarios = {}
    
    # Scenario 1: Two embryos, head-head orientation, poke in embryo A
    scenarios['two_embryos_head_head'] = {
        'embryos': [
            Embryo(id='A', center_x=500, center_y=1024, length=400, width=150, 
                  angle=0, head_x=300, head_y=1024, tail_x=700, tail_y=1024),
            Embryo(id='B', center_x=1500, center_y=1024, length=400, width=150,
                  angle=0, head_x=1300, head_y=1024, tail_x=1700, tail_y=1024)
        ],
        'pokes': [PokeConfig(x=500, y=1024, embryo_id='A', time=0.0)],
        'wave_config': WaveConfig(speed_px_per_s=5.0, duration_s=10.0, radial=True)
    }
    
    # Scenario 2: Three embryos in triangle
    scenarios['three_embryos_triangle'] = {
        'embryos': [
            Embryo(id='A', center_x=600, center_y=800, length=300, width=120,
                  angle=0, head_x=450, head_y=800, tail_x=750, tail_y=800),
            Embryo(id='B', center_x=1500, center_y=600, length=300, width=120,
                  angle=45, head_x=1350, head_y=550, tail_x=1650, tail_y=650),
            Embryo(id='C', center_x=1500, center_y=1400, length=300, width=120,
                  angle=-45, head_x=1350, head_y=1450, tail_x=1650, tail_y=1350)
        ],
        'pokes': [PokeConfig(x=600, y=800, embryo_id='A', time=0.0)],
        'wave_config': WaveConfig(speed_px_per_s=6.0, duration_s=12.0, radial=True)
    }
    
    # Scenario 3: Two embryos, tail-tail orientation
    scenarios['two_embryos_tail_tail'] = {
        'embryos': [
            Embryo(id='A', center_x=500, center_y=1024, length=400, width=150,
                  angle=0, head_x=300, head_y=1024, tail_x=700, tail_y=1024),
            Embryo(id='B', center_x=1500, center_y=1024, length=400, width=150,
                  angle=180, head_x=1700, head_y=1024, tail_x=1300, tail_y=1024)
        ],
        'pokes': [PokeConfig(x=700, y=1024, embryo_id='A', time=0.0)],  # Poke at tail
        'wave_config': WaveConfig(speed_px_per_s=5.0, duration_s=10.0, radial=True)
    }
    
    # Scenario 4: Multiple poke locations in same embryo
    scenarios['multiple_pokes_same_embryo'] = {
        'embryos': [
            Embryo(id='A', center_x=1024, center_y=1024, length=500, width=200,
                  angle=0, head_x=774, head_y=1024, tail_x=1274, tail_y=1024)
        ],
        'pokes': [
            PokeConfig(x=900, y=1024, embryo_id='A', time=0.0),    # Anterior
            PokeConfig(x=1100, y=1024, embryo_id='A', time=15.0),  # Posterior, later
        ],
        'wave_config': WaveConfig(speed_px_per_s=4.0, duration_s=8.0, radial=True)
    }
    
    # Scenario 5: Three embryos in a line
    scenarios['three_embryos_line'] = {
        'embryos': [
            Embryo(id='A', center_x=400, center_y=1024, length=350, width=140,
                  angle=0, head_x=225, head_y=1024, tail_x=575, tail_y=1024),
            Embryo(id='B', center_x=1024, center_y=1024, length=350, width=140,
                  angle=0, head_x=849, head_y=1024, tail_x=1199, tail_y=1024),
            Embryo(id='C', center_x=1648, center_y=1024, length=350, width=140,
                  angle=0, head_x=1473, head_y=1024, tail_x=1823, tail_y=1024)
        ],
        'pokes': [PokeConfig(x=400, y=1024, embryo_id='A', time=0.0)],
        'wave_config': WaveConfig(speed_px_per_s=5.5, duration_s=12.0, radial=True)
    }
    
    # Scenario 6: Four embryos in a square formation
    scenarios['four_embryos_square'] = {
        'embryos': [
            Embryo(id='A', center_x=600, center_y=600, length=300, width=120,
                  angle=0, head_x=450, head_y=600, tail_x=750, tail_y=600),
            Embryo(id='B', center_x=1444, center_y=600, length=300, width=120,
                  angle=0, head_x=1294, head_y=600, tail_x=1594, tail_y=600),
            Embryo(id='C', center_x=600, center_y=1448, length=300, width=120,
                  angle=0, head_x=450, head_y=1448, tail_x=750, tail_y=1448),
            Embryo(id='D', center_x=1444, center_y=1448, length=300, width=120,
                  angle=0, head_x=1294, head_y=1448, tail_x=1594, tail_y=1448)
        ],
        'pokes': [PokeConfig(x=600, y=600, embryo_id='A', time=0.0)],
        'wave_config': WaveConfig(speed_px_per_s=6.0, duration_s=15.0, radial=True)
    }
    
    # Scenario 7: Four embryos in a line
    scenarios['four_embryos_line'] = {
        'embryos': [
            Embryo(id='A', center_x=320, center_y=1024, length=300, width=120,
                  angle=0, head_x=170, head_y=1024, tail_x=470, tail_y=1024),
            Embryo(id='B', center_x=682, center_y=1024, length=300, width=120,
                  angle=0, head_x=532, head_y=1024, tail_x=832, tail_y=1024),
            Embryo(id='C', center_x=1044, center_y=1024, length=300, width=120,
                  angle=0, head_x=894, head_y=1024, tail_x=1194, tail_y=1024),
            Embryo(id='D', center_x=1406, center_y=1024, length=300, width=120,
                  angle=0, head_x=1256, head_y=1024, tail_x=1556, tail_y=1024)
        ],
        'pokes': [PokeConfig(x=320, y=1024, embryo_id='A', time=0.0)],
        'wave_config': WaveConfig(speed_px_per_s=5.5, duration_s=15.0, radial=True)
    }
    
    # Scenario 8: Three embryos, central one poked
    scenarios['three_embryos_central_poke'] = {
        'embryos': [
            Embryo(id='A', center_x=500, center_y=1024, length=350, width=140,
                  angle=0, head_x=325, head_y=1024, tail_x=675, tail_y=1024),
            Embryo(id='B', center_x=1024, center_y=1024, length=350, width=140,
                  angle=0, head_x=849, head_y=1024, tail_x=1199, tail_y=1024),
            Embryo(id='C', center_x=1548, center_y=1024, length=350, width=140,
                  angle=0, head_x=1373, head_y=1024, tail_x=1723, tail_y=1024)
        ],
        'pokes': [PokeConfig(x=1024, y=1024, embryo_id='B', time=0.0)],  # Poke middle embryo
        'wave_config': WaveConfig(speed_px_per_s=5.5, duration_s=12.0, radial=True)
    }
    
    return scenarios


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated Ca²⁺ wave data for hypothesis testing"
    )
    parser.add_argument(
        '--scenario',
        choices=['two_embryos_head_head', 'three_embryos_triangle', 
                'two_embryos_tail_tail', 'multiple_pokes_same_embryo',
                'three_embryos_line', 'four_embryos_square', 'four_embryos_line',
                'three_embryos_central_poke', 'custom'],
        default='two_embryos_head_head',
        help='Predefined experimental scenario'
    )
    parser.add_argument(
        '--output',
        default='simulated_spark_tracks.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=30.0,
        help='Simulation duration in seconds (default: 30.0)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=1.0,
        help='Frames per second (default: 1.0)'
    )
    parser.add_argument(
        '--config',
        help='JSON config file (from calibrate_from_real_data.py or custom config)'
    )
    
    args = parser.parse_args()
    
    # Create generator
    gen = SimulationGenerator(fps=args.fps)
    
    # Check if config file is provided (takes precedence over scenario)
    scenario_name = args.scenario  # Default scenario name
    
    if args.config and Path(args.config).exists():
        # Load config from JSON file (from real data calibration or custom)
        print(f"Loading configuration from {args.config}...")
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Load embryos
        for emb_dict in config.get('embryos', []):
            embryo = Embryo(**emb_dict)
            gen.add_embryo(embryo)
        
        # Load pokes
        for poke_dict in config.get('pokes', []):
            poke = PokeConfig(**poke_dict)
            gen.add_poke(poke)
        
        # Load wave config
        if 'wave_config' in config:
            wave_dict = config['wave_config']
            wave_config = WaveConfig(**wave_dict)
            gen.set_wave_config(wave_config)
        
        print(f"  → Loaded {len(gen.embryos)} embryo(s), {len(gen.pokes)} poke(s)")
        
        if 'source_data' in config:
            print(f"  → Configuration derived from real data")
            source = config['source_data']
            if 'tracks_csv' in source:
                print(f"     Source tracks: {Path(source['tracks_csv']).name}")
        
        scenario_name = "calibrated from real data"
    elif args.config:
        print(f"Error: Config file '{args.config}' not found.")
        return
    else:
        # Use predefined scenario
        scenarios = create_experiment_scenarios()
        if args.scenario not in scenarios:
            print(f"Error: Unknown scenario '{args.scenario}'")
            print(f"Available scenarios: {list(scenarios.keys())}")
            return
        scenario = scenarios[args.scenario]
        
        # Set up embryos
        for embryo in scenario['embryos']:
            gen.add_embryo(embryo)
        
        # Set up pokes
        for poke in scenario['pokes']:
            gen.add_poke(poke)
        
        # Set wave config
        gen.set_wave_config(scenario['wave_config'])
    
    # Generate simulation
    print(f"Generating simulation: {scenario_name}")
    print(f"  Embryos: {len(gen.embryos)}")
    print(f"  Pokes: {len(gen.pokes)}")
    print(f"  Duration: {args.duration}s at {args.fps} fps")
    print()
    
    df = gen.generate(duration_s=args.duration, output_path=args.output)
    
    # Save metadata
    metadata_path = args.output.replace('.csv', '_metadata.json')
    metadata = {
        'scenario': args.scenario,
        'duration_s': args.duration,
        'fps': args.fps,
        'n_embryos': len(gen.embryos),
        'n_pokes': len(gen.pokes),
        'embryos': [asdict(emb) for emb in gen.embryos],
        'pokes': [asdict(poke) for poke in gen.pokes],
        'wave_config': asdict(gen.wave_config),
        'n_tracks': df['track_id'].nunique(),
        'n_track_states': len(df)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to {metadata_path}")
    
    # Generate summary statistics
    print("\n=== Simulation Summary ===")
    print(f"Total track states: {len(df)}")
    print(f"Unique tracks: {df['track_id'].nunique()}")
    print(f"Time range: {df['time_s'].min():.2f} to {df['time_s'].max():.2f} seconds")
    print(f"Embryos represented: {df['embryo_id'].dropna().unique()}")
    print(f"Files: {df['filename'].nunique()}")


if __name__ == "__main__":
    main()

