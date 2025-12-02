# Ca²⁺ Wave Vector Clustering – Output Data Format

This document specifies the output of the Ca²⁺ signal tracking and
vector‑clustering pipeline.

We follow the terminology of:

- Shannon et al. 2017 (Biophys J): wound‑induced **Ca²⁺ signal dynamics**
  around laser‑induced epithelial wounds (local Ca²⁺ entry events and
  propagating Ca²⁺ waves).
- Tung et al. 2024 (Nat Commun): inter‑embryo **injury waves** and
  Ca²⁺/ATP‑mediated signaling between **Xenopus laevis** embryos.

In our data model:

- Each tracked object corresponds to a **Ca²⁺ event**: a local maximum of
  GCaMP fluorescence (a Ca²⁺ *signal initiation site* / “flash”)
  that can move as part of an injury‑evoked **Ca²⁺ wavefront**.
- A *cluster* = one continuous trajectory of a Ca²⁺ event (one `track_id`).

The pipeline produces two tables:

1. `spark_tracks.csv` – per‑frame measurements of Ca²⁺ events  
2. `vector_clusters.csv` – per‑cluster summaries suitable for
   downstream clustering/analysis

Both tables are indexed relative to the **injury/poke time** so we can
relate Ca²⁺ wave dynamics to wounding and inter‑embryo injury‑wave
propagation.

---

## 1. `spark_tracks.csv` – Per‑frame Ca²⁺ event data

**Granularity:** one row per Ca²⁺ event per frame.

Each row describes a GCaMP‑bright region that has been segmented as a
single **Ca²⁺ signal initiation site** or part of an ongoing
**Ca²⁺ wavefront**.

### Columns

| Column      | Type   | Units            | Description |
|-------------|--------|------------------|-------------|
| `track_id`  | int    | –                | Unique identifier for a Ca²⁺ event trajectory. Each `track_id` corresponds to one Ca²⁺ signal initiation site tracked over time (one cluster). |
| `frame_idx` | int    | frames (0‑based) | Index of imaging frame in the time series. |
| `time_s`    | float  | seconds          | Time of this frame relative to the injury/poke: `time_s = (frame_idx − poke_frame_idx) / fps`. `time_s = 0` at wounding. |
| `x`         | float  | pixels           | X coordinate of the Ca²⁺ event centroid (GCaMP intensity maximum) in image coordinates. |
| `y`         | float  | pixels           | Y coordinate of the Ca²⁺ event centroid. |
| `vx`        | float  | pixels / second  | X component of the event’s instantaneous velocity between its previous and current detection (motion of the Ca²⁺ wavefront at that locus). Blank for the first frame of a track. |
| `vy`        | float  | pixels / second  | Y component of instantaneous velocity. |
| `speed`     | float  | pixels / second  | Magnitude of the propagation velocity at this event: `speed = sqrt(vx² + vy²)`. |
| `angle_deg` | float  | degrees          | Direction of Ca²⁺ wave propagation: `0° = right`, `90° = up` in image coordinates. Computed as `atan2(-vy, vx)` (in degrees). |
| `area`      | int    | pixels²          | Area of the segmented Ca²⁺ event (number of bright pixels). Proxy for the spatial extent / amplitude of the Ca²⁺ transient at this time point. |

**Optional columns**

If a poke site is defined or TIFF filenames are available:

| Column                | Type   | Units  | Description |
|-----------------------|--------|--------|-------------|
| `dist_from_poke_px`   | float  | pixels | Distance from the Ca²⁺ event centroid to the injury/poke site. Useful to analyze radial propagation of wound‑induced Ca²⁺ waves. |
| `filename`            | str    | –      | Source image filename (for TIFF stacks / folders). |

---

## 2. `vector_clusters.csv` – Per‑cluster Ca²⁺ wave summaries

**Granularity:** one row per Ca²⁺ event trajectory (one `track_id`).

Each row summarizes:

- **Input coverage** – how many frames/seconds the event spans
- **Ca²⁺ wave kinematics** – how far and how fast it propagates
- **Signal “volume”** – how much Ca²⁺ activity it carries over time

### Columns

| Column                  | Type   | Units              | Description |
|-------------------------|--------|--------------------|-------------|
| `cluster_id`            | int    | –                  | Cluster identifier (equal to `track_id` in `spark_tracks.csv`). |
| `n_frames`              | int    | frames             | Number of frames in which this Ca²⁺ event is detected. |
| `duration_frames`       | int    | frames             | `end_frame_idx − start_frame_idx + 1`. Frames covered between first and last detection (including any gaps). |
| `duration_s`            | float  | seconds            | `end_time_s − start_time_s`. Total time window over which this Ca²⁺ event contributes to the wound‑induced Ca²⁺ dynamics. |
| `start_frame_idx`       | int    | frames             | First frame where this event is detected. |
| `end_frame_idx`         | int    | frames             | Last frame where this event is detected. |
| `start_time_s`          | float  | seconds            | Time (relative to poke) at first detection. |
| `end_time_s`            | float  | seconds            | Time at last detection. |
| `start_x_px`            | float  | pixels             | Initial position of the Ca²⁺ event centroid. Often close to the primary wound / cavitation region. |
| `start_y_px`            | float  | pixels             | Initial Y position. |
| `end_x_px`              | float  | pixels             | Final centroid position, reflecting where the Ca²⁺ wavefront or discrete locus ends. |
| `end_y_px`              | float  | pixels             | Final Y position. |
| `net_displacement_px`   | float  | pixels             | Straight‑line distance from start to end: overall shift of the Ca²⁺ event as the wave propagates. |
| `path_length_px`        | float  | pixels             | Sum of stepwise distances between consecutive detections: total path travelled by the Ca²⁺ event within the epithelial sheet / embryo. |
| `net_speed_px_per_s`    | float  | pixels / second    | Net propagation speed of the event (`net_displacement_px / duration_s`). |
| `mean_speed_px_per_s`   | float  | pixels / second    | Average propagation speed along the trajectory (`path_length_px / duration_s`). |
| `peak_speed_px_per_s`   | float  | pixels / second    | Maximum instantaneous `speed` observed for this event. |
| `mean_angle_deg`        | float  | degrees            | Circular mean of `angle_deg` values: dominant direction of Ca²⁺ wave propagation for this event. |
| `angle_dispersion_deg`  | float  | degrees            | Circular spread of `angle_deg`. Low values = highly directional propagation, high values = more isotropic / noisy motion. |
| `mean_area_px2`         | float  | pixels²            | Mean spatial extent of the Ca²⁺ event over its lifetime (average area of the GCaMP‑bright region). |
| `total_area_px2_frames` | float  | pixels² · frames   | Sum of `area` across frames. Proxy for integrated Ca²⁺ signal “volume” contributed by this event (analogous to integrating fluorescence over space and time). |
| `dist_from_poke_start_px` | float | pixels (optional) | Distance from the poke site to the event’s initial position. Useful for analyzing radial distribution of Ca²⁺ signal initiation sites around the wound. |
| `dist_from_poke_end_px`   | float | pixels (optional) | Distance from the poke site to the final position of the event. |

---

## 3. What you can read out from each cluster

For each `cluster_id` in `vector_clusters.csv` you can extract:

- **Input window for that Ca²⁺ event**
  - `n_frames`, `duration_frames`, `duration_s`  
    → how much of the movie this event occupies.
- **Volume / load of Ca²⁺ signaling**
  - `total_area_px2_frames`, `mean_area_px2`  
    → integrated Ca²⁺ activity (analogous to calcium “load” or integrated GCaMP signal).
- **Wave propagation kinematics**
  - `net_displacement_px`, `path_length_px`,  
    `net_speed_px_per_s`, `mean_speed_px_per_s`, `peak_speed_px_per_s`  
    → how far and how fast the Ca²⁺ wavefront moves at this locus.
- **Directionality**
  - `mean_angle_deg`, `angle_dispersion_deg`  
    → whether Ca²⁺ waves propagate radially, anisotropically, or along preferred axes
      (e.g. away from the wound edge or between neighboring embryos).

These per‑cluster vectors can then be used for **vector clustering**—
for example, grouping Ca²⁺ events by propagation speed and direction
or comparing **injury waves within an embryo** to **inter‑embryo injury
wave transfers** as in the Xenopus experiments.
