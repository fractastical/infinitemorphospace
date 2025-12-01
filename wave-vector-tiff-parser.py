import cv2
import numpy as np
import math
import csv
import os
import re
import tifffile as tiff


def natural_key(s):
    """Sort helper that handles numbers in filenames naturally."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


class SparkTracker:
    """
    Detect and track bright 'spark' clusters frame-by-frame in a sequence of TIFF images.

    Extra features:
      * Automatically segment embryos from a reference frame (first TIFF).
      * Estimate head–tail axis per embryo via PCA.
      * Label each event with embryo_id (A/B), ap_norm (0=head, 1=tail), dv_px.
      * Try to detect the poke site from a pink arrow overlay; otherwise use
        user-provided --poke-x/--poke-y to set poke_xy.
      * Compute distance from each event to the poke site.
    """

    def __init__(
        self,
        min_area=3,
        max_area=1500,
        max_link_distance=20,
        max_gap_frames=1,
        # thresholds tuned for bright yellow-green sparks on dark blue background
        white_v_min=230,
        white_s_max=40,
        green_h_min=30,
        green_h_max=80,
        green_s_min=80,
        green_v_min=190,
        vector_arrow_scale=0.05,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.max_link_distance = max_link_distance
        self.max_gap_frames = max_gap_frames

        self.white_v_min = white_v_min
        self.white_s_max = white_s_max
        self.green_h_min = green_h_min
        self.green_h_max = green_h_max
        self.green_s_min = green_s_min
        self.green_v_min = green_v_min

        self.vector_arrow_scale = vector_arrow_scale

        self.tracks = {}
        self.next_track_id = 0
        self.fps = None

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # geometry-related attributes
        self.height = None
        self.width = None
        self.embryos = {}              # label -> dict(head, tail, centroid, mask)
        self.embryo_labels = []        # list of labels in index order
        self.embryo_label_map = None   # int image: -1 = none, 0..N-1 = embryo index
        self.ap_norm_map = None        # float image, AP position per pixel (0=head,1=tail)
        self.dv_map = None             # float image, distance perpendicular to AP
        self.embryo_union_mask = None  # bool image, union of all embryos
        self.poke_xy = None            # (x,y) of poke location if known/detected

    # ---------- TIFF reading helper ----------
    
    def _get_tiff_page_count(self, path):
        """
        Get the number of pages in a TIFF file without loading image data.
        Returns (num_pages, use_tifffile) tuple.
        """
        try:
            with tiff.TiffFile(path) as tif:
                return len(tif.pages), True
        except Exception:
            # If tifffile fails, assume single page (OpenCV will handle it)
            return 1, False
    
    def _read_tiff_page(self, path, page_idx=0, use_tifffile=True):
        """
        Read a single page from a TIFF file. For memory efficiency.
        Returns (numpy array, is_bgr) tuple.
        """
        if use_tifffile:
            try:
                with tiff.TiffFile(path) as tif:
                    img = tif.asarray(key=page_idx)
                    if isinstance(img, np.ndarray):
                        img = img.copy()
                    else:
                        img = np.array(img)
                    return img, False  # tifffile reads as RGB
            except Exception as e:
                raise RuntimeError(f"Could not read TIFF page {page_idx} from {path}: {str(e)}")
        else:
            # Fallback to OpenCV (single page only, page_idx must be 0)
            if page_idx > 0:
                raise ValueError(f"OpenCV can only read first page, but requested page {page_idx}")
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None or img.size == 0:
                raise RuntimeError(f"OpenCV could not read TIFF file: {path}")
            return img, True  # OpenCV reads as BGR

    # ---------- frame prep / spark detection ----------

    def _prepare_frame(self, raw, is_bgr=False):
        """
        Convert a raw TIFF array (2D or 3D, possibly 16-bit) to 8-bit BGR.
        
        Args:
            raw: numpy array from TIFF file
            is_bgr: if True, assumes input is already BGR (from OpenCV). 
                    if False, assumes RGB (from tifffile) and converts.
        """
        if raw.dtype == np.uint16:
            # simple 16->8 bit compression
            raw8 = (raw >> 8).astype(np.uint8)
        else:
            raw8 = raw.astype(np.uint8)

        if raw8.ndim == 2:
            # grayscale -> BGR
            bgr = cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR)
        elif raw8.ndim == 3:
            # assume (H, W, C). Drop alpha channel if present.
            if raw8.shape[2] == 4:
                raw8 = raw8[..., :3]
            if is_bgr:
                # Already in BGR format (from OpenCV)
                bgr = raw8
            else:
                # TIFFs from tifffile are usually RGB; OpenCV expects BGR
                bgr = cv2.cvtColor(raw8, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported TIFF shape: {raw.shape}")
        return bgr

    def _detect_sparks(self, frame):
        """
        Detect bright white / green 'sparks' in a single BGR frame.
        Returns (clusters, mask).
        """
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # White mask
        lower_white = np.array([0, 0, self.white_v_min], dtype=np.uint8)
        upper_white = np.array([179, self.white_s_max, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Green mask (yellow-green)
        lower_green = np.array([self.green_h_min,
                                self.green_s_min,
                                self.green_v_min], dtype=np.uint8)
        upper_green = np.array([self.green_h_max, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Combine and clean up
        mask = cv2.bitwise_or(white_mask, green_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        clusters = []
        for label in range(1, num_labels):  # skip background
            area = stats[label, cv2.CC_STAT_AREA]
            if area < self.min_area or area > self.max_area:
                continue
            cx, cy = centroids[label]
            clusters.append({"x": float(cx), "y": float(cy), "area": int(area)})
        return clusters, mask

    # ---------- embryo geometry & poke detection ----------

    def _init_geometry_and_poke(self, frame_bgr, user_poke_xy=None):
        """
        From a single BGR frame:

          * Segment up to 2 embryos.
          * Estimate head–tail axis (PCA of contour).
          * Precompute embryo_label_map, ap_norm_map, dv_map.
          * If user_poke_xy is None, try to detect poke location from a pink arrow.
        """
        h, w = frame_bgr.shape[:2]
        self.height = h
        self.width = w

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu threshold
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ensure embryos are white on black
        if th.mean() > 127:
            th = 255 - th

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 0.01 * h * w
        emb_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not emb_contours:
            print("\n=== EMBRYO DETECTION SUMMARY ===")
            print("✗ WARNING: No embryo-sized contours found in reference frame")
            print("  → Geometry-based annotations (embryo_id, ap_norm, dv_px) will be empty")
            print("=" * 35 + "\n")
            return

        # Sort by centroid x position: leftmost = A, rightmost = B
        def contour_cx(c):
            M = cv2.moments(c)
            if M["m00"] == 0:
                return 0
            return M["m10"] / M["m00"]

        emb_contours.sort(key=contour_cx)

        self.embryos = {}
        self.embryo_labels = []
        embryo_masks = []

        labels = ["A", "B"]
        for idx, contour in enumerate(emb_contours[:2]):
            label = labels[idx] if idx < len(labels) else f"E{idx}"
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

            pts = contour.reshape(-1, 2).astype(np.float32)
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
            v = eigenvectors[0]  # principal axis
            v_norm = v / (np.linalg.norm(v) + 1e-9)

            proj = np.dot(pts - mean.reshape(1, 2), v_norm.reshape(2, 1)).ravel()
            min_idx = np.argmin(proj)
            max_idx = np.argmax(proj)
            end1 = pts[min_idx]
            end2 = pts[max_idx]

            # Head = leftmost endpoint
            head = end1
            tail = end2
            if head[0] > tail[0]:
                head, tail = tail, head

            M = cv2.moments(contour)
            if M["m00"] == 0:
                cx, cy = float(head[0]), float(head[1])
            else:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

            self.embryos[label] = {
                "id": label,
                "mask": (mask == 255),
                "head": (float(head[0]), float(head[1])),
                "tail": (float(tail[0]), float(tail[1])),
                "centroid": (float(cx), float(cy)),
            }
            self.embryo_labels.append(label)
            embryo_masks.append(mask == 255)

        # Precompute label / AP / DV maps
        self.embryo_label_map = np.full((h, w), -1, dtype=np.int8)
        self.ap_norm_map = np.full((h, w), np.nan, dtype=np.float32)
        self.dv_map = np.full((h, w), np.nan, dtype=np.float32)

        self.embryo_union_mask = np.zeros((h, w), dtype=bool)

        for idx, label in enumerate(self.embryo_labels):
            emb = self.embryos[label]
            mask = emb["mask"]
            self.embryo_label_map[mask] = idx
            self.embryo_union_mask |= mask

            head = np.array(emb["head"], dtype=np.float32)
            tail = np.array(emb["tail"], dtype=np.float32)
            axis = tail - head
            L = np.linalg.norm(axis)
            if L < 1e-6:
                u = np.array([1.0, 0.0], dtype=np.float32)
            else:
                u = axis / L
            u_perp = np.array([-u[1], u[0]], dtype=np.float32)

            ys, xs = np.where(mask)
            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            delta = pts - head.reshape(1, 2)
            ap = (delta @ u.reshape(2, 1)).ravel() / (L + 1e-9)
            dv = (delta @ u_perp.reshape(2, 1)).ravel()
            self.ap_norm_map[ys, xs] = ap
            self.dv_map[ys, xs] = dv

        # Print embryo geometry summary
        if self.embryo_labels:
            print("\n=== EMBRYO DETECTION SUMMARY ===")
            print(f"✓ Found {len(self.embryo_labels)} embryo(s) with directional axes")
            print("\nEmbryo geometry:")
            for label in self.embryo_labels:
                emb = self.embryos[label]
                head = emb['head']
                tail = emb['tail']
                axis_length = math.hypot(tail[0] - head[0], tail[1] - head[1])
                print(
                    f"  Embryo {label}: head=({head[0]:.1f}, {head[1]:.1f}), "
                    f"tail=({tail[0]:.1f}, {tail[1]:.1f}), "
                    f"axis_length={axis_length:.1f}px, "
                    f"centroid=({emb['centroid'][0]:.1f}, {emb['centroid'][1]:.1f})"
                )
        else:
            print("\n=== EMBRYO DETECTION SUMMARY ===")
            print("✗ WARNING: No embryos detected - geometry-based annotations will be empty")

        # Poke detection: use user coords if provided, else try to auto-detect
        print("\n=== POKE DETECTION SUMMARY ===")
        if user_poke_xy is not None:
            self.poke_xy = user_poke_xy
            print(f"✓ Using user-provided poke location: ({user_poke_xy[0]:.1f}, {user_poke_xy[1]:.1f})")
        else:
            auto_poke = self._detect_poke_from_pink_arrow(frame_bgr)
            self.poke_xy = auto_poke
            if auto_poke is not None:
                print(f"✓ Auto-detected poke location: ({auto_poke[0]:.1f}, {auto_poke[1]:.1f})")
            else:
                print("✗ WARNING: Could not auto-detect pink arrow")
                print("  → dist_from_poke_px will be empty unless you provide --poke-x/--poke-y")
        print("=" * 35 + "\n")

    def _detect_poke_from_pink_arrow(self, frame_bgr):
        """
        Try to detect the poke site by looking for a pink arrow/text and
        choosing the pink pixel closest to the embryo surface.

        Returns (x,y) or None if no suitable pink pixels are found.
        """
        if self.embryo_union_mask is None:
            return None

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Pink / magenta range in HSV. You may tweak these thresholds.
        lower_pink = np.array([140, 50, 80], dtype=np.uint8)
        upper_pink = np.array([179, 255, 255], dtype=np.uint8)
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)

        # Clean up pink mask a bit
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        ys, xs = np.where(pink_mask > 0)
        if len(xs) == 0:
            return None

        # Distance transform from embryo union: smaller = closer to embryo surface
        emb_uint8 = np.where(self.embryo_union_mask, 0, 255).astype(np.uint8)
        distmap = cv2.distanceTransform(emb_uint8, cv2.DIST_L2, 3)

        best_xy = None
        best_dist = None
        for y, x in zip(ys, xs):
            d = distmap[y, x]
            if best_dist is None or d < best_dist:
                best_dist = d
                best_xy = (int(x), int(y))

        return best_xy

    # ---------- tracking & annotation ----------

    def _annotate_state(self, state):
        """
        Attach embryo_id, ap_norm, dv_px, and dist_from_poke_px to a track state.
        """
        x = state["x"]
        y = state["y"]

        embryo_id = ""
        ap_norm = ""
        dv_px = ""

        if (self.embryo_label_map is not None and
                self.ap_norm_map is not None and
                self.dv_map is not None and
                self.width is not None and
                self.height is not None):

            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < self.width and 0 <= iy < self.height:
                idx = int(self.embryo_label_map[iy, ix])
                if 0 <= idx < len(self.embryo_labels):
                    embryo_id = self.embryo_labels[idx]
                    ap_norm_val = float(self.ap_norm_map[iy, ix])
                    dv_val = float(self.dv_map[iy, ix])
                    ap_norm = ap_norm_val
                    dv_px = dv_val

        state["embryo_id"] = embryo_id
        state["ap_norm"] = ap_norm if ap_norm != "" else ""
        state["dv_px"] = dv_px if dv_px != "" else ""

        if self.poke_xy is not None:
            px, py = self.poke_xy
            dx = x - px
            dy = y - py
            state["dist_from_poke_px"] = float(math.hypot(dx, dy))
        else:
            state["dist_from_poke_px"] = ""

    def _update_tracks(self, frame_idx, time_s, clusters):
        """
        Associate clusters with existing tracks and create new ones.
        """
        frame_states = []

        candidate_track_ids = [
            tid for tid, tr in self.tracks.items()
            if frame_idx - tr["last_frame"] <= self.max_gap_frames
        ]
        unmatched_clusters = set(range(len(clusters)))
        unmatched_tracks = set(candidate_track_ids)

        # greedy nearest-neighbour association
        while unmatched_clusters and unmatched_tracks:
            best_pair = None
            best_dist = None
            for tid in unmatched_tracks:
                tr = self.tracks[tid]
                x0, y0 = tr["last_pos"]
                for ci in unmatched_clusters:
                    c = clusters[ci]
                    d = math.hypot(c["x"] - x0, c["y"] - y0)
                    if d > self.max_link_distance:
                        continue
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        best_pair = (tid, ci)
            if best_pair is None:
                break

            tid, ci = best_pair
            unmatched_tracks.remove(tid)
            unmatched_clusters.remove(ci)

            tr = self.tracks[tid]
            prev_x, prev_y = tr["last_pos"]
            prev_frame = tr["last_frame"]
            x = clusters[ci]["x"]
            y = clusters[ci]["y"]
            area = clusters[ci]["area"]

            dt_frames = frame_idx - prev_frame
            dt_s = dt_frames / self.fps if self.fps and dt_frames > 0 else None

            if dt_s and dt_s > 0:
                vx = (x - prev_x) / dt_s
                vy = (y - prev_y) / dt_s
                speed = math.hypot(vx, vy)
                angle_deg = math.degrees(math.atan2(-vy, vx))  # 0° = right, 90° = up
            else:
                vx = vy = speed = angle_deg = None

            state = {
                "track_id": tid,
                "frame_idx": frame_idx,
                "time_s": time_s,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "speed": speed,
                "angle_deg": angle_deg,
                "area": area,
            }
            # annotate with embryo + poke geometry
            self._annotate_state(state)

            tr["last_pos"] = (x, y)
            tr["last_frame"] = frame_idx
            tr["history"].append(state)
            frame_states.append(state)

        # new tracks for unmatched clusters
        for ci in unmatched_clusters:
            c = clusters[ci]
            tid = self.next_track_id
            self.next_track_id += 1
            state = {
                "track_id": tid,
                "frame_idx": frame_idx,
                "time_s": time_s,
                "x": float(c["x"]),
                "y": float(c["y"]),
                "vx": None,
                "vy": None,
                "speed": None,
                "angle_deg": None,
                "area": int(c["area"]),
            }
            self._annotate_state(state)

            self.tracks[tid] = {
                "last_pos": (c["x"], c["y"]),
                "last_frame": frame_idx,
                "history": [state],
            }
            frame_states.append(state)

        return frame_states

    # ---------- overlays ----------

    def _draw_overlays(self, frame, frame_states):
        """
        Draw detected clusters, motion vectors, and geometry markers on the frame.
        """
        # draw embryos head/tail axis and labels
        for label, emb in self.embryos.items():
            hx, hy = int(round(emb["head"][0])), int(round(emb["head"][1]))
            tx, ty = int(round(emb["tail"][0])), int(round(emb["tail"][1]))
            cx, cy = int(round(emb["centroid"][0])), int(round(emb["centroid"][1]))
            cv2.line(frame, (hx, hy), (tx, ty), (255, 255, 0), 1)
            cv2.circle(frame, (hx, hy), 4, (255, 0, 0), 1)   # head = blue
            cv2.circle(frame, (tx, ty), 4, (0, 0, 255), 1)   # tail = red
            cv2.putText(
                frame,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # draw poke marker
        if self.poke_xy is not None:
            cv2.drawMarker(
                frame,
                (int(self.poke_xy[0]), int(self.poke_xy[1])),
                (255, 192, 203),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=15,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        # draw per-cluster markers
        for s in frame_states:
            x = int(round(s["x"]))
            y = int(round(s["y"]))
            track_id = s["track_id"]

            cv2.circle(frame, (x, y), 3, (0, 255, 255), 1)
            cv2.putText(
                frame,
                str(track_id),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if s["vx"] is not None and s["vy"] is not None:
                dx = int(round(s["vx"] * self.vector_arrow_scale))
                dy = int(round(s["vy"] * self.vector_arrow_scale))
                end = (x + dx, y + dy)
                cv2.arrowedLine(frame, (x, y), end, (0, 0, 255), 1, tipLength=0.3)

    # ---------- main TIFF-folder processing ----------

    def process_tiff_folder(
        self,
        folder_path,
        poke_frame_idx,
        fps,
        out_video_path=None,
        csv_path="spark_tracks.csv",
        poke_xy=None,
    ):
        """
        Process a folder of TIFF files as a time-lapse sequence.
        
        Recursively searches folder_path and all subdirectories for TIFF files.

        Assumes each TIFF is a single frame; frames are ordered by filename
        (natural sort) and spaced by 1/fps seconds.

        Outputs a CSV with columns:

          track_id, frame_idx, time_s, x, y, vx, vy, speed, angle_deg, area,
          embryo_id, ap_norm, dv_px, dist_from_poke_px, filename
        
        The filename column contains the relative path from folder_path, preserving
        subdirectory structure (e.g., "2/long name.tif").
        """
        self.tracks = {}
        self.next_track_id = 0
        self.fps = float(fps)

        print(f"\n{'='*60}")
        print(f"Processing TIFF sequence from: {folder_path}")
        print(f"{'='*60}\n")
        
        # Collect TIFF files recursively from folder and all subfolders
        paths = []
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith((".tif", ".tiff")):
                    paths.append(os.path.join(root, f))
        
        if not paths:
            raise RuntimeError(f"No TIFF files found in {folder_path} or its subdirectories")
        
        # Sort by full path (natural sort on filename portion)
        paths.sort(key=lambda p: natural_key(os.path.basename(p)))
        
        print(f"Found {len(paths)} TIFF file(s) (including subdirectories)")
        print(f"Frame rate: {fps} fps")
        print(f"Poke frame index: {poke_frame_idx} (t = 0)\n")

        # Get frame size from first image and initialise geometry + poke
        print("Analyzing reference frame (first image) for geometry...")
        first_path = paths[0]
        print(f"Reading first frame: {os.path.relpath(first_path, folder_path)}")
        
        # Check file size first
        file_size = os.path.getsize(first_path)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        print("Attempting to read TIFF file (using OpenCV for stability)...")
        
        try:
            # Get page count first
            num_pages, use_tifffile = self._get_tiff_page_count(first_path)
            if num_pages > 1:
                print(f"✓ Detected multi-page TIFF file with {num_pages} pages")
            else:
                print(f"✓ Detected single-page TIFF file")
            
            # Read only the first page for geometry initialization
            first_raw, is_bgr = self._read_tiff_page(first_path, page_idx=0, use_tifffile=use_tifffile)
            print(f"Raw TIFF shape: {first_raw.shape}, dtype: {first_raw.dtype}")
            
            first_frame = self._prepare_frame(first_raw, is_bgr=is_bgr)
            height, width = first_frame.shape[:2]
            print(f"Frame dimensions: {width}x{height} pixels\n")
            del first_raw  # Free memory
        except Exception as e:
            print(f"\n✗ ERROR: Failed to read first TIFF file")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise
        
        self._init_geometry_and_poke(first_frame, user_poke_xy=poke_xy)

        # Video writer for overlay
        writer = None
        if out_video_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        # CSV setup
        csv_dir = os.path.dirname(csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)

        csv_file = open(csv_path, mode="w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "track_id",
                "frame_idx",
                "time_s",
                "x",
                "y",
                "vx",
                "vy",
                "speed",
                "angle_deg",
                "area",
                "embryo_id",
                "ap_norm",
                "dv_px",
                "dist_from_poke_px",
                "filename",
            ],
        )
        csv_writer.writeheader()

        print("Processing frames and tracking sparks...")
        print("(Processing frames incrementally to conserve memory...)\n")
        
        global_frame_idx = 0
        total_frames = 0
        
        # Process files one at a time, reading pages one at a time
        try:
            for file_idx, path in enumerate(paths):
                rel_path = os.path.relpath(path, folder_path)
                
                try:
                    # Get page count without loading image data
                    num_pages, use_tifffile = self._get_tiff_page_count(path)
                    
                    if num_pages > 1:
                        print(f"Processing {rel_path} ({num_pages} pages)...")
                    
                    # Process each page one at a time
                    for page_idx in range(num_pages):
                        frame_idx = global_frame_idx
                        global_frame_idx += 1
                        total_frames += 1
                        
                        # Show progress for large files
                        if num_pages > 1 and (page_idx == 0 or (page_idx + 1) % 50 == 0 or (page_idx + 1) == num_pages):
                            print(f"  Page {page_idx + 1}/{num_pages} (frame {frame_idx + 1})")
                        
                        try:
                            # Read only this one page
                            raw, is_bgr = self._read_tiff_page(path, page_idx=page_idx, use_tifffile=use_tifffile)
                            
                            frame = self._prepare_frame(raw, is_bgr=is_bgr)
                            # Free raw memory immediately
                            del raw
                            
                        except Exception as e:
                            print(f"\n⚠ WARNING: Failed to read/prepare frame {frame_idx} from {rel_path} page {page_idx + 1}: {str(e)}")
                            print("Skipping this frame...")
                            continue

                        time_s = (frame_idx - poke_frame_idx) / fps
                        clusters, _ = self._detect_sparks(frame)
                        frame_states = self._update_tracks(frame_idx, time_s, clusters)

                        # write CSV rows
                        for s in frame_states:
                            row = dict(s)
                            for key in ["vx", "vy", "speed", "angle_deg", "ap_norm", "dv_px"]:
                                if row.get(key) is None or row.get(key) != row.get(key):  # NaN check
                                    row[key] = ""
                            # Store relative path with page info if multi-page
                            if num_pages > 1:
                                row["filename"] = f"{rel_path} (page {page_idx+1})"
                            else:
                                row["filename"] = rel_path
                            csv_writer.writerow(row)

                        # overlay video
                        if writer is not None:
                            overlay = frame.copy()
                            self._draw_overlays(overlay, frame_states)
                            writer.write(overlay)
                        
                        # Free frame memory
                        del frame
                    
                except Exception as e:
                    print(f"\n⚠ WARNING: Failed to process file {file_idx} ({rel_path}): {str(e)}")
                    print("Skipping this file...")
                    continue
        finally:
            if writer is not None:
                writer.release()
            csv_file.close()

        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"\nSummary:")
        print(f"  • TIFF files processed: {len(paths)}")
        print(f"  • Total frames processed: {total_frames}")
        print(f"  • Tracks found: {len(self.tracks)}")
        print(f"  • Embryos detected: {len(self.embryo_labels)} ({', '.join(self.embryo_labels) if self.embryo_labels else 'none'})")
        print(f"  • Poke location: {'✓ detected' if self.poke_xy is not None else '✗ not found'}")
        print(f"\nOutput files:")
        print(f"  • CSV: {csv_path}")
        if out_video_path is not None:
            print(f"  • Video: {out_video_path}")
        print(f"{'='*60}\n")


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Track bright Ca²⁺ 'sparks' in a folder of TIFF frames, "
                    "annotating embryo head/tail and poke location."
    )
    parser.add_argument("folder", help="Folder containing TIFF frames.")
    parser.add_argument(
        "poke_frame",
        type=int,
        help="Index (0-based) of frame where poke occurs (t = 0).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second (1 / time between TIFFs).",
    )
    parser.add_argument(
        "--out-video",
        dest="out_video",
        default=None,
        help="Optional MP4 overlay output path.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default="spark_tracks.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--poke-x",
        type=float,
        default=None,
        help="Poke X coordinate in pixels (override auto-detection).",
    )
    parser.add_argument(
        "--poke-y",
        type=float,
        default=None,
        help="Poke Y coordinate in pixels (override auto-detection).",
    )

    args = parser.parse_args()

    if (args.poke_x is None) != (args.poke_y is None):
        parser.error("You must provide both --poke-x and --poke-y, or neither.")

    poke_xy = None
    if args.poke_x is not None and args.poke_y is not None:
        poke_xy = (args.poke_x, args.poke_y)

    tracker = SparkTracker()
    tracker.process_tiff_folder(
        folder_path=args.folder,
        poke_frame_idx=args.poke_frame,
        fps=args.fps,
        out_video_path=args.out_video,
        csv_path=args.csv_path,
        poke_xy=poke_xy,
    )
