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

    def _prepare_frame(self, raw):
        """
        Convert a raw TIFF array (2D or 3D, possibly 16-bit) to 8-bit BGR.
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
            # TIFFs are usually RGB; OpenCV expects BGR
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

            tr["last_pos"] = (x, y)
            tr["last_frame"] = frame_idx
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
            self.tracks[tid] = {
                "last_pos": (c["x"], c["y"]),
                "last_frame": frame_idx,
                "history": [state],
            }
            frame_states.append(state)

        return frame_states

    def _draw_overlays(self, frame, frame_states):
        """
        Draw detected clusters and their motion vectors.
        """
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

        Assumes each TIFF is a single frame; frames are ordered by filename
        (natural sort) and spaced by 1/fps seconds.
        """
        self.tracks = {}
        self.next_track_id = 0
        self.fps = float(fps)

        # Collect TIFF files
        filenames = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".tif", ".tiff"))
        ]
        if not filenames:
            raise RuntimeError(f"No TIFF files found in {folder_path}")
        filenames.sort(key=natural_key)
        paths = [os.path.join(folder_path, f) for f in filenames]

        # Get frame size from first image
        first_raw = tiff.imread(paths[0])
        first_frame = self._prepare_frame(first_raw)
        height, width = first_frame.shape[:2]

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
                "dist_from_poke_px",
                "filename",
            ],
        )
        csv_writer.writeheader()

        try:
            for frame_idx, path in enumerate(paths):
                raw = tiff.imread(path)
                frame = self._prepare_frame(raw)

                time_s = (frame_idx - poke_frame_idx) / fps
                clusters, _ = self._detect_sparks(frame)
                frame_states = self._update_tracks(frame_idx, time_s, clusters)

                # distance from poke, if known
                if poke_xy is not None:
                    px, py = poke_xy
                    for s in frame_states:
                        dx = s["x"] - px
                        dy = s["y"] - py
                        s["dist_from_poke_px"] = float((dx ** 2 + dy ** 2) ** 0.5)
                else:
                    for s in frame_states:
                        s["dist_from_poke_px"] = ""

                # write CSV rows
                for s in frame_states:
                    row = dict(s)
                    for key in ["vx", "vy", "speed", "angle_deg"]:
                        if row[key] is None:
                            row[key] = ""
                    row["filename"] = os.path.basename(path)
                    csv_writer.writerow(row)

                # overlay video
                if writer is not None:
                    overlay = frame.copy()
                    self._draw_overlays(overlay, frame_states)
                    if poke_xy is not None:
                        cv2.drawMarker(
                            overlay,
                            (int(poke_xy[0]), int(poke_xy[1])),
                            (255, 192, 203),
                            markerType=cv2.MARKER_TILTED_CROSS,
                            markerSize=15,
                            thickness=2,
                            line_type=cv2.LINE_AA,
                        )
                    writer.write(overlay)
        finally:
            if writer is not None:
                writer.release()
            csv_file.close()

        print(f"Finished. Frames processed: {len(paths)}")
        print(f"Tracks found: {len(self.tracks)}")
        print(f"CSV written to: {csv_path}")
        if out_video_path is not None:
            print(f"Overlay video written to: {out_video_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Track bright 'sparks' in a folder of TIFF frames."
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
        help="Poke X coordinate in pixels.",
    )
    parser.add_argument(
        "--poke-y",
        type=float,
        default=None,
        help="Poke Y coordinate in pixels.",
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
