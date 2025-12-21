import cv2
import numpy as np
import math
import csv
import os
import re
import tifffile as tiff
from embryo_region_map import (
    load_region_map,
    create_embryo_transform,
    get_region_for_point,
)


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
        # For 16-bit grayscale images, use intensity threshold directly (as percentile of max)
        # e.g., 0.9 means 90% of max intensity in frame
        intensity_threshold_percentile=0.85,
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
        self.intensity_threshold_percentile = intensity_threshold_percentile

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
        self.poke_detection_frame = None  # frame index where poke was detected (if different from specified)
        self.files_with_embryos = 0    # Count of files that had embryos detected
        self.region_map = None         # List of region definitions
        self.embryo_transforms = {}    # Dict mapping embryo_id -> transform dict

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
        
        For 16-bit images with narrow dynamic ranges, stretches contrast to full 8-bit
        range to preserve spark detection capability.
        
        Args:
            raw: numpy array from TIFF file
            is_bgr: if True, assumes input is already BGR (from OpenCV). 
                    if False, assumes RGB (from tifffile) and converts.
        """
        if raw.dtype == np.uint16:
            # For 16-bit, use contrast stretching to preserve narrow dynamic ranges
            # This is critical for scientific images where the full 16-bit range isn't used
            img_min = raw.min()
            img_max = raw.max()
            
            if img_max > img_min:
                # Linear stretch to full 8-bit range
                # Use float32 to avoid overflow during calculation
                raw8 = ((raw.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                # All pixels are the same value
                raw8 = np.full_like(raw, 128, dtype=np.uint8)
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
    
    def _detect_sparks_intensity(self, raw_16bit):
        """
        Detect bright sparks using intensity thresholds directly on 16-bit grayscale data.
        More precise than converting to 8-bit first.
        
        Returns (clusters, mask).
        """
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(raw_16bit.astype(np.float32), (5, 5), 0)
        
        # Use percentile-based threshold (e.g., 85% of max = bright sparks)
        img_max = blurred.max()
        threshold_value = img_max * self.intensity_threshold_percentile
        
        # Create mask for bright pixels
        mask = (blurred >= threshold_value).astype(np.uint8) * 255
        
        # Clean up
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

    def _detect_sparks(self, frame, raw_16bit=None):
        """
        Detect bright sparks in a frame.
        
        If raw_16bit is provided (grayscale 16-bit), uses intensity thresholds directly.
        Otherwise, uses HSV color-based detection on the BGR frame.
        
        Returns (clusters, mask).
        """
        # If we have raw 16-bit grayscale, use intensity-based detection (more precise)
        if raw_16bit is not None and raw_16bit.ndim == 2 and raw_16bit.dtype == np.uint16:
            return self._detect_sparks_intensity(raw_16bit)
        
        # Otherwise, use color-based HSV detection
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

    def _detect_cement_gland(self, end1, end2, axis_direction, mask, raw_16bit=None, frame_bgr=None):
        """
        Detect cement gland (dark spot on ventral anterior/chin region) to help identify head.
        
        The cement gland is a dark spot located on the ventral (bottom) side of the head.
        It should be visible even in dim images if brightness is increased.
        
        Args:
            end1: First endpoint (numpy array [x, y])
            end2: Second endpoint (numpy array [x, y])
            axis_direction: Normalized PCA axis direction vector (numpy array [dx, dy])
            mask: Binary mask of embryo (boolean array)
            raw_16bit: Optional raw 16-bit grayscale image for better detection
            frame_bgr: Optional BGR frame (fallback if raw_16bit not available)
        
        Returns:
            Tuple of (end_name, location) where:
            - end_name: 'end1' if cement gland found near end1, 'end2' if near end2, None if not found
            - location: (x, y) coordinates of cement gland, or None if not found
        """
        h, w = mask.shape
        
        # Get grayscale image with enhanced contrast
        if raw_16bit is not None and raw_16bit.ndim == 2:
            # Use raw 16-bit data, enhance contrast
            gray = raw_16bit.astype(np.float32)
            # Apply contrast enhancement: stretch to full range
            gray_min = gray[mask].min() if mask.sum() > 0 else gray.min()
            gray_max = gray[mask].max() if mask.sum() > 0 else gray.max()
            if gray_max > gray_min:
                gray_enhanced = ((gray - gray_min) / (gray_max - gray_min) * 255).astype(np.uint8)
            else:
                gray_enhanced = gray.astype(np.uint8)
        elif frame_bgr is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            # Enhance contrast within embryo region
            gray_float = gray.astype(np.float32)
            gray_min = gray_float[mask].min() if mask.sum() > 0 else gray_float.min()
            gray_max = gray_float[mask].max() if mask.sum() > 0 else gray_float.max()
            if gray_max > gray_min:
                gray_enhanced = ((gray_float - gray_min) / (gray_max - gray_min) * 255).astype(np.uint8)
            else:
                gray_enhanced = gray
        else:
            return None
        
        # Perpendicular vector (points to ventral side when rotated appropriately)
        u_perp = np.array([-axis_direction[1], axis_direction[0]])
        
        # Check both ends for dark spots in ventral region
        search_radius = min(30, np.linalg.norm(end2 - end1) * 0.15)  # 15% of embryo length, max 30px
        ventral_search_dist = search_radius * 0.5  # Look on ventral side
        
        results = {}
        
        for end_name, end_pt in [('end1', end1), ('end2', end2)]:
            # Define search region: ventral side near the endpoint
            # Ventral is in the direction of u_perp (or -u_perp, we'll check both)
            center_x, center_y = int(round(end_pt[0])), int(round(end_pt[1]))
            
            # Check ventral region (try both directions of perpendicular)
            best_darkness = None
            best_location = None
            
            for perp_dir in [u_perp, -u_perp]:
                ventral_center = end_pt + perp_dir * ventral_search_dist
                vx, vy = int(round(ventral_center[0])), int(round(ventral_center[1]))
                
                # Search in a small circular region around ventral center
                dark_pixels = []
                for dy in range(-int(search_radius), int(search_radius) + 1):
                    for dx in range(-int(search_radius), int(search_radius) + 1):
                        x, y = vx + dx, vy + dy
                        if 0 <= x < w and 0 <= y < h:
                            dist = np.hypot(dx, dy)
                            if dist <= search_radius and mask[y, x]:
                                # This is an embryo pixel in the search region
                                intensity = gray_enhanced[y, x]
                                dark_pixels.append((intensity, (x, y)))
            
            if dark_pixels:
                # Find the darkest pixels (cement gland should be darker than surrounding tissue)
                dark_pixels.sort(key=lambda p: p[0])  # Sort by intensity (darkest first)
                
                # Take darkest 10% of pixels in this region
                num_dark = max(1, len(dark_pixels) // 10)
                darkest = dark_pixels[:num_dark]
                avg_dark_intensity = np.mean([p[0] for p in darkest])
                
                # Compare to average intensity in the endpoint region
                endpoint_pixels = []
                for dy in range(-int(search_radius), int(search_radius) + 1):
                    for dx in range(-int(search_radius), int(search_radius) + 1):
                        x, y = center_x + dx, center_y + dy
                        if 0 <= x < w and 0 <= y < h and mask[y, x]:
                            dist = np.hypot(dx, dy)
                            if dist <= search_radius:
                                endpoint_pixels.append(gray_enhanced[y, x])
                
                if endpoint_pixels:
                    avg_endpoint_intensity = np.mean(endpoint_pixels)
                    # Cement gland should be significantly darker (at least 25% darker)
                    # This ensures we're detecting a real dark spot, not just noise
                    darkness_ratio = avg_dark_intensity / (avg_endpoint_intensity + 1e-9)
                    if darkness_ratio < 0.75:  # Darker than 75% of endpoint average (25% darker)
                        results[end_name] = {
                            'found': True,
                            'darkness_ratio': darkness_ratio,
                            'avg_dark_intensity': avg_dark_intensity,
                            'location': darkest[0][1]  # Location of darkest pixel
                        }
                        continue
            
            results[end_name] = {'found': False}
        
        # Determine which end has the cement gland
        end1_found = results.get('end1', {}).get('found', False)
        end2_found = results.get('end2', {}).get('found', False)
        
        if end1_found and not end2_found:
            location = results['end1'].get('location')
            return ('end1', location)
        elif end2_found and not end1_found:
            location = results['end2'].get('location')
            return ('end2', location)
        elif end1_found and end2_found:
            # Both found - use the darker one
            ratio1 = results['end1'].get('darkness_ratio', 1.0)
            ratio2 = results['end2'].get('darkness_ratio', 1.0)
            if ratio1 < ratio2:
                return ('end1', results['end1'].get('location'))
            else:
                return ('end2', results['end2'].get('location'))
        else:
            return (None, None)

    def _detect_embryos_count(self, frame_bgr, raw_16bit=None, verbose=False):
        """
        Quickly detect and count embryos without modifying state.
        Returns number of embryos found.
        """
        h, w = frame_bgr.shape[:2]
        
        # Use raw 16-bit if available, otherwise use grayscale from BGR frame
        if raw_16bit is not None and raw_16bit.ndim == 2:
            gray = raw_16bit.astype(np.float32)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            flat_intensities = blur.flatten()
            
            # Use more lenient thresholds - try multiple strategies
            # Strategy 1: Use median as threshold (should catch most non-background)
            median_intensity = np.median(flat_intensities)
            
            # Strategy 2: Use percentile that's more permissive
            p25 = np.percentile(flat_intensities, 25)
            
            # Strategy 3: Use mean (often works well for bimodal distributions)
            mean_intensity = flat_intensities.mean()
            
            # Use the lowest reasonable threshold to be more inclusive
            background_estimate = np.percentile(flat_intensities, 5)
            embryo_threshold = max(median_intensity * 0.7, p25, mean_intensity * 0.8, background_estimate * 1.2)
            
            embryo_mask = (blur >= embryo_threshold).astype(np.uint8) * 255
            
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            flat_intensities = blur.flatten()
            
            # More lenient for 8-bit too
            median_intensity = np.median(flat_intensities)
            p25 = np.percentile(flat_intensities, 25)
            mean_intensity = flat_intensities.mean()
            background_estimate = np.percentile(flat_intensities, 5)
            
            embryo_threshold = max(median_intensity * 0.7, p25, mean_intensity * 0.8, max(background_estimate * 1.2, 5))
            embryo_mask = (blur >= embryo_threshold).astype(np.uint8) * 255
        
        # Less aggressive morphology - embryos might be split
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        embryo_mask = cv2.morphologyEx(embryo_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        embryo_mask = cv2.morphologyEx(embryo_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(embryo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # More lenient minimum area - 0.5% of frame instead of 1%
        min_area = 0.005 * h * w
        emb_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        if verbose and len(emb_contours) == 0:
            # Diagnostic: why no contours?
            total_area = embryo_mask.sum() / 255
            print(f"    Diagnostic: mask_area={total_area:.0f}, min_area={min_area:.0f}, "
                  f"threshold={embryo_threshold:.1f}, contours_found={len(contours)}")
        
        return len(emb_contours)
    
    def _init_geometry_and_poke(self, frame_bgr, user_poke_xy=None, skip_poke_detection=False, raw_16bit=None):
        """
        From a single BGR frame (and optionally raw 16-bit data):

          * Segment up to 2 embryos using edge-based detection from non-black pixels.
          * Estimate head–tail axis (PCA of contour).
          * Precompute embryo_label_map, ap_norm_map, dv_map.
          * If user_poke_xy is None and skip_poke_detection is False, try to detect poke location from a pink arrow.
        
        Args:
            frame_bgr: BGR frame for visualization/fallback
            user_poke_xy: Optional poke coordinates
            skip_poke_detection: If True, skip poke detection
            raw_16bit: Optional raw 16-bit grayscale data for better detection
        """
        h, w = frame_bgr.shape[:2]
        self.height = h
        self.width = w

        # Use raw 16-bit if available, otherwise use grayscale from BGR frame
        if raw_16bit is not None and raw_16bit.ndim == 2:
            # Work with 16-bit data directly for better precision
            gray = raw_16bit.astype(np.float32)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Find background threshold: use a percentile of the darkest pixels
            # This identifies what "black" means in this image
            flat_intensities = blur.flatten()
            background_percentile = 10  # Bottom 10% of pixels = background
            background_threshold = np.percentile(flat_intensities, background_percentile)
            
            # Use more lenient thresholds - multiple strategies
            median_intensity = np.median(flat_intensities)
            p25 = np.percentile(flat_intensities, 25)
            mean_intensity = flat_intensities.mean()
            
            # Use the lowest reasonable threshold to be more inclusive
            embryo_threshold = max(median_intensity * 0.7, p25, mean_intensity * 0.8, background_threshold * 1.2)
            
            print(f"  16-bit embryo detection: background={background_threshold:.1f}, embryo_threshold={embryo_threshold:.1f}")
            
            # Create binary mask: embryo pixels (above threshold)
            embryo_mask = (blur >= embryo_threshold).astype(np.uint8) * 255
            
        else:
            # Fallback to 8-bit grayscale from BGR frame
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Find background: very dark pixels (near black)
            flat_intensities = blur.flatten()
            background_percentile = 10
            background_threshold = np.percentile(flat_intensities, background_percentile)
            
            # Use more lenient thresholds - multiple strategies
            median_intensity = np.median(flat_intensities)
            p25 = np.percentile(flat_intensities, 25)
            mean_intensity = flat_intensities.mean()
            
            # Use the lowest reasonable threshold to be more inclusive
            embryo_threshold = max(median_intensity * 0.7, p25, mean_intensity * 0.8, max(background_threshold * 1.2, 5))
            
            print(f"  8-bit embryo detection: background={background_threshold:.1f}, embryo_threshold={embryo_threshold:.1f}")
            
            # Create binary mask
            embryo_mask = (blur >= embryo_threshold).astype(np.uint8) * 255

        # Clean up the mask: fill holes and smooth edges (less aggressive)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        embryo_mask = cv2.morphologyEx(embryo_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        embryo_mask = cv2.morphologyEx(embryo_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(embryo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # More lenient minimum area - 0.5% instead of 1%
        min_area = 0.005 * h * w
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
        # Store temporary data for both embryos before finalizing
        embryo_data = []
        
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

            # Calculate centroid first (needed for morphological analysis)
            M = cv2.moments(contour)
            if M["m00"] == 0:
                cx, cy = float(mean[0, 0]), float(mean[0, 1])
            else:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            centroid_pt = np.array([cx, cy])

            # Step 1: Check for cement gland (dark spot on ventral anterior/chin) as sanity check
            # This is the FIRST step in head identification
            cement_gland_end, cement_gland_location = self._detect_cement_gland(
                end1, end2, v_norm, mask == 255, 
                raw_16bit=raw_16bit, frame_bgr=frame_bgr
            )
            
            if cement_gland_end == 'end1':
                print(f"  [Embryo {label}] ✓ Cement gland detected near end1 (head candidate)")
            elif cement_gland_end == 'end2':
                print(f"  [Embryo {label}] ✓ Cement gland detected near end2 (head candidate)")
            else:
                print(f"  [Embryo {label}] ⚠ Cement gland not detected (will use width analysis)")

            # Step 2: Determine head vs tail using morphological features
            # Head is typically wider/rounder than tail
            # Calculate width profile along the axis to identify which end is wider
            proj_normalized = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
            
            # Sample width at each end (average width in first/last 20% of length)
            end1_region_mask = (proj_normalized < 0.2)
            end2_region_mask = (proj_normalized > 0.8)
            end1_region = pts[end1_region_mask]
            end2_region = pts[end2_region_mask]
            
            # Calculate average distance from axis for each region
            if len(end1_region) > 0 and len(end2_region) > 0:
                # Project perpendicular distances
                u_perp = np.array([-v_norm[1], v_norm[0]])
                delta1 = end1_region - mean.reshape(1, 2)
                delta2 = end2_region - mean.reshape(1, 2)
                dist1 = np.abs(delta1 @ u_perp.reshape(2, 1)).ravel()
                dist2 = np.abs(delta2 @ u_perp.reshape(2, 1)).ravel()
                
                avg_width1 = dist1.mean() if len(dist1) > 0 else 0
                avg_width2 = dist2.mean() if len(dist2) > 0 else 0
                
                # Determine head based on cement gland first, then width analysis
                if cement_gland_end == 'end1':
                    # Cement gland found at end1, so end1 is head
                    head = end1
                    initial_tail = end2
                    if avg_width1 < avg_width2:
                        print(f"    → Note: Width analysis suggests end2 is wider, but cement gland at end1 confirms head")
                elif cement_gland_end == 'end2':
                    # Cement gland found at end2, so end2 is head
                    head = end2
                    initial_tail = end1
                    if avg_width2 < avg_width1:
                        print(f"    → Note: Width analysis suggests end1 is wider, but cement gland at end2 confirms head")
                else:
                    # No cement gland found, use width analysis
                    # Wider end = head (more bulbous/rounded)
                    # Use a threshold to avoid noise: only flip if difference is significant
                    width_diff_ratio = abs(avg_width1 - avg_width2) / (max(avg_width1, avg_width2) + 1e-9)
                    if width_diff_ratio > 0.1:  # At least 10% difference
                        if avg_width1 > avg_width2:
                            head = end1
                            initial_tail = end2
                        else:
                            head = end2
                            initial_tail = end1
                    else:
                        # Widths too similar, use area near endpoint as fallback
                        # Can't determine from width, keep as-is but warn
                        head = end1  # Default assignment
                        initial_tail = end2
                        print(f"  ⚠ Warning: Cannot determine head/tail from morphology for embryo {label}. "
                              f"Using default assignment. Consider manual specification.")
            else:
                # Not enough points in regions, use cement gland if available, otherwise default
                if cement_gland_end == 'end1':
                    head = end1
                    initial_tail = end2
                    print(f"    → Using cement gland detection (insufficient points for width analysis)")
                elif cement_gland_end == 'end2':
                    head = end2
                    initial_tail = end1
                    print(f"    → Using cement gland detection (insufficient points for width analysis)")
                else:
                    head = end1
                    initial_tail = end2
                    print(f"  ⚠ Warning: Cannot determine head/tail from morphology for embryo {label}. "
                          f"Insufficient points. Using default assignment.")

            # Validate that head and initial tail are different points
            head_tail_dist = np.linalg.norm(head - initial_tail)
            if head_tail_dist < 5.0:  # Minimum reasonable distance between head and tail (5 pixels)
                print(f"  ⚠ Warning: Embryo {label} has head and tail too close ({head_tail_dist:.2f} px). "
                      f"This may cause numerical issues. Consider manual specification.")
                # Ensure minimum separation
                if head_tail_dist < 1e-3:
                    # Head and tail are essentially the same point - use centroid as reference
                    dir_vec = np.array([1.0, 0.0])  # Default direction
                    head = centroid_pt - 50 * dir_vec
                    initial_tail = centroid_pt + 50 * dir_vec
                    print(f"    → Adjusted to use centroid-based axis")
                    head_tail_dist = np.linalg.norm(head - initial_tail)

            # Store temporary data for later processing
            embryo_data.append({
                "label": label,
                "mask": (mask == 255),
                "head": head.copy(),
                "initial_tail": initial_tail.copy(),
                "centroid": (float(cx), float(cy)),
                "axis_direction": v_norm.copy(),  # PCA axis direction
                "head_tail_dist": head_tail_dist,
                "cement_gland_location": cement_gland_location,  # Store cement gland location
            })

        # Step 2: Calculate consistent head-tail distance
        # Find both heads first, then use consistent distance for tails
        if len(embryo_data) >= 2:
            # Calculate standard head-tail length from both embryos
            head_tail_distances = [emb["head_tail_dist"] for emb in embryo_data]
            standard_head_tail_length = np.mean(head_tail_distances)
            print(f"\n  → Standard head-tail length: {standard_head_tail_length:.1f}px "
                  f"(from {len(head_tail_distances)} embryo(s))")
        elif len(embryo_data) == 1:
            # Only one embryo, use its distance as standard
            standard_head_tail_length = embryo_data[0]["head_tail_dist"]
            print(f"\n  → Using single embryo head-tail length: {standard_head_tail_length:.1f}px")
        else:
            standard_head_tail_length = None

        # Step 3: Recalculate tails using consistent distance
        for emb_data in embryo_data:
            label = emb_data["label"]
            head = emb_data["head"]
            axis_direction = emb_data["axis_direction"]
            
            # Calculate tail using standard distance along PCA axis
            if standard_head_tail_length is not None and standard_head_tail_length > 1e-3:
                # Direction from head to initial tail (to determine orientation)
                initial_tail = emb_data["initial_tail"]
                initial_axis = initial_tail - head
                initial_axis_norm = initial_axis / (np.linalg.norm(initial_axis) + 1e-9)
                
                # Use PCA axis direction, but ensure it points in same direction as initial tail
                if np.dot(axis_direction, initial_axis_norm) < 0:
                    axis_direction = -axis_direction
                
                # Calculate new tail position
                tail = head + axis_direction * standard_head_tail_length
            else:
                # Fallback to initial tail if standard length is invalid
                tail = emb_data["initial_tail"]
                print(f"  ⚠ Warning: Using initial tail for embryo {label} (standard length invalid)")

            self.embryos[label] = {
                "id": label,
                "mask": emb_data["mask"],
                "head": (float(head[0]), float(head[1])),
                "tail": (float(tail[0]), float(tail[1])),
                "centroid": emb_data["centroid"],
                "cement_gland": emb_data.get("cement_gland_location"),  # Store cement gland location
            }
            self.embryo_labels.append(label)
            embryo_masks.append(emb_data["mask"])

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
            
            # Skip AP/DV calculation if head and tail are too close (invalid axis)
            if L < 1.0:  # Minimum reasonable axis length (1 pixel)
                print(f"  ⚠ Warning: Embryo {label} has invalid axis (head and tail too close: L={L:.3f}). Skipping AP/DV mapping.")
                # Mark all pixels as NaN for this embryo
                self.ap_norm_map[mask] = np.nan
                self.dv_map[mask] = np.nan
                continue
            
            # Normalize axis vector (safe division since L >= 1.0)
            u = axis / L
            u_perp = np.array([-u[1], u[0]], dtype=np.float32)

            # Compute AP and DV coordinates
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            
            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            delta = pts - head.reshape(1, 2)
            
            # Compute AP position (projection along axis, normalized by axis length)
            # Use np.dot which is more stable than @ for 1D arrays
            ap_proj = np.dot(delta, u)
            ap = ap_proj / L
            
            # Compute DV position (perpendicular distance)
            dv = np.dot(delta, u_perp)
            
            # Handle NaN/Inf values and clip extreme values to reasonable ranges
            ap = np.nan_to_num(ap, nan=0.0, posinf=10.0, neginf=-10.0)
            ap = np.clip(ap, -10.0, 10.0)
            dv = np.nan_to_num(dv, nan=0.0, posinf=1000.0, neginf=-1000.0)
            dv = np.clip(dv, -1000.0, 1000.0)
            
            self.ap_norm_map[ys, xs] = ap
            self.dv_map[ys, xs] = dv

        # Initialize region map and create transforms for each embryo
        self.region_map = load_region_map()  # Load default region map
        self.embryo_transforms = {}
        
        for label in self.embryo_labels:
            emb = self.embryos[label]
            head = emb['head']
            tail = emb['tail']
            # Create transform for this embryo (map coordinates default to left-to-right)
            # For embryo A (left side), head should map to left side of map
            # For embryo B (right side), head should map to right side of map
            if label == 'A':
                # A is on left, so head maps to left (tail region), tail maps to right
                map_head = (0.1, 0.5)  # Left side, middle (Tail region)
                map_tail = (0.95, 0.1)  # Right side, top (Brain region)
            else:
                # B is on right, so head maps to right (brain region), tail maps to left
                map_head = (0.95, 0.1)  # Right side, top (Brain region)
                map_tail = (0.1, 0.5)   # Left side, middle (Tail region)
            
            transform = create_embryo_transform(
                head, tail,
                map_head=map_head,
                map_tail=map_tail
            )
            self.embryo_transforms[label] = transform

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

        # Poke detection: use user coords if provided, else try to auto-detect (unless skipped)
        if not skip_poke_detection:
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

    def _detect_poke_from_first_sparks(self, early_sparks_data, time_window_s=5.0):
        """
        Detect poke location from the centroid of the first spark clusters.
        
        The poke typically triggers the first Ca²⁺ sparks, so we use the earliest
        spark clusters as a proxy for the poke location.
        
        Args:
            early_sparks_data: List of dicts with keys: 'frame_idx', 'time_s', 'clusters'
                              Clusters should be list of dicts with 'x', 'y', 'area'
            time_window_s: Maximum time after poke to consider (default 5 seconds)
        
        Returns:
            (x, y) tuple or None if insufficient data
        """
        if not early_sparks_data:
            return None
        
        # Collect all clusters within the time window
        all_early_clusters = []
        
        for frame_data in early_sparks_data:
            time_s = frame_data.get('time_s', float('inf'))
            if time_s < 0 or time_s > time_window_s:
                continue  # Only consider post-poke clusters
            
            clusters = frame_data.get('clusters', [])
            for cluster in clusters:
                if 'x' in cluster and 'y' in cluster:
                    all_early_clusters.append(cluster)
        
        if len(all_early_clusters) == 0:
            return None
        
        # Calculate weighted centroid (weighted by area, larger sparks more important)
        total_area = sum(c.get('area', 1) for c in all_early_clusters)
        if total_area == 0:
            total_area = len(all_early_clusters)  # Fallback to unweighted
        
        centroid_x = sum(c['x'] * c.get('area', 1) for c in all_early_clusters) / total_area
        centroid_y = sum(c['y'] * c.get('area', 1) for c in all_early_clusters) / total_area
        
        # If embryos are detected, prefer centroid near embryo surface
        if self.embryo_union_mask is not None:
            # Check if centroid is near an embryo
            cy_int = int(round(centroid_y))
            cx_int = int(round(centroid_x))
            
            h, w = self.embryo_union_mask.shape
            if 0 <= cy_int < h and 0 <= cx_int < w:
                if not self.embryo_union_mask[cy_int, cx_int]:
                    # Not on embryo, find closest point on embryo surface
                    emb_uint8 = np.where(self.embryo_union_mask, 0, 255).astype(np.uint8)
                    distmap = cv2.distanceTransform(emb_uint8, cv2.DIST_L2, 3)
                    
                    # Search in a radius around centroid for closest embryo point
                    search_radius = 50
                    min_dist = float('inf')
                    best_on_embryo = None
                    
                    for dy in range(-search_radius, search_radius + 1):
                        for dx in range(-search_radius, search_radius + 1):
                            y = cy_int + dy
                            x = cx_int + dx
                            if 0 <= y < h and 0 <= x < w:
                                if self.embryo_union_mask[y, x]:
                                    dist_to_centroid = math.hypot(dx, dy)
                                    if dist_to_centroid < min_dist:
                                        min_dist = dist_to_centroid
                                        best_on_embryo = (x, y)
                    
                    if best_on_embryo is not None:
                        # Use point on embryo that's closest to spark centroid
                        return best_on_embryo
        
        return (float(centroid_x), float(centroid_y))

    # ---------- tracking & annotation ----------

    def _annotate_state(self, state):
        """
        Attach embryo_id, ap_norm, dv_px, dist_from_poke_px, and region to a track state.
        """
        x = state["x"]
        y = state["y"]

        embryo_id = ""
        ap_norm = ""
        dv_px = ""
        region = ""

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
                    
                    # Add region mapping if transform is available
                    if (self.embryo_transforms is not None and 
                        embryo_id in self.embryo_transforms and
                        self.region_map is not None):
                        transform = self.embryo_transforms[embryo_id]
                        region = get_region_for_point(x, y, transform, self.region_map)

        state["embryo_id"] = embryo_id
        state["ap_norm"] = ap_norm if ap_norm != "" else ""
        state["dv_px"] = dv_px if dv_px != "" else ""
        state["region"] = region if region != "" else ""

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
        self.poke_detection_frame = None  # Reset for each processing run
        self.files_with_embryos = 0  # Reset for each processing run

        print(f"\n{'='*60}")
        print(f"Processing TIFF sequence from: {folder_path}")
        print(f"{'='*60}\n")
        
        # Extract folder name for filename prefix
        folder_name = os.path.basename(os.path.normpath(folder_path))
        
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
        
        # Build frame index mapping (frame_idx -> (file_path, page_idx)) without loading images
        print("Building frame index mapping...")
        frame_mapping = []  # List of (file_path, page_idx, use_tifffile) tuples
        for path in paths:
            try:
                num_pages, use_tifffile = self._get_tiff_page_count(path)
                for page_idx in range(num_pages):
                    frame_mapping.append((path, page_idx, use_tifffile))
            except Exception as e:
                print(f"⚠ Warning: Could not read page count from {os.path.relpath(path, folder_path)}: {str(e)}")
                # Assume single page
                frame_mapping.append((path, 0, False))
        
        total_frames_available = len(frame_mapping)
        print(f"Total frames available: {total_frames_available}\n")
        
        # Validate poke_frame_idx
        if poke_frame_idx < 0 or poke_frame_idx >= total_frames_available:
            raise ValueError(f"poke_frame_idx {poke_frame_idx} is out of range. "
                           f"Valid range: 0 to {total_frames_available - 1}")

        # Try to detect embryos from first frame of each file until we find one with 2 embryos
        print("Searching for reference frame with embryos...")
        reference_found = False
        files_with_embryos = 0
        reference_frame_idx = None
        reference_frame = None
        reference_raw_16bit = None
        best_embryo_count = 0
        height = None
        width = None
        
        # Get unique files (first page of each file)
        unique_files = {}
        for frame_idx, (path, page_idx, use_tifffile) in enumerate(frame_mapping):
            if path not in unique_files:
                unique_files[path] = (frame_idx, page_idx, use_tifffile)
        
        print(f"Checking first frame of {len(unique_files)} file(s) for embryos...\n")
        
        for file_idx, (file_path, (first_frame_idx, first_page_idx, use_tifffile)) in enumerate(unique_files.items()):
            try:
                # Get page count for this file
                num_pages, _ = self._get_tiff_page_count(file_path)
                
                # Try multiple pages: first, middle, and a few others
                pages_to_check = [0]  # Always check first page
                if num_pages > 1:
                    pages_to_check.append(min(50, num_pages // 2))  # Middle or page 50
                if num_pages > 100:
                    pages_to_check.append(100)  # Page 100
                if num_pages > 200:
                    pages_to_check.append(200)  # Page 200
                
                best_num_embryos = 0
                best_page_idx = 0
                best_frame = None
                best_raw_16bit = None
                
                rel_path = os.path.relpath(file_path, folder_path)
                
                # Check multiple pages and use the best one
                for check_page_idx in pages_to_check:
                    if check_page_idx >= num_pages:
                        continue
                    
                    # Read frame from this page
                    raw, is_bgr = self._read_tiff_page(file_path, page_idx=check_page_idx, use_tifffile=use_tifffile)
                    frame = self._prepare_frame(raw, is_bgr=is_bgr)
                    
                    if height is None:
                        height, width = frame.shape[:2]
                        self.height = height
                        self.width = width
                        print(f"Frame dimensions: {width}x{height} pixels\n")
                    
                    # Prepare raw 16-bit data for embryo detection if available
                    raw_16bit = None
                    if raw.dtype == np.uint16 and raw.ndim == 2:
                        raw_16bit = raw
                    
                    # Count embryos without modifying state
                    verbose_diag = (file_idx < 3 and check_page_idx == 0)  # Show diagnostics for first file, first page
                    num_embryos = self._detect_embryos_count(frame, raw_16bit=raw_16bit, verbose=verbose_diag)
                    
                    # Keep track of best result from this file
                    if num_embryos > best_num_embryos:
                        best_num_embryos = num_embryos
                        best_page_idx = check_page_idx
                        best_frame = frame.copy()
                        if raw_16bit is not None:
                            best_raw_16bit = raw_16bit.copy()
                    
                    del raw, frame
                
                # Report best result from this file
                if best_num_embryos > 0:
                    files_with_embryos += 1
                    self.files_with_embryos = files_with_embryos
                    page_note = f" (page {best_page_idx + 1})" if num_pages > 1 else ""
                    print(f"  File {file_idx + 1}/{len(unique_files)}: {rel_path}{page_note} → {best_num_embryos} embryo(s) detected")
                elif file_idx < 5:  # Show first 5 failures
                    print(f"  File {file_idx + 1}/{len(unique_files)}: {rel_path} → 0 embryos")
                
                # If we found 2 embryos, use this as reference
                if best_num_embryos == 2 and not reference_found and best_frame is not None:
                    reference_found = True
                    # Find the actual frame index in frame_mapping
                    for fidx, (fpath, pidx, _) in enumerate(frame_mapping):
                        if fpath == file_path and pidx == best_page_idx:
                            reference_frame_idx = fidx
                            break
                    reference_frame = best_frame
                    reference_raw_16bit = best_raw_16bit
                    print(f"    ✓ Using this file as reference (found 2 embryos)")
                    break
                elif best_num_embryos > 0 and not reference_found and best_frame is not None:
                    # Keep the best we've found so far (most embryos)
                    if best_num_embryos > best_embryo_count:
                        best_embryo_count = best_num_embryos
                        # Find the actual frame index
                        for fidx, (fpath, pidx, _) in enumerate(frame_mapping):
                            if fpath == file_path and pidx == best_page_idx:
                                reference_frame_idx = fidx
                                break
                        reference_frame = best_frame
                        reference_raw_16bit = best_raw_16bit
                
                if best_frame is not None:
                    del best_frame
                if best_raw_16bit is not None:
                    del best_raw_16bit
                
            except Exception as e:
                if file_idx < 5:  # Show first few errors
                    print(f"  ⚠ Warning: Could not check {os.path.relpath(file_path, folder_path)}: {str(e)}")
                continue
        
        if not reference_found:
            print(f"\n⚠ WARNING: No file with exactly 2 embryos found")
            if reference_frame is not None:
                print(f"  Using best available detection ({len(self.embryo_labels)} embryo(s))")
            else:
                print(f"  No embryos detected in any file")
        
        print(f"\n  Files with embryos detected: {files_with_embryos}/{len(unique_files)}")
        
        # Now initialize geometry with the reference frame we found
        if reference_frame is not None:
            print(f"\nInitializing geometry from reference frame...")
            skip_poke = (poke_frame_idx != 0)
            self._init_geometry_and_poke(reference_frame, user_poke_xy=poke_xy, skip_poke_detection=skip_poke, raw_16bit=reference_raw_16bit)
            del reference_frame
            if reference_raw_16bit is not None:
                del reference_raw_16bit
        else:
            print("\n⚠ WARNING: Could not find suitable reference frame for geometry")
            print("  → Geometry-based annotations will be empty")
            # Still need to set height/width from first frame for video writer
            if height is None:
                try:
                    first_path, first_page_idx, first_use_tifffile = frame_mapping[0]
                    first_raw, is_bgr = self._read_tiff_page(first_path, page_idx=first_page_idx, use_tifffile=first_use_tifffile)
                    first_frame = self._prepare_frame(first_raw, is_bgr=is_bgr)
                    self.height, self.width = first_frame.shape[:2]
                    del first_raw, first_frame
                except Exception:
                    pass
        
        # If poke was detected at frame 0, track it
        if self.poke_xy is not None and poke_frame_idx == 0:
            self.poke_detection_frame = 0
        
        # If poke is not at frame 0, try to detect poke from the actual poke frame (and forward if needed)
        if poke_xy is None and poke_frame_idx != 0:
            print(f"\nAnalyzing poke frame (frame {poke_frame_idx}) for poke detection...")
            poke_detected = False
            
            # Try the specified poke frame first
            if poke_frame_idx < len(frame_mapping):
                poke_path, poke_page_idx, poke_use_tifffile = frame_mapping[poke_frame_idx]
                print(f"Reading poke frame: {os.path.relpath(poke_path, folder_path)} (page {poke_page_idx + 1})")
                
                try:
                    poke_raw, is_bgr = self._read_tiff_page(poke_path, page_idx=poke_page_idx, use_tifffile=poke_use_tifffile)
                    poke_frame = self._prepare_frame(poke_raw, is_bgr=is_bgr)
                    del poke_raw
                    
                    # Try to detect poke from this frame
                    auto_poke = self._detect_poke_from_pink_arrow(poke_frame)
                    if auto_poke is not None:
                        self.poke_xy = auto_poke
                        self.poke_detection_frame = poke_frame_idx
                        poke_detected = True
                        print(f"✓ Auto-detected poke location from frame {poke_frame_idx}: ({auto_poke[0]:.1f}, {auto_poke[1]:.1f})")
                    del poke_frame
                except Exception as e:
                    print(f"⚠ Warning: Could not read poke frame for poke detection: {str(e)}")
            
            # If not found, search forward frames (up to 10 frames ahead)
            if not poke_detected:
                print(f"✗ Could not auto-detect poke location from frame {poke_frame_idx}")
                print("Searching subsequent frames for poke location...")
                search_frames = min(10, len(frame_mapping) - poke_frame_idx - 1)
                for offset in range(1, search_frames + 1):
                    search_frame_idx = poke_frame_idx + offset
                    search_path, search_page_idx, search_use_tifffile = frame_mapping[search_frame_idx]
                    print(f"  Trying frame {search_frame_idx}...", end=" ")
                    
                    try:
                        search_raw, is_bgr = self._read_tiff_page(search_path, page_idx=search_page_idx, use_tifffile=search_use_tifffile)
                        search_frame = self._prepare_frame(search_raw, is_bgr=is_bgr)
                        del search_raw
                        
                        auto_poke = self._detect_poke_from_pink_arrow(search_frame)
                        if auto_poke is not None:
                            self.poke_xy = auto_poke
                            self.poke_detection_frame = search_frame_idx
                            poke_detected = True
                            print(f"✓ FOUND at frame {search_frame_idx}: ({auto_poke[0]:.1f}, {auto_poke[1]:.1f})")
                            del search_frame
                            break
                        else:
                            print("not found")
                        del search_frame
                    except Exception:
                        print("error")
                        continue
                
                if not poke_detected:
                    print(f"  Could not find poke location in frames {poke_frame_idx} to {poke_frame_idx + search_frames}")
        
        # Override with user-provided poke coordinates if given
        if poke_xy is not None:
            self.poke_xy = poke_xy
            self.poke_detection_frame = None  # User-provided, not auto-detected
            print(f"\n✓ Using user-provided poke location: ({poke_xy[0]:.1f}, {poke_xy[1]:.1f})")
        
        # If poke still not detected, try to infer from first spark clusters
        if self.poke_xy is None:
            print("\n⚠ Poke location not detected via pink arrow.")
            print("  → Attempting to infer poke location from first spark clusters...")
            print("    (Processing early post-poke frames to find initial spark centroids)")
            
            # Process early frames to collect spark data
            early_sparks_data = []
            time_window_s = 5.0  # Look at first 5 seconds after poke
            max_frames_to_check = min(20, len(frame_mapping) - poke_frame_idx)  # Check up to 20 frames after poke
            
            for offset in range(0, max_frames_to_check):
                check_frame_idx = poke_frame_idx + offset
                if check_frame_idx >= len(frame_mapping):
                    break
                
                time_s = offset / fps if fps > 0 else 0
                if time_s > time_window_s:
                    break  # Beyond time window
                
                path, page_idx, use_tifffile = frame_mapping[check_frame_idx]
                
                try:
                    raw, is_bgr = self._read_tiff_page(path, page_idx=page_idx, use_tifffile=use_tifffile)
                    frame = self._prepare_frame(raw, is_bgr=is_bgr)
                    
                    # Detect sparks
                    raw_for_detection = None
                    if raw.dtype == np.uint16 and raw.ndim == 2:
                        raw_for_detection = raw
                    
                    clusters, _ = self._detect_sparks(frame, raw_16bit=raw_for_detection)
                    
                    if len(clusters) > 0:
                        early_sparks_data.append({
                            'frame_idx': check_frame_idx,
                            'time_s': time_s,
                            'clusters': clusters
                        })
                    
                    del raw, frame
                except Exception as e:
                    print(f"    ⚠ Warning: Could not read frame {check_frame_idx} for spark-based poke detection: {str(e)}")
                    continue
            
            # Try to detect poke from early sparks
            if early_sparks_data:
                inferred_poke = self._detect_poke_from_first_sparks(early_sparks_data, time_window_s=time_window_s)
                if inferred_poke is not None:
                    self.poke_xy = inferred_poke
                    self.poke_detection_frame = None  # Spark-based, not from a specific frame
                    print(f"  ✓ Inferred poke location from {sum(len(d['clusters']) for d in early_sparks_data)} early spark(s): ({inferred_poke[0]:.1f}, {inferred_poke[1]:.1f})")
                else:
                    print(f"  ✗ Could not infer poke location from {len(early_sparks_data)} frame(s) with sparks")
            else:
                print(f"  ✗ No sparks found in first {max_frames_to_check} post-poke frames")
        
        if self.poke_xy is None:
            print("\n⚠ WARNING: Poke location not detected. dist_from_poke_px will be empty.")
            print("  → Consider providing --poke-x/--poke-y if auto-detection fails.")
        
        print()

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
                "region",
                "filename",
            ],
        )
        csv_writer.writeheader()

        print("Processing frames and tracking sparks...")
        print("(Processing frames incrementally to conserve memory...)\n")
        
        total_frames = len(frame_mapping)
        total_raw_detections = 0  # Total raw spark detections (clusters) across all frames
        total_track_states = 0  # Total CSV rows written (track states)
        sparks_by_frame = {}  # Track sparks per frame for statistics
        
        # Process frames using the frame mapping we already built
        try:
            for frame_idx, (path, page_idx, use_tifffile) in enumerate(frame_mapping):
                rel_path = os.path.relpath(path, folder_path)
                # Prepend folder name to filename for visualization script matching
                if rel_path and not rel_path.startswith(folder_name):
                    rel_path = f"{folder_name}/{rel_path}"
                
                # Show progress for large files and when processing poke frame
                if frame_idx % 100 == 0 or frame_idx == total_frames - 1 or frame_idx == poke_frame_idx:
                    status = ""
                    if frame_idx == poke_frame_idx:
                        status = " [POKE FRAME - t=0]"
                    print(f"Processing frame {frame_idx + 1}/{total_frames}{status}...")
                
                try:
                    # Read only this one page
                    raw, is_bgr = self._read_tiff_page(path, page_idx=page_idx, use_tifffile=use_tifffile)
                    
                    frame = self._prepare_frame(raw, is_bgr=is_bgr)
                    
                except Exception as e:
                    print(f"\n⚠ WARNING: Failed to read/prepare frame {frame_idx} from {rel_path} page {page_idx + 1}: {str(e)}")
                    print("Skipping this frame...")
                    sparks_by_frame[frame_idx] = 0  # Track skipped frames as 0
                    continue

                time_s = (frame_idx - poke_frame_idx) / fps
                
                # Detect sparks - use raw 16-bit if grayscale, otherwise use prepared frame
                raw_for_detection = None
                if raw.dtype == np.uint16 and raw.ndim == 2:
                    raw_for_detection = raw  # Use raw 16-bit grayscale directly
                
                clusters, _ = self._detect_sparks(frame, raw_16bit=raw_for_detection)
                frame_states = self._update_tracks(frame_idx, time_s, clusters)
                
                # Free raw memory after detection
                del raw
                
                # Track spark statistics
                num_detections_this_frame = len(clusters)  # Raw detections in this frame
                num_states_this_frame = len(frame_states)  # Track states (CSV rows) for this frame
                total_raw_detections += num_detections_this_frame
                total_track_states += num_states_this_frame
                sparks_by_frame[frame_idx] = num_detections_this_frame
                
                # Show spark count when processing poke frame
                if frame_idx == poke_frame_idx:
                    print(f"  → Detected {num_detections_this_frame} spark detection(s), {num_states_this_frame} track state(s) in poke frame")

                # write CSV rows
                for s in frame_states:
                    row = dict(s)
                    for key in ["vx", "vy", "speed", "angle_deg", "ap_norm", "dv_px"]:
                        if row.get(key) is None or row.get(key) != row.get(key):  # NaN check
                            row[key] = ""
                    # Store relative path with page info if multi-page (page_idx > 0 means multi-page)
                    if page_idx > 0:
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
            print(f"\n⚠ WARNING: Error during processing: {str(e)}")
            raise
        finally:
            if writer is not None:
                writer.release()
            csv_file.close()

        # Calculate spark statistics
        frames_with_sparks = sum(1 for count in sparks_by_frame.values() if count > 0)
        max_detections_in_frame = max(sparks_by_frame.values()) if sparks_by_frame else 0
        avg_detections_per_frame = total_raw_detections / total_frames if total_frames > 0 else 0
        
        # Sanity check: if max in one frame is more than total, something is wrong
        if max_detections_in_frame > total_raw_detections:
            print(f"\n⚠ WARNING: Inconsistency detected - max detections in frame ({max_detections_in_frame}) "
                  f"exceeds total detections ({total_raw_detections}). This suggests a counting error.")
        
        # Verify we processed all frames
        if len(sparks_by_frame) != total_frames:
            print(f"\n⚠ WARNING: Frame count mismatch - processed {len(sparks_by_frame)} frames but expected {total_frames}")
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"\nSummary:")
        print(f"  • TIFF files processed: {len(paths)}")
        print(f"  • Total frames processed: {total_frames}")
        print(f"  • Spark detections:")
        print(f"    - Total raw detections: {total_raw_detections}")
        print(f"    - Total track states (CSV rows): {total_track_states}")
        print(f"    - Frames with detections: {frames_with_sparks}/{total_frames}")
        print(f"    - Average detections per frame: {avg_detections_per_frame:.2f}")
        print(f"    - Max detections in single frame: {max_detections_in_frame}")
        print(f"  • Unique tracks: {len(self.tracks)}")
        print(f"  • Embryos detected: {len(self.embryo_labels)} ({', '.join(self.embryo_labels) if self.embryo_labels else 'none'})")
        if self.files_with_embryos > 0:
            print(f"  • Files with embryos: {self.files_with_embryos}/{len(paths)}")
        poke_status = "✓ detected" if self.poke_xy is not None else "✗ not found"
        if self.poke_xy is not None:
            poke_status += f" at ({self.poke_xy[0]:.1f}, {self.poke_xy[1]:.1f})"
            if self.poke_detection_frame is not None:
                if self.poke_detection_frame != poke_frame_idx:
                    poke_status += f" (found at frame {self.poke_detection_frame}, specified: {poke_frame_idx})"
                else:
                    poke_status += f" (frame {self.poke_detection_frame})"
        print(f"  • Poke location: {poke_status}")
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
