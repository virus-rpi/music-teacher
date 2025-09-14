from pathlib import Path
import tempfile
import glob
import os
import shutil
import cv2
from skimage.metrics import structural_similarity as ssim
import pygame
from PIL import Image, ImageOps
from music21 import converter, environment, note as m21note, stream as m21stream
import numpy as np
from collections import deque
from typing import Optional

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except Exception:
    _resampling = getattr(Image, "Resampling", Image)
    RESAMPLE_LANCZOS = getattr(_resampling, "LANCZOS", getattr(_resampling, "BICUBIC", getattr(Image, "NEAREST", 0)))

_WHITESPACE_THRESHOLD = 0.90  # fraction of white pixels in a row to be considered whitespace
_MIN_SYSTEM_HEIGHT = 18       # ignore tiny slices
_MORPH_BLUR = 3               # smooth projection to merge near whitespace gaps
_MIN_SYSTEM_GAP_FRAC = 0.03   # minimum fraction of page height for a gap to be considered inter-system
_GAP_PERCENTILE = 0.70        # percentile of whitespace gaps to classify as separators
_MIN_PADDING = 24             # minimum vertical padding around detected system
_MAX_PADDING = 140            # cap vertical padding so we don't eat entire page
_PADDING_FRAC = 0.22          # padding as fraction of detected system height

# Cropping of leading instrument/clef/key content
_BARLINE_SCAN_FRAC = 0.45     # only scan this fraction of width from left for first barline
_DARK_THRESH = 100            # grayscale threshold for dark pixel
_BARLINE_MIN_FRAC = 0.30      # min fraction of image height that must be dark in a column
_BARLINE_MIN_FRAC_HALF = 0.10 # min fraction in both top/bottom halves to avoid clef-only detections
_BARLINE_SMOOTH = 2           # smooth columns by this half-window

# Theme colors
_BG_GRAY = 20
_BG_RGBA = (20, 20, 20, 255)

_SPLIT_CACHE_VERSION = "v1"

class SheetMusicRenderer:
    def __init__(self, midi_path, screen_width, height=260, debug=True):
        """
        midi_path: path to midi file (music21 will parse it)
        screen_width: width in pixels of pygame window (used for view)
        height: vertical height of the final strip when scaled
        debug: if True, prints some diagnostics to stdout
        """
        self.midi_path = Path(midi_path)
        self.screen_width = int(screen_width)
        self.strip_height = int(height)
        self.debug = debug

        self.full_surface = None
        self.full_width = 0
        self.note_to_xs = {}  # midi -> list of x coords on the full strip (approx)
        self.system_boxes = []  # list of (x, w) for each system in order
        self.notehead_xs = []   # detected visual notehead x positions on the resized strip

        # do the heavy lifting now
        self._prepare_strip()

    # ------------------------------
    # Utilities: MuseScore / music21
    # ------------------------------
    def _ensure_musescore(self):
        """
        Try to make music21 use a MuseScore binary. If music21 already has a configured
        musicxmlPath it will be left alone. We try several common executable names.
        """
        env = environment.UserSettings()
        try:
            current = env['musicxmlPath']
            if current and self.debug:
                print("music21 already has musicxmlPath:", current)
                return True
        except Exception:
            pass

        # try common candidates; prefer explicit full path if available on PATH
        candidates = ["musescore", "mscore", "musescore4", "MuseScore4", "MuseScore3", "MuseScore"]
        found = None
        for c in candidates:
            if shutil.which(c):
                try:
                    env['musicxmlPath'] = shutil.which(c)
                    found = env['musicxmlPath']
                    if self.debug:
                        print("set music21 musicxmlPath to", found)
                    break
                except Exception:
                    continue

        if not found:
            # last attempt: set generic names (music21 will try to call them)
            for c in candidates:
                try:
                    env['musicxmlPath'] = c
                    found = c
                    if self.debug:
                        print("set music21 musicxmlPath to candidate", c)
                    break
                except Exception:
                    continue

        return found is not None

    # ------------------------------
    # Caching helpers
    # ------------------------------
    @staticmethod
    def _get_cache_dir():
        """
        Returns the cache directory path (creates if missing).
        """
        cache_dir = Path('.sheet_music_cache')
        cache_dir.mkdir(exist_ok=True)
        return cache_dir

    def _midi_cache_key(self):
        """
        Returns a unique cache key for the midi file (based on path and mtime).
        """
        try:
            stat = self.midi_path.stat()
            mtime = int(stat.st_mtime)
            size = stat.st_size
        except Exception:
            mtime = 0
            size = 0
        # Use filename, mtime, and size for cache key
        key = f"{self.midi_path.name}_{mtime}_{size}"
        return key

    def _export_pages_png(self, tmpdir: Path):
        """
        Use music21 to parse the midi and produce PNG page images into tmpdir.
        Returns list of file paths (sorted).
        Uses cache if available.
        """
        cache_dir = self._get_cache_dir()
        cache_key = self._midi_cache_key()
        cache_subdir = cache_dir / cache_key
        cache_subdir.mkdir(exist_ok=True)
        # Check for cached PNGs
        cached_pngs = sorted(glob.glob(str(cache_subdir / "score*.png")))
        if cached_pngs:
            # Copy cached PNGs to tmpdir
            for f in cached_pngs:
                shutil.copy(f, tmpdir)
            files = sorted(glob.glob(str(tmpdir / "score*.png")))
            # Load score from midi
            score = converter.parse(str(self.midi_path))
            if self.debug:
                print("Loaded PNG pages from cache:", files)
            return files, score

        # Not cached: render and cache
        if not self._ensure_musescore():
            if self.debug:
                print("Warning: MuseScore not found or couldn't be configured in music21. Export may fail.")

        score = converter.parse(str(self.midi_path))
        png_fp = tmpdir / "score.png"
        score.write("musicxml.png", fp=str(png_fp))
        glob_patterns = [str(tmpdir / "score*.png"), str(tmpdir / "score-*.png"), str(tmpdir / "score_*.png")]
        files = []
        for patt in glob_patterns:
            files.extend(glob.glob(patt))
        files = sorted(set(files))
        # Save PNGs to cache
        for f in files:
            shutil.copy(f, cache_subdir)
        if self.debug:
            print("Exported PNG pages:", files)
        return files, score

    # ------------------------------
    # Image processing helpers
    # ------------------------------
    @staticmethod
    def _is_row_blank(row, white_threshold=0.98):
        """
        Return True if the given row (sequence of pixel values 0..255 grayscale or alpha)
        is effectively blank/white.
        """
        if len(row) == 0:
            return True
        # compute fraction of pixels near white
        white_pixels = sum(1 for v in row if v >= 240)
        return (white_pixels / len(row)) >= white_threshold

    @staticmethod
    def _horizontal_projection_grayscale(im: Image.Image):
        """
        Return an array of normalized brightness per row (0..1), 1 means all white.
        """
        gray = im.convert("L")
        w, h = gray.size
        px = gray.load()
        proj = []
        for y in range(h):
            white = 0
            for x in range(w):
                if px[x, y] >= 240:
                    white += 1
            proj.append(white / max(1, w))
        return proj

    @staticmethod
    def _recolor_to_dark_theme(im: Image.Image):
        """Map white background to #141414 and black strokes to white, keeping alpha if present."""
        base = Image.new("RGB", im.size, (255, 255, 255))
        if im.mode != "RGBA":
            rgba = im.convert("RGBA")
        else:
            rgba = im
        base.paste(rgba, (0, 0), rgba)
        gray = ImageOps.grayscale(base)  # 0..255
        k = 255 - _BG_GRAY
        lut = [int(255 - (k * v) / 255) for v in range(256)]
        mapped = gray.point(lut)
        # Expand to RGB and attach original alpha (or full opaque if none)
        rgb = Image.merge("RGB", (mapped, mapped, mapped))
        try:
            alpha = rgba.split()[3]
        except Exception:
            alpha = Image.new("L", im.size, 255)
        out = Image.merge("RGBA", (*rgb.split(), alpha))
        return out

    # ------------------------------
    # Leading content cropping helpers
    # ------------------------------
    @staticmethod
    def _find_first_barline_x(im: Image.Image):
        gray = im.convert("L")
        w, h = gray.size
        px = gray.load()
        max_x = max(1, int(w * _BARLINE_SCAN_FRAC))
        # compute dark counts per column and per half
        col_tot = [0] * max_x
        col_top = [0] * max_x
        col_bot = [0] * max_x
        half_y = h // 2
        for x in range(max_x):
            t = 0
            top = 0
            bot = 0
            for y in range(h):
                v = px[x, y]
                if v < _DARK_THRESH:
                    t += 1
                    if y < half_y:
                        top += 1
                    else:
                        bot += 1
            col_tot[x] = t
            col_top[x] = top
            col_bot[x] = bot
        # smooth
        if _BARLINE_SMOOTH > 0:
            k = _BARLINE_SMOOTH
            def smooth(arr):
                out = [0.0] * len(arr)
                for i in range(len(arr)):
                    lo = max(0, i - k)
                    hi = min(len(arr), i + k + 1)
                    s = 0.0
                    for j in range(lo, hi):
                        s += arr[j]
                    out[i] = s / (hi - lo)
                return out
            col_tot = smooth(col_tot)
            col_top = smooth(col_top)
            col_bot = smooth(col_bot)
        # thresholds
        min_tot = max(int(h * _BARLINE_MIN_FRAC), 30)
        min_half = max(int(h * _BARLINE_MIN_FRAC_HALF), 8)
        # find first x satisfying all
        for x in range(max_x):
            if col_tot[x] >= min_tot and col_top[x] >= min_half and col_bot[x] >= min_half:
                return x
        return None

    def _maybe_crop_leading_content(self, im: Image.Image):
        """
        Crop leading instrument up to first barline.
        - Always crop on the very first encountered system (to remove instrument names).
        Returns (cropped_image, did_crop: bool).
        """
        bar_x = self._find_first_barline_x(im)
        if bar_x is None or bar_x <= 2:
            return im, False
        return im.crop((bar_x, 0, im.width, im.height)), True

    @staticmethod
    def _remove_tempo_marking(im_rgba: Image.Image) -> Image.Image:
        if im_rgba.mode != "RGBA":
            im = im_rgba.convert("RGBA")
        else:
            im = im_rgba
        gray = ImageOps.grayscale(im)
        arr = np.asarray(gray)
        h, w = arr.shape
        if h < 10 or w < 10:
            return im
        row_density = (arr >= 230).mean(axis=1)
        dense = row_density >= max(0.18, float(np.percentile(row_density, 98) * 0.6))
        dense = np.logical_or.reduce([dense, np.roll(dense, 1), np.roll(dense, -1)])
        ys = np.where(dense)[0]
        if ys.size == 0:
            top_staff = int(h * 0.12)
        else:
            top_staff = int(max(4, ys[0]))
        y1 = max(0, top_staff - max(20, int(0.6 * max(6, int(round(h * 0.025))))))
        if y1 <= 0:
            return im
        wiped = im.copy()
        wipe_img = Image.new("RGBA", (220, y1), _BG_RGBA)
        wiped.paste(wipe_img, (200, 0), wipe_img)
        return wiped

    # ------------------------------
    # System slicing and cropping
    # ------------------------------
    @staticmethod
    def _has_full_barline(im: Image.Image, y_start, y_end):
        """
        Return True if there is a vertical dark line connecting the region from y_start to y_end.
        """
        gray = im.convert("L")
        w, h = gray.size
        px = gray.load()
        for x in range(w):
            dark_count = sum(1 for y in range(y_start, y_end) if px[x, y] < _DARK_THRESH)
            if dark_count >= (y_end - y_start) * 0.9:  # at least 90% of rows are dark
                return True
        return False

    def _find_system_slices(self, im: Image.Image):
        w, h = im.size
        proj = self._horizontal_projection_grayscale(im)

        # smooth projection with a small moving average to avoid tiny gaps splitting systems
        kernel = _MORPH_BLUR
        smoothed = []
        for i in range(len(proj)):
            lo = max(0, i - kernel)
            hi = min(len(proj), i + kernel + 1)
            smoothed.append(sum(proj[lo:hi]) / (hi - lo))

        # rows where smoothed value is close to 1 are whitespace; treat those as separators
        whitespace = [v >= _WHITESPACE_THRESHOLD for v in smoothed]

        # Identify contiguous whitespace runs
        whitespace_runs = []  # (start, end) with end exclusive
        in_ws = False
        ws_start = 0
        for y, is_ws in enumerate(whitespace):
            if is_ws and not in_ws:
                in_ws = True
                ws_start = y
            elif not is_ws and in_ws:
                in_ws = False
                whitespace_runs.append((ws_start, y))
        if in_ws:
            whitespace_runs.append((ws_start, len(whitespace)))

        # Internal gaps for thresholding
        internal_gaps = []
        for s, e in whitespace_runs:
            if s == 0 or e == len(whitespace):
                continue
            internal_gaps.append(e - s)

        if not internal_gaps:
            return [(0, h)]

        # dynamic gap threshold
        gaps_sorted = sorted(internal_gaps)
        idx = max(0, min(len(gaps_sorted) - 1, int(len(gaps_sorted) * _GAP_PERCENTILE)))
        perc_thresh = gaps_sorted[idx]
        min_abs_gap = max(18, int(h * _MIN_SYSTEM_GAP_FRAC))
        gap_threshold = max(perc_thresh, min_abs_gap)

        # helper: check for full vertical barline in a region
        def has_full_barline(y_start, y_end):
            gray = im.convert("L")
            px = gray.load()
            for x in range(w):
                dark_count = sum(1 for y in range(y_start, y_end) if px[x, y] < _DARK_THRESH)
                if dark_count >= (y_end - y_start) * 0.9:  # 90% of rows dark
                    return True
            return False

        # Select separators: large gaps that do NOT have a full vertical barline
        separators = []
        for s, e in whitespace_runs:
            if s == 0 or e == len(whitespace):
                continue
            gap_size = e - s
            if gap_size >= gap_threshold and not has_full_barline(s, e):
                separators.append((s, e))

        # fallback: if no separator, take largest gap that isn’t top/bottom
        if not separators:
            largest = max(((e - s, (s, e)) for s, e in whitespace_runs if s != 0 and e != len(whitespace)),
                          default=None)
            if largest:
                separators = [largest[1]]

        # build segments between separators
        segments = []
        prev_end = 0
        for (s, e) in separators:
            seg = (prev_end, s)
            if (seg[1] - seg[0]) >= _MIN_SYSTEM_HEIGHT:
                segments.append(seg)
            prev_end = e
        # trailing segment
        if (h - prev_end) >= _MIN_SYSTEM_HEIGHT:
            segments.append((prev_end, h))

        # fallback if nothing found
        if not segments:
            segments = [(0, h)]

        return segments

    def _crop_redundant_headers(self, slices: list[Image.Image]) -> list[Image.Image]:
        """
        Removes redundant clef/key/time markers at the start of each system strip by:
        - Dynamically estimating the header width using the first detected notehead x.
        - Normalizing header patches: remove staff lines and trim empty margins.
        - Comparing normalized patches with SSIM to be robust to small shifts.
        """
        processed: list[Image.Image] = []
        prev_headers = None  # (treble_patch, bass_patch)

        if self.debug:
            print("Cropping redundant headers...")

        def _threshold_and_staff(mask_source: np.ndarray):
            # Compute threshold and staff rows on a binary mask derived from grayscale
            thr = self._otsu_threshold(mask_source)
            thr = min(250, max(100, int(thr * 1.05)))
            mask = mask_source >= thr
            staff_space = self._estimate_staff_space(mask)
            row_density = mask.mean(axis=1)
            row_thr = max(float(np.percentile(row_density, 99) * 0.7), 0.12)
            staff_rows = row_density >= row_thr
            staff_rows = np.logical_or.reduce([
                staff_rows,
                np.roll(staff_rows, 1),
                np.roll(staff_rows, -1)
            ])
            return mask, staff_rows, staff_space

        def _prep_patch(gray_half: np.ndarray, global_mask: np.ndarray, staff_rows: np.ndarray) -> Optional[np.ndarray]:
            # Remove staff lines and trim to content; return uint8 image with 0 background and 255 strokes
            if gray_half.size == 0:
                return None
            # Build a local mask from global (slice rows subset)
            h = gray_half.shape[0]
            # global_mask aligns with full system; we pass already sliced halves, so rebuild local via threshold too
            thr = self._otsu_threshold(gray_half)
            thr = min(250, max(100, int(thr * 1.05)))
            mask_local = gray_half >= thr
            # Estimate staff rows in local half and remove them
            row_density = mask_local.mean(axis=1) if h > 0 else np.array([0.0])
            row_thr = max(float(np.percentile(row_density, 99) * 0.7) if row_density.size else 0.0, 0.12)
            staff_rows_local = row_density >= row_thr if row_density.size else np.zeros((h,), dtype=bool)
            staff_rows_local = np.logical_or.reduce([
                staff_rows_local,
                np.roll(staff_rows_local, 1) if h > 1 else staff_rows_local,
                np.roll(staff_rows_local, -1) if h > 1 else staff_rows_local,
            ])
            mask_wo_staff = mask_local.copy()
            if h > 0:
                mask_wo_staff[staff_rows_local, :] = False
            if not mask_wo_staff.any():
                return None
            ys, xs = np.where(mask_wo_staff)
            if xs.size == 0 or ys.size == 0:
                return None
            minx, maxx = int(xs.min()), int(xs.max())
            miny, maxy = int(ys.min()), int(ys.max())
            patch = mask_wo_staff[miny:maxy+1, minx:maxx+1]
            # Convert to uint8 0..255
            patch_u8 = (patch.astype(np.uint8)) * 255
            return patch_u8

        def _compare_patches(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
            if a is None or b is None:
                return 0.0
            # Resize to same width and pad to same height; keep aspect to be translation tolerant
            target_w = 120
            def _resize_keep_w(img: np.ndarray):
                h, w = img.shape
                if w == 0 or h == 0:
                    return None
                scale = target_w / float(w)
                new_h = max(1, int(round(h * scale)))
                return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)
            a_r = _resize_keep_w(a)
            b_r = _resize_keep_w(b)
            if a_r is None or b_r is None:
                return 0.0
            hmax = max(a_r.shape[0], b_r.shape[0])
            def _pad_h(img: np.ndarray):
                pad = hmax - img.shape[0]
                if pad <= 0:
                    return img
                return np.pad(img, ((0, pad), (0, 0)), mode="constant", constant_values=0)
            a_p = _pad_h(a_r)
            b_p = _pad_h(b_r)
            try:
                return float(ssim(a_p, b_p))
            except Exception:
                # Fallback: normalized dot-product similarity
                a_f = (a_p.astype(np.float32) / 255.0).ravel()
                b_f = (b_p.astype(np.float32) / 255.0).ravel()
                denom = float(np.linalg.norm(a_f) * np.linalg.norm(b_f))
                return float((a_f @ b_f) / denom) if denom > 0 else 0.0

        cropped = 0
        for idx, im in enumerate(slices):
            # Prepare grayscale and mask for this system
            arr_gray = np.array(ImageOps.grayscale(im))
            h, w = arr_gray.shape
            if w <= 0 or h <= 0:
                processed.append(im)
                continue

            mask_full, staff_rows_full, staff_space = _threshold_and_staff(arr_gray)

            # Find first notehead x on this system; fallback to small fraction
            try:
                xs_local = self._detect_notehead_xs_on_image(im)
                first_note_x = min(xs_local) if xs_local else None
            except Exception:
                first_note_x = None

            if first_note_x is None:
                header_end_x = max(12, int(round(w * 0.075)))
            else:
                pad = max(8, int(round(0.6 * max(6, staff_space))))  # a lil bit before the first note
                header_end_x = max(8, min(w - 8, int(first_note_x - pad)))
                # If pad pushes beyond start, ensure a minimum header width
                if header_end_x < 10:
                    header_end_x = max(12, int(round(w * 0.06)))

            # Extract header ROI and split halves
            header_roi = arr_gray[:, :header_end_x]
            mid = h // 2
            treble_half = header_roi[:mid, :] if mid > 0 else header_roi
            bass_half = header_roi[mid:, :] if mid > 0 else None

            # Normalize patches (remove staff lines, trim empty)
            treble_patch = _prep_patch(treble_half, mask_full[:mid, :header_end_x] if mid > 0 else mask_full[:, :header_end_x], staff_rows_full[:mid] if mid > 0 else staff_rows_full)
            bass_patch = _prep_patch(bass_half, mask_full[mid:, :header_end_x] if (mid > 0 and bass_half is not None) else mask_full[:, :header_end_x], staff_rows_full[mid:] if mid > 0 else staff_rows_full) if bass_half is not None else None

            crop_needed = False
            if prev_headers is not None:
                prev_treble, prev_bass = prev_headers
                score_treble = _compare_patches(prev_treble, treble_patch)
                # If both present, compare bass too; otherwise rely on treble only
                score_bass = _compare_patches(prev_bass, bass_patch) if (prev_bass is not None and bass_patch is not None) else score_treble
                # Slightly relaxed threshold to allow for engraving changes
                if score_treble >= 0.86 and score_bass >= 0.86:
                    crop_needed = True

            # Apply crop if redundant header detected
            if crop_needed:
                im = im.crop((header_end_x, 0, w, h))
                cropped += 1

            # Update previous headers for next comparison (use current patches)
            prev_headers = (treble_patch, bass_patch)
            processed.append(im)

        if self.debug:
            print(f"Cropped {cropped} redundant headers.")

        return processed


    def _slice_page_into_system_images(self, page_path: str):
        im = Image.open(page_path).convert("RGBA")
        bbox = ImageOps.invert(im.convert("L")).getbbox()
        if bbox:
            # shrink horizontally only, keep vertical intact
            l, t, r, b = bbox
            im = im.crop((l, 0, r, im.height))

        systems = []
        segs = self._find_system_slices(im)
        page_h = im.height
        first_in_page = True
        for (top, bottom) in segs:
            seg_h = max(1, bottom - top)
            pad_est = int(seg_h * _PADDING_FRAC)
            pad = min(_MAX_PADDING, max(_MIN_PADDING, pad_est))
            t = max(0, top - pad)
            b = min(page_h, bottom + pad)
            crop = im.crop((0, t, im.width, b))
            crop2, did = self._maybe_crop_leading_content(crop)
            recolored = self._recolor_to_dark_theme(crop2)
            # remove tempo marking only for the first system of each page
            if first_in_page:
                recolored = self._remove_tempo_marking(recolored)
                first_in_page = False
            systems.append(recolored)
        systems = self._crop_redundant_headers(systems)
        return systems

    # ------------------------------
    # Notehead detection (visual)
    # ------------------------------
    @staticmethod
    def _otsu_threshold(gray_arr: np.ndarray) -> int:
        # gray_arr uint8
        hist = np.bincount(gray_arr.ravel(), minlength=256).astype(np.float64)
        total: float = float(gray_arr.size)
        sum_total: float = float(np.dot(np.arange(256), hist))
        sumB: float = 0.0
        wB: float = 0.0
        max_var: float = -1.0
        threshold: int = 127
        for t in range(256):
            wB += float(hist[t])
            if wB <= 0.0:
                continue
            wF = total - wB
            if wF <= 0.0:
                break
            sumB += float(t) * float(hist[t])
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            varBetween = wB * wF * (mB - mF) ** 2
            if varBetween > max_var:
                max_var = varBetween
                threshold = t
        return int(threshold)

    @staticmethod
    def _find_runs(mask_1d: np.ndarray):
        # returns list of (start, end) indices for True runs
        if mask_1d.size == 0:
            return []
        diffs = np.diff(mask_1d.astype(np.int8), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        return list(zip(starts, ends))

    def _estimate_staff_space(self, mask: np.ndarray) -> int:
        # mask: boolean foreground (notes+lines) with bright strokes on dark
        h, w = mask.shape
        row_density = mask.mean(axis=1)
        # consider very dense rows as staff-line candidates
        thr = max(np.percentile(row_density, 99) * 0.7, 0.12)
        staff_rows = row_density >= thr
        # merge thickness: dilate by 1 to join 2-3 px lines
        staff_rows = np.logical_or.reduce([
            staff_rows,
            np.roll(staff_rows, 1),
            np.roll(staff_rows, -1)
        ])
        runs = self._find_runs(staff_rows)
        centers = [int((s + e) // 2) for s, e in runs if (e - s) <= max(6, h // 100)]
        if len(centers) < 2:
            # heuristic fallback
            return max(6, int(round(h * 0.025)))
        diffs = np.diff(sorted(centers))
        diffs = diffs[(diffs >= 3) & (diffs <= h // 6)]
        if diffs.size == 0:
            return max(6, int(round(h * 0.025)))
        return int(max(5, np.median(diffs)))

    # NEW helpers for robust detection
    @staticmethod
    def _suppress_vertical_stems(mask: np.ndarray, staff_space: int) -> np.ndarray:
        """Remove tall, thin vertical runs (stems) from a binary mask."""
        h, w = mask.shape
        out = mask.copy()
        min_len = max(12, int(1.6 * staff_space))
        for x in range(w):
            y = 0
            while y < h:
                if out[y, x]:
                    y0 = y
                    while y < h and out[y, x]:
                        y += 1
                    run_len = y - y0
                    if run_len >= min_len:
                        out[y0:y, x] = False
                else:
                    y += 1
        return out

    def _connected_components(self, mask: np.ndarray):
        """Return list of components as dicts: {minx, maxx, miny, maxy, area, cx, cy}. 4-connected."""
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        comps = []
        # quick exit
        if not mask.any():
            return comps
        for y in range(h):
            row = mask[y]
            for x in range(w):
                if row[x] and not visited[y, x]:
                    q = deque()
                    q.append((y, x))
                    visited[y, x] = True
                    miny = maxy = y
                    minx = maxx = x
                    area = 0
                    sumx = 0.0
                    sumy = 0.0
                    while q:
                        cy, cx = q.popleft()
                        area += 1
                        sumx += cx
                        sumy += cy
                        if cy < miny: miny = cy
                        if cy > maxy: maxy = cy
                        if cx < minx: minx = cx
                        if cx > maxx: maxx = cx
                        if cy > 0 and mask[cy-1, cx] and not visited[cy-1, cx]:
                            visited[cy-1, cx] = True; q.append((cy-1, cx))
                        if cy + 1 < h and mask[cy+1, cx] and not visited[cy+1, cx]:
                            visited[cy+1, cx] = True; q.append((cy+1, cx))
                        if cx > 0 and mask[cy, cx-1] and not visited[cy, cx-1]:
                            visited[cy, cx-1] = True; q.append((cy, cx-1))
                        if cx + 1 < w and mask[cy, cx+1] and not visited[cy, cx+1]:
                            visited[cy, cx+1] = True; q.append((cy, cx+1))
                    if area > 0:
                        comps.append({
                            'minx': minx, 'maxx': maxx, 'miny': miny, 'maxy': maxy,
                            'area': area,
                            'cx': sumx / area,
                            'cy': sumy / area,
                        })
        return comps

    def _detect_notehead_xs_on_image(self, im: Image.Image) -> list:
        """
        Detect notehead x-positions on a single system image (recolored dark theme).
        Returns x positions in this image's coordinate space (pixels).
        """
        # Convert to grayscale ndarray
        if im.mode != "L":
            gray = ImageOps.grayscale(im)
        else:
            gray = im
        arr = np.asarray(gray)
        # In dark theme, background is dark (~20), strokes are bright (~255)
        # Use Otsu to split bright strokes robustly
        thr = self._otsu_threshold(arr)
        # Slightly bias threshold higher to avoid background noise
        thr = min(250, max(100, int(thr * 1.05)))
        mask = arr >= thr  # True where strokes (notes/lines) are

        h, w = mask.shape
        if h < 10 or w < 10:
            return []

        # Remove staff lines: identify dense rows and zero them
        staff_space = self._estimate_staff_space(mask)
        row_density = mask.mean(axis=1)
        row_thr = max(np.percentile(row_density, 99) * 0.7, 0.12)
        staff_rows = row_density >= row_thr
        staff_rows = np.logical_or.reduce([
            staff_rows,
            np.roll(staff_rows, 1),
            np.roll(staff_rows, -1)
        ])
        mask_wo_staff = mask.copy()
        mask_wo_staff[staff_rows, :] = False

        # Suppress vertical stems (long vertical runs)
        mask_no_stems = self._suppress_vertical_stems(mask_wo_staff, staff_space)

        # Optional: remove very thin horizontal runs (ledger/text lines) by zeroing rows with extremely low thickness
        col_density = mask_no_stems.mean(axis=0)
        if col_density.size:
            # if a column is extremely sparse, keep it; we target rows instead:
            row_density2 = mask_no_stems.mean(axis=1)
            low_rows = row_density2 <= max(0.02, float(np.percentile(row_density2, 30) * 0.8))
            # but don't kill too much: only if contiguous run is longer than width fraction (text lines)
            # We keep this conservative; comment out if it hurts recall
            # mask_no_stems[low_rows, :] = False
            pass

        # Connected components to find noteheads by size/shape
        comps = self._connected_components(mask_no_stems)
        if not comps:
            return []

        s = max(6, staff_space)
        min_area = int(0.35 * s * s)
        max_area = int(7.0 * s * s)
        min_side = max(3, int(0.45 * s))
        xs = []
        for c in comps:
            wbb = c['maxx'] - c['minx'] + 1
            hbb = c['maxy'] - c['miny'] + 1
            area = c['area']
            if area < min_area or area > max_area:
                continue
            if wbb < min_side or hbb < min_side:
                continue
            aspect = wbb / max(1.0, float(hbb))
            if aspect < 0.5 or aspect > 2.6:
                continue
            fill = area / float(wbb * hbb)
            if fill < 0.18 or fill > 0.90:
                continue
            xs.append(int(round(c['cx'])))

        # de-duplicate and sort
        xs = sorted(set(xs))
        # prune too-close duplicates
        pruned = []
        min_sep = max(4, int(round(0.6 * s)))
        for x0 in xs:
            if not pruned or (x0 - pruned[-1]) >= min_sep:
                pruned.append(x0)
        return pruned

    # ------------------------------
    # Main processing: build continuous strip
    # ------------------------------
    def _prepare_strip(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="sheetstrip_"))
        try:
            files, score = self._export_pages_png(tmpdir)
        except Exception as e:
            # fallback simpler behavior (use music21 score drawing fallback)
            print("SheetMusicRenderer: failed to export pages via MuseScore:", e)
            self._simple_fallback(None)
            return

        # persistent cache for split systems
        cache_dir = self._get_cache_dir()
        cache_key = self._midi_cache_key()
        systems_cache_dir = cache_dir / cache_key / f"systems_{_SPLIT_CACHE_VERSION}"
        cached_system_paths = sorted(glob.glob(str(systems_cache_dir / "sys_*.png"))) if systems_cache_dir.exists() else []

        all_system_images = []
        page_system_counts = []

        if cached_system_paths:
            # Load cached split systems
            try:
                for p in cached_system_paths:
                    im = Image.open(p).convert("RGBA")
                    all_system_images.append(im)
                if self.debug:
                    print(f"Loaded {len(all_system_images)} split systems from cache")
            except Exception as e:
                if self.debug:
                    print("Failed to load cached systems, will resplit:", e)
                all_system_images = []

        if not all_system_images:
            # split every page into systems and save to cache
            for f in files:
                try:
                    systems = self._slice_page_into_system_images(f)
                    if self.debug:
                        print(f"page {f} -> {len(systems)} systems")
                    page_system_counts.append(len(systems))
                    all_system_images.extend(systems)
                except Exception as e:
                    if self.debug:
                        print("failed slicing page", f, ":", e)
                    # fallback: use entire page
                    fallback_im = Image.open(f).convert("RGBA")
                    all_system_images.append(self._recolor_to_dark_theme(fallback_im))
                    page_system_counts.append(1)

            # save to systems cache
            try:
                systems_cache_dir.mkdir(parents=True, exist_ok=True)
                for idx, im in enumerate(all_system_images):
                    out_path = systems_cache_dir / f"sys_{idx:04d}.png"
                    # Use lossless PNG to preserve quality
                    im.save(out_path)
                if self.debug:
                    print(f"Saved {len(all_system_images)} split systems to cache: {systems_cache_dir}")
            except Exception as e:
                if self.debug:
                    print("Warning: failed to save system cache:", e)

        if not all_system_images:
            raise RuntimeError("No system images were produced from MuseScore pages.")

        # Now concatenate systems horizontally in page order to produce a continuous strip.
        widths = [im.width for im in all_system_images]
        heights = [im.height for im in all_system_images]
        total_width = sum(widths)
        max_height = max(heights)

        strip = Image.new("RGBA", (total_width, max_height), _BG_RGBA)
        x = 0
        self.system_boxes = []
        # Also collect notehead x positions per system before resizing
        per_system_note_xs = []
        for i, im in enumerate(all_system_images):
            y = (max_height - im.height) // 2
            strip.paste(im, (x, y), im)
            self.system_boxes.append((x, im.width))
            # detect noteheads on this system image
            try:
                xs_local = self._detect_notehead_xs_on_image(im)
            except Exception as e:
                if self.debug:
                    print("notehead detection failed on system", i, ":", e)
                xs_local = []
            per_system_note_xs.append((x, xs_local))
            x += im.width

        # scale strip to requested strip_height while keeping aspect ratio horizontally scaled
        scale = self.strip_height / max_height
        new_w = max(1, int(total_width * scale))
        strip_resized = strip.resize((new_w, self.strip_height), RESAMPLE_LANCZOS).convert("RGBA")

        # convert to pygame surface
        data = strip_resized.tobytes()
        self.full_surface = pygame.image.fromstring(data, strip_resized.size, "RGBA").convert_alpha()
        self.full_width = strip_resized.size[0]

        if self.debug:
            print(f"Built continuous strip: {len(all_system_images)} systems -> width={self.full_width}px at height={self.strip_height}px")

        # Map local detections to resized-strip x positions
        orig_full_width = sum(w for _, w in self.system_boxes) or 1
        sx = self.full_width / orig_full_width
        self.notehead_xs = []
        for sys_x, xs_local in per_system_note_xs:
            for lx in xs_local:
                self.notehead_xs.append(int((sys_x + lx) * sx))
        self.notehead_xs.sort()
        if self.debug:
            print(f"Detected {len(self.notehead_xs)} visual noteheads across strip")

        try:
            self._build_note_mapping_from_score(score)
        except Exception as e:
            if self.debug:
                print("Warning: building note mapping failed:", e)
            self.note_to_xs = {}

        # cleanup tmpdir
        try:
            for f in files:
                os.remove(f)
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    def _simple_fallback(self, score=None):
        """
        If museScore export fails, draw a simple two-staff strip using music21 note list or nothing.
        """
        w = max(self.screen_width * 3, 2000)
        h = self.strip_height
        img = Image.new("RGBA", (w, h), _BG_RGBA)
        # convert to pygame surface
        data = img.tobytes()
        self.full_surface = pygame.image.fromstring(data, (w, h), "RGBA").convert_alpha()
        self.full_width = w
        self.note_to_xs = {}
        self.notehead_xs = []

    # ------------------------------
    # Map notes -> x positions (approx)
    # ------------------------------
    def _build_note_mapping_from_score(self, score: m21stream.Score):
        """
        Build note_to_xs: midi -> list of x positions (float) on the full_surface.
        Strategy:
         - Collect all notes from the parsed music21 score in chronological order (offset).
         - Count total notes and split them across systems in the same order (approx each system gets proportionally
           the same number of notes as it visually occupies).
         - Inside each system, distribute the notes evenly across that system image width.
        This is approximate but gives usable highlights that line up roughly with visual note groups.
        """
        # collect notes from score (flattened)
        notes = []
        for n in score.recurse().getElementsByClass(m21note.Note):
            try:
                on = float(n.getOffsetInHierarchy(score))
            except Exception:
                on = float(n.offset if hasattr(n, "offset") else 0.0)
            midi = n.pitch.midi if n.pitch is not None else None
            notes.append((on, midi))
        if not notes:
            return
        notes.sort(key=lambda x: x[0])
        total_notes = len(notes)

        # compute how many notes per system by simple proportional split
        # We weight systems by their pixel width fraction of the full width
        system_widths = [w for (_, w) in self.system_boxes]
        total_sys_w = sum(system_widths) or 1
        sys_note_counts = [max(1, int(round((w / total_sys_w) * total_notes))) for w in system_widths]

        # Adjust rounding to match total_notes exactly
        diff = sum(sys_note_counts) - total_notes
        i = 0
        while diff != 0:
            if diff > 0:
                # remove 1 from some system (that has >1)
                if sys_note_counts[i] > 1:
                    sys_note_counts[i] -= 1
                    diff -= 1
            else:
                sys_note_counts[i] += 1
                diff += 1
            i = (i + 1) % len(sys_note_counts)

        # Now assign note indices to systems sequentially
        idx = 0
        note_positions = []  # x positions in pixel (unscaled)
        for sys_idx, cnt in enumerate(sys_note_counts):
            sys_x, sys_w = self.system_boxes[sys_idx]
            if sys_w <= 0:
                sys_w = 1
            for k in range(cnt):
                if idx >= total_notes:
                    break
                # position inside system: left margin + fraction
                frac = (k + 0.5) / max(1, cnt)
                # compute x on resized strip: scale original sys_x and sys_w to resized coordinates
                orig_full_width = sum(w for _, w in self.system_boxes)
                # compute scale factor used during resize
                if orig_full_width == 0:
                    sx = 0
                else:
                    sx = self.full_width / orig_full_width
                x_on_resized = int((sys_x + frac * sys_w) * sx)
                note_positions.append(x_on_resized)
                idx += 1
            if idx >= total_notes:
                break

        # map notes by midi to list of x positions
        self.note_to_xs = {}
        for ((_, midi), x) in zip(notes, note_positions):
            if midi is None:
                continue
            self.note_to_xs.setdefault(int(midi), []).append(x)

        if self.debug:
            counts = {m: len(xs) for m, xs in self.note_to_xs.items()}
            print("Built mapping: midi->count:", counts)

    # ------------------------------
    # Public: expose detected notehead x positions
    # ------------------------------
    def find_notehead_positions(self):
        """Return list of x pixel positions (on the resized strip) for detected noteheads."""
        return list(self.notehead_xs)

    # ------------------------------
    # Draw: called each frame
    # ------------------------------
    def draw(self, screen, y, progress, highlight_pitches=None):
        """
        Draw the strip at vertical position y with horizontal scrolling determined by progress (0..1).
        highlight_pitches: iterable of midi ints (pitches to highlight). optional.
        """
        if self.full_surface is None:
            return

        # clamp progress
        p = min(max(progress, 0.0), 1.0)
        view_w = self.screen_width
        x_off = int(p * max(0, (self.full_width - view_w)))

        # blit visible rect
        src_rect = pygame.Rect(x_off, 0, view_w, self.strip_height)
        screen.blit(self.full_surface, (0, y), src_rect)

        # draw playhead marker (center-ish)
        play_x = int(view_w * 0.45)  # a bit left-of-center so upcoming notes appear
        pygame.draw.line(screen, (0, 200, 255), (play_x, y), (play_x, y + self.strip_height), 3)

        # draw detected notehead positions as thin vertical lines (debug)
        if self.debug and self.notehead_xs:
            for gx in self.notehead_xs:
                sx = gx - x_off
                if -4 <= sx <= view_w + 4:
                    pygame.draw.line(screen, (90, 255, 120), (int(sx), y), (int(sx), y + self.strip_height), 1)

        # highlight requested pitches (approx mapping)
        if highlight_pitches:
            # build a list of (x_on_screen, color) for each occurrence we should show
            to_draw = []
            for midi in highlight_pitches:
                xs = self.note_to_xs.get(int(midi), [])
                if not xs:
                    continue
                # pick the occurrence closest to the playhead (x_off + play_x)
                abs_play_x = x_off + play_x
                best = min(xs, key=lambda xv: abs(xv - abs_play_x))
                screen_x = best - x_off
                # choose color by pitch range (simple): left-hand (low) blue, right-hand (high) orange
                color = (0, 200, 255) if midi >= 60 else (255, 140, 60)
                to_draw.append((screen_x, color))
            # draw highlight blobs/circles
            for sx, color in to_draw:
                # only draw if visible on screen
                if -24 <= sx <= view_w + 24:
                    r = 12
                    pygame.draw.circle(screen, color, (int(sx), int(y + self.strip_height * 0.5)), r)
                    # semi-transparent ring
                    surf = pygame.Surface((r*4, r*4), pygame.SRCALPHA)
                    pygame.draw.circle(surf, (*color, 90), (r*2, r*2), int(r*1.9))
                    screen.blit(surf, (int(sx - r*2), int(y + self.strip_height*0.5 - r*2)))

        # small border
        pygame.draw.rect(screen, (20,20,20), (0, y, view_w, self.strip_height), 2)
