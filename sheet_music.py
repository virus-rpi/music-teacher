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
from typing import Optional, Iterable

_resampling = getattr(Image, "Resampling", Image)
RESAMPLE_LANCZOS = getattr(_resampling, "LANCZOS", getattr(_resampling, "BICUBIC", getattr(Image, "NEAREST", 0)))

_WHITESPACE_THRESHOLD = 0.90
_MIN_SYSTEM_HEIGHT = 18
_MORPH_BLUR = 3
_MIN_SYSTEM_GAP_FRAC = 0.03
_GAP_PERCENTILE = 0.70
_MIN_PADDING = 24
_MAX_PADDING = 140
_PADDING_FRAC = 0.22

_BARLINE_SCAN_FRAC = 0.45
_DARK_THRESH = 100
_BARLINE_MIN_FRAC = 0.30
_BARLINE_MIN_FRAC_HALF = 0.10
_BARLINE_SMOOTH = 2

_BG_GRAY = 20
_BG_RGBA = (20, 20, 20, 255)

_SPLIT_CACHE_VERSION = "v1"


class SheetMusicRenderer:
    def __init__(self, midi_path: str | Path, screen_width: int, height: int = 260, debug: bool = True) -> None:
        self.midi_path = Path(midi_path)
        self.screen_width = int(screen_width)
        self.strip_height = int(height)
        self.debug = debug

        self.full_surface: Optional[pygame.Surface] = None
        self.full_width: int = 0
        self.note_to_xs: dict[int, list[int]] = {}
        self.system_boxes: list[tuple[int, int]] = []
        self.notehead_xs: list[int] = []
        self._notehead_xs_per_system: list[list[int]] = []

        self._prepare_strip()

    def _ensure_musescore(self) -> bool:
        env = environment.UserSettings()
        try:
            current = env["musicxmlPath"]
            if current and self.debug:
                print("music21 already has musicxmlPath:", current)
                return True
        except KeyError:
            pass

        candidates = ["musescore", "mscore", "musescore4", "MuseScore4", "MuseScore3", "MuseScore"]
        found: Optional[str] = None
        for c in candidates:
            path = shutil.which(c)
            if path:
                try:
                    env["musicxmlPath"] = path
                    found = env["musicxmlPath"]
                    if self.debug:
                        print("set music21 musicxmlPath to", found)
                    break
                except KeyError:
                    continue

        if not found:
            for c in candidates:
                try:
                    env["musicxmlPath"] = c
                    found = c
                    if self.debug:
                        print("set music21 musicxmlPath to candidate", c)
                    break
                except KeyError:
                    continue

        return found is not None

    @staticmethod
    def _get_cache_dir() -> Path:
        cache_dir = Path(".sheet_music_cache")
        cache_dir.mkdir(exist_ok=True)
        return cache_dir

    def _midi_cache_key(self) -> str:
        try:
            stat = self.midi_path.stat()
            mtime = int(stat.st_mtime)
            size = stat.st_size
        except OSError:
            mtime = 0
            size = 0
        key = f"{self.midi_path.name}_{mtime}_{size}"
        return key

    def _export_pages_png(self, tmpdir: Path) -> tuple[list[str], m21stream.Score]:
        cache_dir = self._get_cache_dir()
        cache_key = self._midi_cache_key()
        cache_subdir = cache_dir / cache_key
        cache_subdir.mkdir(exist_ok=True)
        cached_pngs = sorted(glob.glob(str(cache_subdir / "score*.png")))
        if cached_pngs:
            for f in cached_pngs:
                shutil.copy(f, tmpdir)
            files = sorted(glob.glob(str(tmpdir / "score*.png")))
            score = converter.parse(str(self.midi_path))
            if self.debug:
                print("Loaded PNG pages from cache:", files)
            return files, score

        if not self._ensure_musescore():
            if self.debug:
                print("Warning: MuseScore not found or couldn't be configured in music21. Export may fail.")

        score = converter.parse(str(self.midi_path))
        png_fp = tmpdir / "score.png"
        score.write("musicxml.png", fp=str(png_fp))
        glob_patterns = [str(tmpdir / "score*.png"), str(tmpdir / "score-*.png"), str(tmpdir / "score_*.png")]
        files: list[str] = []
        for patt in glob_patterns:
            files.extend(glob.glob(patt))
        files = sorted(set(files))
        for f in files:
            shutil.copy(f, cache_subdir)
        if self.debug:
            print("Exported PNG pages:", files)
        return files, score

    @staticmethod
    def _is_row_blank(row: Iterable[int], white_threshold: float = 0.98) -> bool:
        seq = list(row)
        if len(seq) == 0:
            return True
        white_pixels = sum(1 for v in seq if v >= 240)
        return (white_pixels / len(seq)) >= white_threshold

    @staticmethod
    def _horizontal_projection_grayscale(im: Image.Image) -> list[float]:
        gray = im.convert("L")
        w, h = gray.size
        px = gray.load()
        proj: list[float] = []
        for y in range(h):
            white = 0
            for x in range(w):
                if px[x, y] >= 240:
                    white += 1
            proj.append(white / max(1, w))
        return proj

    @staticmethod
    def _recolor_to_dark_theme(im: Image.Image) -> Image.Image:
        base = Image.new("RGB", im.size, (255, 255, 255))
        if im.mode != "RGBA":
            rgba = im.convert("RGBA")
        else:
            rgba = im
        base.paste(rgba, (0, 0), rgba)
        gray = ImageOps.grayscale(base)
        k = 255 - _BG_GRAY
        lut = [int(255 - (k * v) / 255) for v in range(256)]
        mapped = gray.point(lut)
        rgb = Image.merge("RGB", (mapped, mapped, mapped))
        try:
            alpha = rgba.split()[3]
        except IndexError:
            alpha = Image.new("L", im.size, 255)
        out = Image.merge("RGBA", (*rgb.split(), alpha))
        return out

    @staticmethod
    def _find_first_barline_x(im: Image.Image) -> Optional[int]:
        gray = im.convert("L")
        w, h = gray.size
        px = gray.load()
        max_x = max(1, int(w * _BARLINE_SCAN_FRAC))
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
        if _BARLINE_SMOOTH > 0:
            k = _BARLINE_SMOOTH

            def smooth(arr: list[float] | list[int]) -> list[float]:
                out = [0.0] * len(arr)
                for i in range(len(arr)):
                    lo = max(0, i - k)
                    hi = min(len(arr), i + k + 1)
                    s = 0.0
                    for j in range(lo, hi):
                        s += float(arr[j])
                    out[i] = s / (hi - lo)
                return out

            col_tot = smooth(col_tot)
            col_top = smooth(col_top)
            col_bot = smooth(col_bot)
        min_tot = max(int(h * _BARLINE_MIN_FRAC), 30)
        min_half = max(int(h * _BARLINE_MIN_FRAC_HALF), 8)
        for x in range(max_x):
            if col_tot[x] >= min_tot and col_top[x] >= min_half and col_bot[x] >= min_half:
                return x
        return None

    def _maybe_crop_leading_content(self, im: Image.Image) -> tuple[Image.Image, bool]:
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
            top_staff = int(max(4, int(ys[0])))
        y1 = max(0, top_staff - max(20, int(0.6 * max(6, int(round(h * 0.025))))))
        if y1 <= 0:
            return im
        wiped = im.copy()
        wipe_img = Image.new("RGBA", (220, y1), _BG_RGBA)
        wiped.paste(wipe_img, (200, 0), wipe_img)
        return wiped

    @staticmethod
    def _has_full_barline(im: Image.Image, y_start: int, y_end: int) -> bool:
        gray = im.convert("L")
        w, _ = gray.size
        px = gray.load()
        for x in range(w):
            dark_count = sum(1 for y in range(y_start, y_end) if px[x, y] < _DARK_THRESH)
            if dark_count >= (y_end - y_start) * 0.9:
                return True
        return False

    def _find_system_slices(self, im: Image.Image) -> list[tuple[int, int]]:
        w, h = im.size
        proj = self._horizontal_projection_grayscale(im)
        kernel = _MORPH_BLUR
        smoothed: list[float] = []
        for i in range(len(proj)):
            lo = max(0, i - kernel)
            hi = min(len(proj), i + kernel + 1)
            smoothed.append(sum(proj[lo:hi]) / (hi - lo))
        whitespace = [v >= _WHITESPACE_THRESHOLD for v in smoothed]
        whitespace_runs: list[tuple[int, int]] = []
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
        internal_gaps: list[int] = []
        for s, e in whitespace_runs:
            if s == 0 or e == len(whitespace):
                continue
            internal_gaps.append(e - s)
        if not internal_gaps:
            return [(0, h)]
        gaps_sorted = sorted(internal_gaps)
        idx = max(0, min(len(gaps_sorted) - 1, int(len(gaps_sorted) * _GAP_PERCENTILE)))
        perc_thresh = gaps_sorted[idx]
        min_abs_gap = max(18, int(h * _MIN_SYSTEM_GAP_FRAC))
        gap_threshold = max(perc_thresh, min_abs_gap)
        separators: list[tuple[int, int]] = []
        for s, e in whitespace_runs:
            if s == 0 or e == len(whitespace):
                continue
            gap_size = e - s
            if gap_size >= gap_threshold and not self._has_full_barline(im, s, e):
                separators.append((s, e))
        if not separators:
            largest = max(((e - s, (s, e)) for s, e in whitespace_runs if s != 0 and e != len(whitespace)), default=None)
            if largest:
                separators = [largest[1]]
        segments: list[tuple[int, int]] = []
        prev_end = 0
        for s, e in separators:
            seg = (prev_end, s)
            if (seg[1] - seg[0]) >= _MIN_SYSTEM_HEIGHT:
                segments.append(seg)
            prev_end = e
        if (h - prev_end) >= _MIN_SYSTEM_HEIGHT:
            segments.append((prev_end, h))
        if not segments:
            segments = [(0, h)]
        return segments

    def _crop_redundant_headers(self, slices: list[Image.Image]) -> list[Image.Image]:
        processed: list[Image.Image] = []
        prev_headers: Optional[tuple[Optional[np.ndarray], Optional[np.ndarray]]] = None
        if self.debug:
            print("Cropping redundant headers...")

        def _threshold_and_staff(mask_source: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
            thr = self._otsu_threshold(mask_source)
            thr = min(250, max(100, int(thr * 1.05)))
            mask = mask_source >= thr
            staff_space = self._estimate_staff_space(mask)
            row_density = mask.mean(axis=1)
            staff_rows = row_density >= max(float(np.percentile(row_density, 99) * 0.7), 0.12)
            staff_rows = np.logical_or.reduce([staff_rows, np.roll(staff_rows, 1), np.roll(staff_rows, -1)])
            return mask, staff_rows, staff_space

        def _prep_patch(gray_half: np.ndarray, _global_mask: np.ndarray, _staff_rows: np.ndarray) -> Optional[np.ndarray]:
            if gray_half.size == 0:
                return None
            h = gray_half.shape[0]
            thr = self._otsu_threshold(gray_half)
            thr = min(250, max(100, int(thr * 1.05)))
            mask_local = gray_half >= thr
            row_density = mask_local.mean(axis=1) if h > 0 else np.array([0.0])
            if row_density.size:
                thr_val = float(max(float(np.percentile(row_density, 99)) * 0.7, 0.12))
                staff_rows_local = row_density >= thr_val
            else:
                staff_rows_local = np.zeros((h,), dtype=bool)
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
            patch = mask_wo_staff[miny : maxy + 1, minx : maxx + 1]
            patch_u8 = (patch.astype(np.uint8)) * 255
            return patch_u8

        def _compare_patches(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
            if a is None or b is None:
                return 0.0
            target_w = 120

            def _resize_keep_w(img: np.ndarray) -> Optional[np.ndarray]:
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

            def _pad_h(img: np.ndarray) -> np.ndarray:
                pad = hmax - img.shape[0]
                if pad <= 0:
                    return img
                return np.pad(img, ((0, pad), (0, 0)), mode="constant", constant_values=0)

            a_p = _pad_h(a_r)
            b_p = _pad_h(b_r)
            try:
                return float(ssim(a_p, b_p))
            except (ValueError, RuntimeError):
                a_f = (a_p.astype(np.float32) / 255.0).ravel()
                b_f = (b_p.astype(np.float32) / 255.0).ravel()
                denom = float(np.linalg.norm(a_f)) * float(np.linalg.norm(b_f))
                return float((a_f @ b_f) / denom) if denom > 0 else 0.0

        cropped = 0
        for _, im in enumerate(slices):
            arr_gray = np.array(ImageOps.grayscale(im))
            h, w = arr_gray.shape
            if w <= 0 or h <= 0:
                processed.append(im)
                continue
            mask_full, staff_rows_full, staff_space = _threshold_and_staff(arr_gray)
            try:
                xs_local = self._detect_notehead_xs_on_image(im)
                first_note_x: Optional[int] = min(xs_local) if xs_local else None
            except (ValueError, RuntimeError, cv2.error):
                first_note_x = None
            if first_note_x is None:
                header_end_x = max(12, int(round(w * 0.075)))
            else:
                pad = max(8, int(round(0.6 * max(6, staff_space))))
                header_end_x = max(8, min(w - 8, int(first_note_x - pad)))
                if header_end_x < 10:
                    header_end_x = max(12, int(round(w * 0.06)))
            header_roi = arr_gray[:, :header_end_x]
            mid = h // 2
            treble_half = header_roi[:mid, :] if mid > 0 else header_roi
            bass_half = header_roi[mid:, :] if mid > 0 else None
            treble_patch = _prep_patch(
                treble_half,
                mask_full[:mid, :header_end_x] if mid > 0 else mask_full[:, :header_end_x],
                staff_rows_full[:mid] if mid > 0 else staff_rows_full,
            )
            bass_patch = (
                _prep_patch(
                    bass_half,
                    mask_full[mid:, :header_end_x]
                    if (mid > 0 and bass_half is not None)
                    else mask_full[:, :header_end_x],
                    staff_rows_full[mid:] if mid > 0 else staff_rows_full,
                )
                if bass_half is not None
                else None
            )
            crop_needed = False
            if prev_headers is not None:
                prev_treble, prev_bass = prev_headers
                score_treble = _compare_patches(prev_treble, treble_patch)
                score_bass = _compare_patches(prev_bass, bass_patch) if (prev_bass is not None and bass_patch is not None) else score_treble
                if score_treble >= 0.86 and score_bass >= 0.86:
                    crop_needed = True
            if crop_needed:
                im = im.crop((header_end_x, 0, w, h))
                cropped += 1
            prev_headers = (treble_patch, bass_patch)
            processed.append(im)
        if self.debug:
            print(f"Cropped {cropped} redundant headers.")
        return processed

    def _slice_page_into_system_images(self, page_path: str) -> list[Image.Image]:
        im = Image.open(page_path).convert("RGBA")
        bbox = ImageOps.invert(im.convert("L")).getbbox()
        if bbox is not None:
            l, _, r, _ = map(int, bbox)
            im = im.crop((l, 0, r, im.height))
        systems: list[Image.Image] = []
        segs = self._find_system_slices(im)
        page_h = im.height
        first_in_page = True
        for top, bottom in segs:
            seg_h = max(1, bottom - top)
            pad_est = int(seg_h * _PADDING_FRAC)
            pad = min(_MAX_PADDING, max(_MIN_PADDING, pad_est))
            t = max(0, top - pad)
            b = min(page_h, bottom + pad)
            crop = im.crop((0, t, im.width, b))
            crop2, _did = self._maybe_crop_leading_content(crop)
            recolored = self._recolor_to_dark_theme(crop2)
            if first_in_page:
                recolored = self._remove_tempo_marking(recolored)
                first_in_page = False
            systems.append(recolored)
        systems = self._crop_redundant_headers(systems)
        return systems

    @staticmethod
    def _otsu_threshold(gray_arr: np.ndarray) -> int:
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
    def _find_runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
        if mask_1d.size == 0:
            return []
        diffs = np.diff(mask_1d.astype(np.int8), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        return list(zip(starts, ends))

    def _estimate_staff_space(self, mask: np.ndarray) -> int:
        h, _w = mask.shape
        row_density = mask.mean(axis=1)
        thr = max(float(np.percentile(row_density, 99) * 0.7), 0.12)
        staff_rows = row_density >= thr
        staff_rows = np.logical_or.reduce([staff_rows, np.roll(staff_rows, 1), np.roll(staff_rows, -1)])
        runs = self._find_runs(staff_rows)
        centers = [int((s + e) // 2) for s, e in runs if (e - s) <= max(6, h // 100)]
        if len(centers) < 2:
            return max(6, int(round(h * 0.025)))
        diffs = np.diff(sorted(centers))
        diffs = diffs[(diffs >= 3) & (diffs <= h // 6)]
        if diffs.size == 0:
            return max(6, int(round(h * 0.025)))
        median_val = int(round(float(np.median(diffs))))
        return max(5, median_val)

    @staticmethod
    def _suppress_vertical_stems(mask: np.ndarray, staff_space: int) -> np.ndarray:
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

    @staticmethod
    def _connected_components(mask: np.ndarray):
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        comps = []
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
                        if cy < miny:
                            miny = cy
                        if cy > maxy:
                            maxy = cy
                        if cx < minx:
                            minx = cx
                        if cx > maxx:
                            maxx = cx
                        if cy > 0 and mask[cy - 1, cx] and not visited[cy - 1, cx]:
                            visited[cy - 1, cx] = True
                            q.append((cy - 1, cx))
                        if cy + 1 < h and mask[cy + 1, cx] and not visited[cy + 1, cx]:
                            visited[cy + 1, cx] = True
                            q.append((cy + 1, cx))
                        if cx > 0 and mask[cy, cx - 1] and not visited[cy, cx - 1]:
                            visited[cy, cx - 1] = True
                            q.append((cy, cx - 1))
                        if cx + 1 < w and mask[cy, cx + 1] and not visited[cy, cx + 1]:
                            visited[cy, cx + 1] = True
                            q.append((cy, cx + 1))
                    if area > 0:
                        comps.append(
                            {
                                "minx": minx,
                                "maxx": maxx,
                                "miny": miny,
                                "maxy": maxy,
                                "area": area,
                                "cx": sumx / area,
                                "cy": sumy / area,
                            }
                        )
        return comps

    def _detect_notehead_xs_on_image(self, im: Image.Image) -> list[int]:
        if im.mode != "L":
            gray = ImageOps.grayscale(im)
        else:
            gray = im
        arr = np.asarray(gray)
        thr = self._otsu_threshold(arr)
        thr = min(250, max(100, int(thr * 1.05)))
        mask = arr >= thr
        h, w = mask.shape
        if h < 10 or w < 10:
            return []
        staff_space = self._estimate_staff_space(mask)
        row_density = mask.mean(axis=1)
        row_thr = float(max(float(np.percentile(row_density, 99) * 0.7), 0.12))
        staff_rows = row_density >= row_thr
        staff_rows = np.logical_or.reduce([staff_rows, np.roll(staff_rows, 1), np.roll(staff_rows, -1)])
        mask_wo_staff = mask.copy()
        mask_wo_staff[staff_rows, :] = False
        mask_no_stems = self._suppress_vertical_stems(mask_wo_staff, staff_space)
        _ = mask_no_stems.mean(axis=0)
        comps = self._connected_components(mask_no_stems)
        if not comps:
            return []
        try:
            bar_x = self._find_first_barline_x(im)
        except Exception:
            bar_x = None
        if bar_x is not None and bar_x > 2:
            ignore_left_until = max(8, int(bar_x + max(6, int(0.8 * staff_space))))
        else:
            ignore_left_until = max(12, int(round(w * 0.075)))
        px = np.asarray(gray)
        dark_density = (px < _DARK_THRESH).mean(axis=1)
        y_lo = int(h * 0.20)
        y_hi = int(h * 0.88)
        gap_mask = np.zeros(h, dtype=bool)
        if y_hi > y_lo:
            band = dark_density.copy()
            thr_gap = float(min(0.06, np.percentile(band[y_lo:y_hi], 15))) if (y_hi - y_lo) > 5 else 0.04
            gap_mask[y_lo:y_hi] = band[y_lo:y_hi] <= thr_gap
        runs = self._find_runs(gap_mask)
        treble_limit_y: Optional[int] = None
        if runs:
            longest = max(runs, key=lambda se: (se[1] - se[0]))
            run_len = longest[1] - longest[0]
            min_gap = max(8, int(0.35 * max(6, staff_space)))
            if run_len >= min_gap:
                treble_limit_y = int((longest[0] + longest[1]) // 2)

        s = max(6, staff_space)
        min_area = int(0.35 * s * s)
        max_area = int(7.0 * s * s)
        min_side = max(3, int(0.45 * s))
        xs: list[int] = []
        for c in comps:
            if c["maxx"] < ignore_left_until:
                continue
            if treble_limit_y is not None and c["cy"] >= treble_limit_y:
                continue
            wbb = c["maxx"] - c["minx"] + 1
            hbb = c["maxy"] - c["miny"] + 1
            area = c["area"]
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
            xs.append(int(round(c["cx"])))
        xs = sorted(set(xs))
        pruned: list[int] = []
        min_sep = max(4, int(round(0.6 * s)))
        for x0 in xs:
            if not pruned or (x0 - pruned[-1]) >= min_sep:
                pruned.append(x0)
        return pruned

    def _prepare_strip(self) -> None:
        tmpdir = Path(tempfile.mkdtemp(prefix="sheetstrip_"))
        try:
            files, score = self._export_pages_png(tmpdir)
        except (OSError, FileNotFoundError, ValueError, RuntimeError, KeyError, AttributeError) as e:
            print("SheetMusicRenderer: failed to export pages via MuseScore:", e)
            return
        cache_dir = self._get_cache_dir()
        cache_key = self._midi_cache_key()
        systems_cache_dir = cache_dir / cache_key / f"systems_{_SPLIT_CACHE_VERSION}"
        cached_system_paths = sorted(glob.glob(str(systems_cache_dir / "sys_*.png"))) if systems_cache_dir.exists() else []
        all_system_images: list[Image.Image] = []
        page_system_counts: list[int] = []
        if cached_system_paths:
            try:
                for p in cached_system_paths:
                    im = Image.open(p).convert("RGBA")
                    all_system_images.append(im)
                if self.debug:
                    print(f"Loaded {len(all_system_images)} split systems from cache")
            except (OSError, ValueError, RuntimeError) as e:
                if self.debug:
                    print("Failed to load cached systems, will resplit:", e)
                all_system_images = []
        if not all_system_images:
            for f in files:
                try:
                    systems = self._slice_page_into_system_images(f)
                    if self.debug:
                        print(f"page {f} -> {len(systems)} systems")
                    page_system_counts.append(len(systems))
                    all_system_images.extend(systems)
                except (OSError, ValueError, RuntimeError, cv2.error) as e:
                    if self.debug:
                        print("failed slicing page", f, ":", e)
                    fallback_im = Image.open(f).convert("RGBA")
                    all_system_images.append(self._recolor_to_dark_theme(fallback_im))
                    page_system_counts.append(1)
            try:
                systems_cache_dir.mkdir(parents=True, exist_ok=True)
                for idx, im in enumerate(all_system_images):
                    out_path = systems_cache_dir / f"sys_{idx:04d}.png"
                    im.save(out_path)
                if self.debug:
                    print(f"Saved {len(all_system_images)} split systems to cache: {systems_cache_dir}")
            except (OSError, ValueError) as e:
                if self.debug:
                    print("Warning: failed to save system cache:", e)
        if not all_system_images:
            raise RuntimeError("No system images were produced from MuseScore pages.")
        widths = [im.width for im in all_system_images]
        heights = [im.height for im in all_system_images]
        total_width = sum(widths)
        max_height = max(heights)
        strip = Image.new("RGBA", (total_width, max_height), _BG_RGBA)
        x = 0
        self.system_boxes = []
        per_system_note_xs: list[tuple[int, list[int]]] = []
        for i, im in enumerate(all_system_images):
            y = (max_height - im.height) // 2
            strip.paste(im, (x, y), im)
            self.system_boxes.append((x, im.width))
            try:
                xs_local = self._detect_notehead_xs_on_image(im)
            except (ValueError, RuntimeError, cv2.error) as e:
                if self.debug:
                    print("notehead detection failed on system", i, ":", e)
                xs_local = []
            per_system_note_xs.append((x, xs_local))
            x += im.width
        scale = self.strip_height / max_height
        new_w = max(1, int(total_width * scale))
        strip_resized = strip.resize((new_w, self.strip_height), RESAMPLE_LANCZOS).convert("RGBA")
        data = strip_resized.tobytes()
        self.full_surface = pygame.image.fromstring(data, strip_resized.size, "RGBA").convert_alpha()
        self.full_width = strip_resized.size[0]
        if self.debug:
            print(f"Built continuous strip: {len(all_system_images)} systems -> width={self.full_width}px at height={self.strip_height}px")
        orig_full_width = sum(w for _, w in self.system_boxes) or 1
        sx = self.full_width / orig_full_width
        self.notehead_xs = []
        self._notehead_xs_per_system = []
        for sys_x, xs_local in per_system_note_xs:
            scaled = [int((sys_x + lx) * sx) for lx in xs_local]
            self._notehead_xs_per_system.append(scaled)
            self.notehead_xs.extend(scaled)
        self.notehead_xs.sort()
        if self.debug:
            print(f"Detected {len(self.notehead_xs)} visual noteheads across strip")
        try:
            self._build_note_mapping_from_score(score)
        except (ValueError, RuntimeError, AttributeError) as e:
            if self.debug:
                print("Warning: building note mapping failed:", e)
            self.note_to_xs = {}
        try:
            for f in files:
                os.remove(f)
            shutil.rmtree(tmpdir)
        except (OSError, FileNotFoundError):
            pass

    def _build_note_mapping_from_score(self, score: m21stream.Score) -> None:
        notes: list[tuple[float, Optional[int]]] = []
        for n in score.recurse().getElementsByClass(m21note.Note):
            try:
                on = float(n.getOffsetInHierarchy(score))
            except (AttributeError, TypeError):
                on = float(n.offset if hasattr(n, "offset") else 0.0)
            midi = n.pitch.midi if n.pitch is not None else None
            notes.append((on, midi))
        if not notes:
            return
        notes.sort(key=lambda x: x[0])

        treble_notes = [(on, m) for (on, m) in notes if (m is not None and int(m) >= 60)]
        used_notes = treble_notes if treble_notes else notes

        positions = list(self.notehead_xs) if self.notehead_xs else []
        self.note_to_xs = {}
        for ((_, midi), x) in zip(used_notes, positions):
            if midi is None:
                continue
            self.note_to_xs.setdefault(int(midi), []).append(x)
        if self.debug:
            counts = {m: len(xs) for m, xs in self.note_to_xs.items()}
            print("Built mapping: midi->count:", counts)

    def draw(self, screen: pygame.Surface, y: int, progress: float, highlight_pitches: Optional[Iterable[int]] = None) -> None:
        if self.full_surface is None:
            return
        p = min(max(progress, 0.0), 1.0)
        view_w = self.screen_width
        x_off = int(p * max(0, (self.full_width - view_w)))
        src_rect = pygame.Rect(x_off, 0, view_w, self.strip_height)
        screen.blit(self.full_surface, (0, y), src_rect)
        play_x = int(view_w * 0.45)
        pygame.draw.line(screen, (0, 200, 255), (play_x, y), (play_x, y + self.strip_height), 3)
        if self.debug and self.notehead_xs:
            for gx in self.notehead_xs:
                sx = gx - x_off
                if -4 <= sx <= view_w + 4:
                    pygame.draw.line(screen, (90, 255, 120), (int(sx), y), (int(sx), y + self.strip_height), 1)
        if highlight_pitches:
            to_draw: list[tuple[int, tuple[int, int, int]]] = []
            for midi in highlight_pitches:
                xs = self.note_to_xs.get(int(midi), [])
                if not xs:
                    continue
                abs_play_x = x_off + play_x
                best = min(xs, key=lambda xv: abs(xv - abs_play_x))
                screen_x = best - x_off
                color = (0, 200, 255) if int(midi) >= 60 else (255, 140, 60)
                to_draw.append((screen_x, color))
            for sx, color in to_draw:
                if -24 <= sx <= view_w + 24:
                    r = 12
                    pygame.draw.circle(screen, color, (int(sx), int(y + self.strip_height * 0.5)), r)
                    surf = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
                    pygame.draw.circle(surf, (*color, 90), (r * 2, r * 2), int(r * 1.9))
                    screen.blit(surf, (int(sx - r * 2), int(y + self.strip_height * 0.5 - r * 2)))
        pygame.draw.rect(screen, (20, 20, 20), (0, y, view_w, self.strip_height), 2)
