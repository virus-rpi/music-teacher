from pathlib import Path
import tempfile
import glob
import os
import shutil

import pygame
from PIL import Image, ImageOps
from music21 import converter, environment, note as m21note, stream as m21stream

# Pillow resampling constant compatibility (Pillow 10 moved LANCZOS under Image.Resampling)
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback for older/newer Pillow
    _resampling = getattr(Image, "Resampling", Image)
    RESAMPLE_LANCZOS = getattr(_resampling, "LANCZOS", getattr(_resampling, "BICUBIC", getattr(Image, "NEAREST", 0)))

# tune these to detect whitespace between systems
_WHITESPACE_THRESHOLD = 0.90  # fraction of white pixels in a row to be considered whitespace
_MIN_SYSTEM_HEIGHT = 18       # ignore tiny slices
_MORPH_BLUR = 3               # smooth projection to merge near whitespace gaps
# New heuristics to avoid splitting treble/bass apart
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
_FP_WIDTH_FRAC = 0.22         # width fraction of left region used for fingerprint (before barline)
_FP_SIZE = 24                 # downscaled square for fingerprint
_FP_DIFF_THRESH = 0.14        # normalized diff threshold to consider two left regions the same

class SheetMusicRenderer:
    def __init__(self, midi_path, screen_width, height=260, debug=False):
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

        # NEW: for detection of first barline and cropping of leading content
        self._prev_clef_key_fp = None  # image-based fingerprint of left-of-barline region

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
            current = None

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
    def _get_cache_dir(self):
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

    # ------------------------------
    # Leading content cropping helpers
    # ------------------------------
    def _find_first_barline_x(self, im: Image.Image):
        """Return the x position (int) of the first strong vertical barline within the left region, or None."""
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
                out = [0] * len(arr)
                for i in range(len(arr)):
                    lo = max(0, i - k)
                    hi = min(len(arr), i + k + 1)
                    s = 0
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

    def _left_region_fingerprint(self, im: Image.Image, bar_x: int):
        """Return a compact fingerprint (tuple of ints) of the left-of-barline region for similarity tests."""
        w, h = im.size
        if bar_x <= 0:
            return None
        fp_w = max(20, min(bar_x, int(w * _FP_WIDTH_FRAC)))
        box = (max(0, bar_x - fp_w), 0, bar_x, h)
        region = im.convert("L").crop(box)
        region = ImageOps.autocontrast(region)
        ds = region.resize((_FP_SIZE, _FP_SIZE))
        data = list(ds.getdata())  # 0..255 ints
        return tuple(data)

    @staticmethod
    def _fp_diff(a, b):
        if not a or not b or len(a) != len(b):
            return 1.0
        # normalized mean absolute difference (0..1)
        diffs = 0
        n = len(a)
        for i in range(n):
            diffs += abs(a[i] - b[i])
        return (diffs / 255.0) / n

    def _maybe_crop_leading_content(self, im: Image.Image):
        """
        Crop leading instrument/clef/key up to first barline.
        - Always crop on the very first encountered system (to remove instrument names).
        - For subsequent systems, crop only if the left-of-barline fingerprint matches the previous one (clef/key unchanged).
        Returns (cropped_image, did_crop: bool).
        """
        bar_x = self._find_first_barline_x(im)
        if bar_x is None or bar_x <= 2:
            return im, False
        fp = self._left_region_fingerprint(im, bar_x)
        do_crop = False
        if self._prev_clef_key_fp is None:
            # first system: remove instrument block and initial clef/key
            do_crop = True
        else:
            diff = self._fp_diff(fp, self._prev_clef_key_fp)
            do_crop = diff <= _FP_DIFF_THRESH
            if self.debug:
                print(f"fp diff={diff:.3f} -> {'crop' if do_crop else 'keep'}")
        # update previous fingerprint regardless (for detection of changes later)
        self._prev_clef_key_fp = fp
        if do_crop:
            return im.crop((bar_x, 0, im.width, im.height)), True
        return im, False

    # ------------------------------
    # System slicing and cropping
    # ------------------------------
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

        # Compute gap sizes (ignore top/bottom margins for classification)
        internal_gaps = []
        for s, e in whitespace_runs:
            if s == 0 or e == len(whitespace):
                continue
            internal_gaps.append(e - s)

        # If no internal gaps, treat whole page as one system
        if not internal_gaps:
            return [(0, h)]

        # Set dynamic gap threshold: use percentile of gap sizes plus a minimal absolute threshold
        gaps_sorted = sorted(internal_gaps)
        idx = max(0, min(len(gaps_sorted) - 1, int(len(gaps_sorted) * _GAP_PERCENTILE)))
        perc_thresh = gaps_sorted[idx]
        min_abs_gap = max(18, int(h * _MIN_SYSTEM_GAP_FRAC))
        gap_threshold = max(perc_thresh, min_abs_gap)

        # Select whitespace runs that qualify as inter-system separators
        separators = []  # (start, end)
        for s, e in whitespace_runs:
            if s == 0 or e == len(whitespace):
                continue
            if (e - s) >= gap_threshold:
                separators.append((s, e))

        # If still nothing classified, fallback to the largest one
        if not separators:
            largest = max(((e - s, (s, e)) for s, e in whitespace_runs if s != 0 and e != len(whitespace)), default=None)
            if largest:
                separators = [largest[1]]

        # Build segments between separators
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

        # If we found nothing (margins different), fallback to splitting into 1 system (full page)
        if not segments:
            segments = [(0, h)]

        return segments

    def _slice_page_into_system_images(self, page_path: str):
        im = Image.open(page_path).convert("RGBA")
        # Don’t crop too aggressively: keep outer whitespace
        # Just trim left/right margins a bit
        bbox = ImageOps.invert(im.convert("L")).getbbox()
        if bbox:
            # shrink horizontally only, keep vertical intact
            l, t, r, b = bbox
            im = im.crop((l, 0, r, im.height))

        systems = []
        segs = self._find_system_slices(im)
        page_h = im.height
        for (top, bottom) in segs:
            seg_h = max(1, bottom - top)
            # dynamic padding keeps both staves intact and avoids cutting ledger lines
            pad_est = int(seg_h * _PADDING_FRAC)
            pad = min(_MAX_PADDING, max(_MIN_PADDING, pad_est))
            t = max(0, top - pad)
            b = min(page_h, bottom + pad)
            crop = im.crop((0, t, im.width, b))
            # NEW: optionally crop leading instrument/clef/key area per system
            crop2, did = self._maybe_crop_leading_content(crop)
            systems.append(crop2)
        return systems

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

        # split every page into systems
        all_system_images = []
        page_system_counts = []
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
                all_system_images.append(Image.open(f).convert("RGBA"))
                page_system_counts.append(1)

        if not all_system_images:
            raise RuntimeError("No system images were produced from MuseScore pages.")

        # Now concatenate systems horizontally in page order to produce a continuous strip.
        widths = [im.width for im in all_system_images]
        heights = [im.height for im in all_system_images]
        total_width = sum(widths)
        max_height = max(heights)

        strip = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 255))
        x = 0
        self.system_boxes = []
        for i, im in enumerate(all_system_images):
            # vertically center each system in the strip
            y = (max_height - im.height) // 2
            strip.paste(im, (x, y), im)
            self.system_boxes.append((x, im.width))
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
            print(f"Built continuous strip: {len(all_system_images)} systems -> width={self.full_width}px")

        # Build an approximate mapping from score notes -> x positions along the full strip
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
        img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        # convert to pygame surface
        data = img.tobytes()
        self.full_surface = pygame.image.fromstring(data, (w, h), "RGBA").convert_alpha()
        self.full_width = w
        self.note_to_xs = {}

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
        offsets = [o for o, _ in notes]
        # total number of systems
        system_count = max(1, len(self.system_boxes))
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


# Example usage if run as a script (not executed when imported)
if __name__ == "__main__":
    import sys, time
    pygame.init()
    w, h = 1200, 300
    screen = pygame.display.set_mode((w, h))
    midi = sys.argv[1] if len(sys.argv) > 1 else None
    if not midi:
        print("usage: python sheet_music.py path/to/file.mid")
        sys.exit(1)
    r = SheetMusicRenderer(midi, w, height=200, debug=True)
    clock = pygame.time.Clock()
    prog = 0.0
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
        screen.fill((30,30,30))
        r.draw(screen, 20, prog, highlight_pitches=[60,64])
        pygame.display.flip()
        prog += 0.002
        if prog > 1.0:
            prog = 0.0
        clock.tick(60)
    pygame.quit()
