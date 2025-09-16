from pathlib import Path
import math
import pygame
from PIL import Image, ImageOps
from sheet_music_renderer import midi_to_svg
from typing import Optional, Iterable
import cairosvg
import json

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
    def __init__(self, midi_path: str | Path, screen_width: int, height: int = 260, debug: bool = False) -> None:
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
        # index of the currently focused note (for centering). If None, use progress-based scrolling.
        self._current_note_idx: int = 0
        # Smooth scrolling state: current view offset (float) and target offset (int)
        self._view_x_off: float = 0.0
        self._target_x_off: float = 0.0
        # time constant for exponential smoothing (seconds)
        self._scroll_tau: float = 0.08
        # smoothed screen play x (pixels within view) for the highlight rectangle/line
        self._screen_play_x: Optional[float] = None
        # time constant for smoothing the play-line/rect movement
        # make play overlay move faster (smaller tau = snappier)
        self._play_tau: float = 0.03
        # last tick used for smoothing
        try:
            self._last_time_ms: int = pygame.time.get_ticks()
        except Exception:
            self._last_time_ms = 0

        self._prepare_strip()

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

    def _prepare_strip(self) -> None:
        cache_dir = self._get_cache_dir()
        cache_key = self._midi_cache_key()
        cache_subdir = cache_dir / cache_key
        cache_subdir.mkdir(parents=True, exist_ok=True)

        strip_png = cache_subdir / "strip.png"
        svg_out = cache_subdir / f"{self.midi_path.stem}.svg"

        # Try to load cached note x positions if available (keep as floats for scaling)
        note_xs_cache = cache_subdir / "note_xs.json"
        if note_xs_cache.exists():
            try:
                with note_xs_cache.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                    # ensure floats; leave as floats for later scaling
                    self.notehead_xs = [float(x) for x in raw if x is not None]
            except Exception:
                self.notehead_xs = []

        # If we don't have a cached PNG or cached note positions, (re)render the SVG and convert
        if not strip_png.exists() or not self.notehead_xs:
            try:
                try:
                    res = midi_to_svg(str(self.midi_path), str(cache_subdir))
                except TypeError:
                    res = midi_to_svg(str(self.midi_path), str(cache_subdir), "mscore")
                # res is a list of x positions (floats)
                if isinstance(res, list):
                    # save raw floats to cache file
                    try:
                        with note_xs_cache.open("w", encoding="utf-8") as fh:
                            json.dump([float(x) for x in res], fh)
                    except Exception:
                        pass

                    self.notehead_xs = [float(x) for x in res]
            except Exception as e:
                print("SheetMusicRenderer: failed to render SVG via sheet_music_renderer:", e)
                return

            if not svg_out.exists():
                svgs = list(cache_subdir.glob(f"{self.midi_path.stem}*.svg"))
                if svgs:
                    svg_out = svgs[0]

            if not svg_out.exists():
                print("No SVG produced by sheet_music_renderer; aborting strip preparation.")
                return

            converted = False
            try:
                cairosvg.svg2png(url=str(svg_out), write_to=str(strip_png), dpi=600)
                converted = True
            except Exception:
                pass

            if not converted:
                import subprocess
                try:
                    subprocess.run(["rsvg-convert", str(svg_out), "-o", str(strip_png)], check=True)
                    converted = True
                except Exception:
                    try:
                        subprocess.run(["inkscape", str(svg_out), "--export-type=png", "--export-filename", str(strip_png)], check=True)
                        converted = True
                    except Exception:
                        converted = False

            if not converted:
                print("Failed to convert SVG to PNG. Install 'cairosvg' Python package or 'rsvg-convert' / 'inkscape' command-line tools.")
                return

        try:
            im = Image.open(str(strip_png)).convert("RGBA")
        except Exception as e:
            print("Failed to open generated PNG:", e)
            return

        # Crop transparent/empty borders if present and record left crop
        left_crop = 0
        try:
            bbox = ImageOps.invert(im.convert("L")).getbbox()
            if bbox is not None:
                l, _, r, _ = map(int, bbox)
                if r - l > 0:
                    left_crop = l
                    im = im.crop((l, 0, r, im.height))
        except Exception:
            left_crop = 0

        cropped_w = im.width

        try:
            recolored = self._recolor_to_dark_theme(im)
        except Exception:
            recolored = im

        try:
            scale = self.strip_height / float(max(1, recolored.height))
            new_w = max(1, int(recolored.width * scale))
            strip_resized = recolored.resize((new_w, self.strip_height), RESAMPLE_LANCZOS).convert("RGBA")
        except Exception as e:
            print("Failed to resize strip image:", e)
            return

        data = strip_resized.tobytes()
        try:
            self.full_surface = pygame.image.fromstring(data, strip_resized.size, "RGBA").convert_alpha()
        except Exception as e:
            print("Failed to create pygame surface from strip image:", e)
            return

        self.full_width = strip_resized.size[0]

        # If we loaded SVG-derived note x positions, adjust them for crop & resize into final pixel coords
        if self.notehead_xs:
            try:
                final_scale = float(strip_resized.width) / float(max(1, cropped_w))
                pxs = []
                for x in self.notehead_xs:
                    # subtract left crop (positions were relative to original SVG/PNG before cropping)
                    x_adj = float(x) - float(left_crop)
                    if x_adj < 0 or x_adj > float(cropped_w):
                        continue
                    pxs.append(int(round(x_adj * final_scale)))
                pxs = sorted(set(pxs))
                self.notehead_xs = pxs
            except Exception as e:
                if self.debug:
                    print("Failed to scale cached note x positions:", e)
                self.notehead_xs = []
        else:
            try:
                self.notehead_xs.sort()
            except Exception as e:
                print("Notehead detection failed:", e)
                self.notehead_xs = []

        if self.debug:
            print(f"Built strip from SVG: width={self.full_width}px, detected {len(self.notehead_xs)} visual noteheads")

    def draw(self, screen: pygame.Surface, y: int, progress: float, highlight_pitches: Optional[Iterable[int]] = None) -> None:
        if self.full_surface is None:
            return
        view_w = self.screen_width

        # Compute desired target offset for this frame (based on current note index or progress)
        max_off = max(0, self.full_width - view_w)
        if self.notehead_xs and 0 <= int(self._current_note_idx) < len(self.notehead_xs):
            play_x = int(view_w * 0.45)
            target_x = int(self.notehead_xs[int(self._current_note_idx)])
            desired = int(target_x - play_x)
            if desired < 0:
                desired = 0
            if desired > max_off:
                desired = max_off
            self._target_x_off = float(desired)
        else:
            p = min(max(progress, 0.0), 1.0)
            self._target_x_off = float(int(p * max_off))

        # update smoothing based on elapsed time
        now_ms = pygame.time.get_ticks()
        dt = max(0.0, (now_ms - getattr(self, '_last_time_ms', now_ms)) / 1000.0)
        self._last_time_ms = now_ms
        if dt > 0.0:
            # exponential smoothing factor
            alpha = 1.0 - math.exp(-dt / max(1e-6, self._scroll_tau))
            self._view_x_off += (self._target_x_off - self._view_x_off) * alpha
            # snap if very close
            if abs(self._target_x_off - self._view_x_off) < 0.5:
                self._view_x_off = float(self._target_x_off)

        x_off = int(round(max(0.0, min(self._view_x_off, float(max_off)))))
        src_rect = pygame.Rect(x_off, 0, view_w, self.strip_height)
        screen.blit(self.full_surface, (0, y), src_rect)
        # Determine desired screen play position (relative to view) and smooth it
        if self.notehead_xs and 0 <= int(self._current_note_idx) < len(self.notehead_xs):
            desired_screen_play = float(int(self.notehead_xs[int(self._current_note_idx)]) - x_off)
        else:
            desired_screen_play = float(int(view_w * 0.45))

        # initialize smoothed value if needed
        if self._screen_play_x is None:
            self._screen_play_x = desired_screen_play

        # smooth the play x using the same dt as the view smoothing (frame-time)
        dt_play = dt if dt > 0.0 else (1.0 / 60.0)
        alpha_play = 1.0 - math.exp(-dt_play / max(1e-6, self._play_tau))
        self._screen_play_x += (desired_screen_play - self._screen_play_x) * alpha_play
        # snap if very close to avoid tiny subpixel jitter
        if abs(desired_screen_play - self._screen_play_x) < 0.5:
            self._screen_play_x = float(desired_screen_play)

        screen_play_x = float(self._screen_play_x)

        # Draw semi-transparent highlight rectangle (about 30px wide) and a semi-transparent vertical line
        overlay = pygame.Surface((view_w, self.strip_height), pygame.SRCALPHA)
        rect_w = 30
        rect_h = self.strip_height
        rect_x = int(round(screen_play_x - rect_w // 2))
        rect_y = 0
        # rectangle color (semi-transparent cyan)
        rect_color = (0, 200, 255, 128)
        pygame.draw.rect(overlay, rect_color, (rect_x, rect_y, rect_w, rect_h))
        # vertical line (slightly more opaque)
        line_color = (0, 200, 255, 200)
        pygame.draw.line(overlay, line_color, (int(round(screen_play_x)), 0), (int(round(screen_play_x)), rect_h), 3)
        # blit overlay onto screen at the strip position
        screen.blit(overlay, (0, y))

        if self.debug:
            # draw a green vertical line at every note position
            positions = set()
            try:
                positions.update(int(x) for x in self.notehead_xs)
            except Exception:
                pass
            try:
                for xs in self.note_to_xs.values():
                    for xv in xs:
                        positions.add(int(xv))
            except Exception:
                pass
            for gx in sorted(positions):
                sx = gx - x_off
                if -4 <= sx <= view_w + 4:
                    pygame.draw.line(screen, (90, 255, 120), (int(sx), y), (int(sx), y + self.strip_height), 1)
        if highlight_pitches:
            to_draw: list[tuple[int, tuple[int, int, int]]] = []
            # absolute play-line X in strip coordinates (used to choose nearest visual note)
            abs_play_x = x_off + int(round(screen_play_x))
            for midi in highlight_pitches:
                xs = self.note_to_xs.get(int(midi), [])
                if not xs:
                    continue
                # Use the actual absolute play-line X when selecting which visual note to highlight
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

    def advance_note(self) -> None:
        """Advance the internal current note index by one (clamped)."""
        try:
            if not self.notehead_xs:
                return
            self._current_note_idx = min(len(self.notehead_xs) - 1, int(self._current_note_idx) + 1)
            # compute and set target offset immediately so smooth scroll begins
            view_w = int(self.screen_width)
            play_x = int(view_w * 0.45)
            max_off = max(0, self.full_width - view_w)
            tgt = int(self.notehead_xs[int(self._current_note_idx)]) - play_x
            tgt = max(0, min(tgt, max_off))
            self._target_x_off = float(tgt)
            if self.debug:
                print(f"Advanced current note index to {self._current_note_idx}, target offset {self._target_x_off}")
        except Exception:
            pass

    def reset_note_index(self) -> None:
        """Reset the current note index to the start."""
        self._current_note_idx = 0
        # jump view to start immediately
        self._view_x_off = 0.0
        self._target_x_off = 0.0

    def seek_to_progress(self, progress: float) -> None:
        """Seek the visible strip to the given progress (0.0-1.0). If visual noteheads exist, choose nearest visual note index."""
        if self.full_surface is None:
            return
        view_w = int(self.screen_width)
        max_off = max(0, self.full_width - view_w)
        p = max(0.0, min(1.0, float(progress)))
        # Prefer snapping to nearest detected visual note
        if self.notehead_xs:
            idx = int(round(p * (len(self.notehead_xs) - 1)))
            idx = max(0, min(idx, len(self.notehead_xs) - 1))
            self._current_note_idx = idx
            play_x = int(view_w * 0.45)
            tgt = int(self.notehead_xs[int(self._current_note_idx)]) - play_x
            tgt = max(0, min(tgt, max_off))
            self._target_x_off = float(tgt)
            # also set view immediately so user sees immediate jump
            self._view_x_off = float(self._target_x_off)
            self._screen_play_x = float(int(self.notehead_xs[int(self._current_note_idx)]) - int(self._view_x_off))
        else:
            tgt = int(p * max_off)
            self._target_x_off = float(tgt)
            self._view_x_off = float(self._target_x_off)
            self._screen_play_x = float(int(view_w * 0.45))
