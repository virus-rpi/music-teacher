from pathlib import Path
import math
import pygame
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError
from sheet_music_renderer import midi_to_svg
from typing import Optional
import cairosvg
import json
import xml.etree.ElementTree as ElementTree

_resampling = getattr(Image, "Resampling", Image)
RESAMPLE_LANCZOS = getattr(_resampling, "LANCZOS", getattr(_resampling, "BICUBIC", getattr(Image, "NEAREST", 0)))
_CAIROSVG_DPI = 1200

class SheetMusicRenderer:
    def __init__(self, midi_path: str | Path, screen_width: int, height: int = 260, debug: bool = False) -> None:
        self.midi_path = Path(midi_path)
        self.screen_width = int(screen_width)
        self.strip_height = int(height)
        self.debug = debug

        self.full_surface: Optional[pygame.Surface] = None
        self.full_width: int = 0
        self.notehead_xs: list[int] = []
        self.measure_data: list = []
        self._current_note_idx: int = 0
        self.view_x_off: float = 0.0
        self._target_x_off: float = 0.0
        self._scroll_tau: float = 0.08
        self._screen_play_x: Optional[float] = None
        self._play_tau: float = 0.03
        try:
            self._last_time_ms: int = pygame.time.get_ticks()
        except (pygame.error, AttributeError):
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
        except (OSError, AttributeError):
            mtime = 0
            size = 0
        key = f"{self.midi_path.name}_{mtime}_{size}"
        return key

    def _load_cache_data(self, cache_subdir: Path) -> None:
        note_xs_cache = cache_subdir / "note_xs.json"
        measure_data_cache = cache_subdir / "measure_data.json"

        if note_xs_cache.exists():
            try:
                with note_xs_cache.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                    self.notehead_xs = [float(x) for x in raw if x is not None]
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                self.notehead_xs = []
        
        if measure_data_cache.exists():
            try:
                with measure_data_cache.open("r", encoding="utf-8") as fh:
                    raw_measure_data = json.load(fh)
                    self.measure_data = []
                    for tup in raw_measure_data:
                        if tup is not None and len(tup) == 3:
                            sx, ex, n = tup
                            self.measure_data.append((float(sx) if sx is not None else None, float(ex) if ex is not None else None, int(n)))
                        else:
                            self.measure_data.append((None, None, 0))
            except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
                print(f"SheetMusicRenderer: failed to load measure data cache: {e}")
                self.measure_data = []

    def _call_midi_to_svg(self, cache_subdir: Path, svg_out: Path) -> None:
        note_xs_cache = cache_subdir / "note_xs.json"
        measure_data_cache = cache_subdir / "measure_data.json"
        try:
            try:
                res_note_xs, res_measure_data = midi_to_svg(str(self.midi_path), str(cache_subdir))
            except TypeError:
                res_note_xs, res_measure_data = midi_to_svg(str(self.midi_path), str(cache_subdir), "mscore")
            
            self.measure_data = res_measure_data
            if isinstance(res_note_xs, list):
                try:
                    with note_xs_cache.open("w", encoding="utf-8") as fh:
                        json.dump([float(x) for x in res_note_xs], fh)
                except (OSError, TypeError, ValueError) as e:
                    print("SheetMusicRenderer: failed to save note x positions:", e)
                self.notehead_xs = [float(x) for x in res_note_xs]

            # Save measure data
            if isinstance(res_measure_data, list):
                try:
                    with measure_data_cache.open("w", encoding="utf-8") as fh:
                        # Save as list of (start_x, end_x, n_notes)
                        serializable_measure_data = []
                        for tup in res_measure_data:
                            if tup is not None and len(tup) == 3:
                                sx, ex, n = tup
                                serializable_measure_data.append([sx, ex, n])
                            else:
                                serializable_measure_data.append([None, None, 0])
                        json.dump(serializable_measure_data, fh)
                except (OSError, TypeError, ValueError) as e:
                    print(f"SheetMusicRenderer: failed to save measure data: {e}")

        except (RuntimeError, OSError, ValueError, TypeError) as e:
            print("SheetMusicRenderer: failed to render SVG via sheet_music_renderer:", e)
            return

        if not svg_out.exists():
            svgs = list(cache_subdir.glob(f"{self.midi_path.stem}*.svg"))
            if svgs:
                svg_out = svgs[0]

        if not svg_out.exists():
            print("No SVG produced by sheet_music_renderer; aborting strip preparation.")
            return

    @staticmethod
    def _convert_svg_to_png(svg_out: Path, strip_png: Path) -> bool:
        try:
            cairosvg.svg2png(url=str(svg_out), write_to=str(strip_png), dpi=_CAIROSVG_DPI, scale=2.0, negate_colors=True)
            return True
        except (RuntimeError, OSError, ValueError) as e:
            print("Failed to convert SVG to PNG: ", e)
            return False

    def _open_and_process_image(self, strip_png: Path, svg_path: Path) -> tuple[Optional[Image.Image], Optional[float], int, int]:
        """Open PNG, crop whitespace on the left, resize to strip_height.
        Returns (pil_image, svg_width_if_found_or_None, cropped_w, left_crop)
        """
        try:
            im = Image.open(str(strip_png)).convert("RGBA")
        except (FileNotFoundError, OSError, UnidentifiedImageError) as e:
            print("Failed to open generated PNG:", e)
            return None, None, 0, 0

        svg_width = None
        if svg_path.exists():
            try:
                tree = ElementTree.parse(str(svg_path))
                root = tree.getroot()
                w = root.get('width') or root.get('{http://www.w3.org/2000/svg}width')
                if w:
                    if isinstance(w, str) and w.endswith('px'):
                        w = w[:-2]
                    svg_width = float(w)
            except (ElementTree.ParseError, OSError, ValueError):
                svg_width = None

        left_crop = 0
        bbox = ImageOps.invert(im.convert("L")).getbbox()
        if bbox is not None:
            l, _, r, _ = map(int, bbox)
            if r - l > 0:
                left_crop = l
                im = im.crop((l, 0, r, im.height))

        cropped_w = im.width
        scale = self.strip_height / float(max(1, im.height))
        new_w = max(1, int(im.width * scale))
        strip_resized = im.resize((new_w, self.strip_height), RESAMPLE_LANCZOS).convert("RGBA")
        strip_resized = strip_resized.filter(ImageFilter.UnsharpMask(radius=0.5, percent=150, threshold=2))
        return strip_resized, svg_width, cropped_w, left_crop

    def _scale_cached_note_positions(self, svg_width, cropped_w, left_crop, strip_resized):
        if not self.notehead_xs:
            return
        try:
            if svg_width and svg_width > 0:
                pixels_per_svg_unit = float(cropped_w) / float(svg_width)
            else:
                pixels_per_svg_unit = 1.0

            final_scale = float(strip_resized.width) / float(max(1, cropped_w))
            pxs = []
            for x in self.notehead_xs:
                try:
                    x_px = float(x) * pixels_per_svg_unit
                except (TypeError, ValueError):
                    continue
                x_adj_px = x_px - float(left_crop)
                if x_adj_px < 0 or x_adj_px > float(cropped_w):
                    continue
                pxs.append(int(round(x_adj_px * final_scale)))
            pxs = sorted(set(pxs))
            self.notehead_xs = pxs

            # Scale measure_data start_x and end_x
            if self.measure_data:
                scaled_measure_data = []
                for tup in self.measure_data:
                    if tup is not None and len(tup) == 3:
                        sx, ex, n = tup
                        if sx is not None:
                            try:
                                sx_px = float(sx) * pixels_per_svg_unit
                                sx_adj_px = sx_px - float(left_crop)
                                if 0 <= sx_adj_px <= float(cropped_w):
                                    sx_scaled = int(round(sx_adj_px * final_scale))
                                else:
                                    sx_scaled = None
                            except Exception:
                                sx_scaled = None
                        else:
                            sx_scaled = None
                        if ex is not None:
                            try:
                                ex_px = float(ex) * pixels_per_svg_unit
                                ex_adj_px = ex_px - float(left_crop)
                                if 0 <= ex_adj_px <= float(cropped_w):
                                    ex_scaled = int(round(ex_adj_px * final_scale))
                                else:
                                    ex_scaled = None
                            except Exception:
                                ex_scaled = None
                        else:
                            ex_scaled = None
                        scaled_measure_data.append((sx_scaled, ex_scaled, int(n)))
                    else:
                        scaled_measure_data.append((None, None, 0))
                self.measure_data = scaled_measure_data
        except (TypeError, ValueError, ZeroDivisionError) as e:
            if self.debug:
                print("Failed to scale cached note x positions and measure data:", e)
            self.notehead_xs = []
            self.measure_data = []

    def _prepare_strip(self) -> None:
        cache_dir = self._get_cache_dir()
        cache_key = self._midi_cache_key()
        cache_subdir = cache_dir / cache_key
        cache_subdir.mkdir(parents=True, exist_ok=True)

        strip_png = cache_subdir / "strip.png"
        svg_out = cache_subdir / f"{self.midi_path.stem}.svg"
        note_xs_cache = cache_subdir / "note_xs.json"
        measure_data_cache = cache_subdir / "measure_data.json"

        self._load_cache_data(cache_subdir)

        if not strip_png.exists() or not self.notehead_xs or not self.measure_data:
            self._call_midi_to_svg(cache_subdir, svg_out)
            if not svg_out.exists():
                svgs = list(cache_subdir.glob(f"{self.midi_path.stem}*.svg"))
                if svgs:
                    svg_out = svgs[0]
            if not svg_out.exists():
                return

            ok = self._convert_svg_to_png(svg_out, strip_png)
            if not ok:
                return

        strip_resized, svg_width, cropped_w, left_crop = self._open_and_process_image(strip_png, svg_out)
        if strip_resized is None:
            return
        try:
            data = strip_resized.tobytes()
            self.full_surface = pygame.image.fromstring(data, strip_resized.size, "RGBA").convert_alpha()
        except (pygame.error, ValueError, TypeError) as e:
            print("Failed to create pygame surface from strip image:", e)
            return

        self.full_width = strip_resized.size[0]
        self._scale_cached_note_positions(svg_width=svg_width, cropped_w=cropped_w, left_crop=left_crop, strip_resized=strip_resized)

        if self.debug:
            print(f"Built strip from SVG: width={self.full_width}px, detected {len(self.notehead_xs)} visual noteheads")

    def draw(self, screen: pygame.Surface, y: int, progress: float, alpha: float = 1.0) -> None:
        if self.full_surface is None:
            return
        view_w = self.screen_width

        self._compute_target_offset(view_w, progress)
        dt = self._update_view_smoothing()

        x_off = int(round(max(0.0, min(self.view_x_off, float(max(0, self.full_width - view_w))))))
        src_rect = pygame.Rect(x_off, 0, view_w, self.strip_height)

        overlay = pygame.Surface((view_w, self.strip_height), pygame.SRCALPHA)
        overlay.blit(self.full_surface, (0, 0), src_rect)

        screen_play_x = self._compute_screen_play_x(view_w, x_off, dt)
        self._draw_overlay(overlay, 0, view_w, screen_play_x)
        self._draw_debug_lines(overlay, 0, x_off, view_w)
        aa = max(0.0, min(1.0, float(alpha)))
        overlay.set_alpha(int(aa * 255))
        screen.blit(overlay, (0, y))

    def _compute_target_offset(self, view_w: int, progress: float) -> None:
        """Compute and set self._target_x_off based on the current note index or progress."""
        max_off = max(0, self.full_width - view_w)
        play_x = int(view_w * 0.45)
        if self.notehead_xs and 0 <= int(self._current_note_idx) < len(self.notehead_xs):
            target_x = int(self.notehead_xs[int(self._current_note_idx)])
            desired = int(target_x - play_x)
            desired = max(0, min(desired, max_off))
            self._target_x_off = float(desired)
            return
        p = min(max(progress, 0.0), 1.0)
        self._target_x_off = float(int(p * max_off))

    def _update_view_smoothing(self) -> float:
        """Update view offset smoothing; returns dt in seconds used for this frame."""
        now_ms = pygame.time.get_ticks()
        dt = max(0.0, (now_ms - getattr(self, '_last_time_ms', now_ms)) / 1000.0)
        self._last_time_ms = now_ms
        if dt > 0.0:
            alpha = 1.0 - math.exp(-dt / max(1e-6, self._scroll_tau))
            self.view_x_off += (self._target_x_off - self.view_x_off) * alpha
            if abs(self._target_x_off - self.view_x_off) < 0.5:
                self.view_x_off = float(self._target_x_off)
        return dt

    def _compute_screen_play_x(self, view_w: int, x_off: int, dt: float) -> float:
        """Compute and smooth the play-line X position relative to the view.

        The play-line is smoothed toward the desired screen coordinate where the
        current chord/note visual is expected. This mirrors the original logic,
        so the overlay and strip scrolling remain synchronized.
        """
        if self.notehead_xs and 0 <= int(self._current_note_idx) < len(self.notehead_xs):
            desired_screen_play = float(int(self.notehead_xs[int(self._current_note_idx)]) - x_off)
        else:
            desired_screen_play = float(int(view_w * 0.45))

        if self._screen_play_x is None:
            self._screen_play_x = desired_screen_play

        dt_play = dt if dt > 0.0 else (1.0 / 60.0)
        alpha_play = 1.0 - math.exp(-dt_play / max(1e-6, self._play_tau))
        self._screen_play_x += (desired_screen_play - self._screen_play_x) * alpha_play
        if abs(desired_screen_play - self._screen_play_x) < 0.5:
            self._screen_play_x = float(desired_screen_play)
        return float(self._screen_play_x)

    def _init_screen_play_x(self, view_w: int, x_off: int) -> None:
        """Initialize the smoothed play-line X position if not already set.

        This consolidates duplicated logic used in advance_note and seek_to_index.
        """
        if self._screen_play_x is not None:
            return
        try:
            cur_idx = int(self._current_note_idx) if (self.notehead_xs and 0 <= int(self._current_note_idx) < len(self.notehead_xs)) else None
            if cur_idx is not None:
                start_play = float(int(self.notehead_xs[cur_idx]) - x_off)
            else:
                start_play = float(int(view_w * 0.45))
            self._screen_play_x = float(start_play)
        except (IndexError, TypeError, ValueError):
            self._screen_play_x = float(int(view_w * 0.45))

    def _compute_view_and_xoff(self) -> tuple[int, int]:
        """Compute a safe (view_width, x_off) pair used in several methods.

        Tries to clamp the current floating _view_x_off to the allowable range
        [0, full_width - view_w] and falls back to a best-effort value when
        attributes are missing or invalid.
        """
        view_w = self.screen_width
        try:
            x_off = int(round(max(0.0, min(self.view_x_off, float(max(0, self.full_width - view_w))))))
        except (AttributeError, TypeError, ValueError):
            x_off = int(round(self.view_x_off)) if hasattr(self, '_view_x_off') else 0
        return view_w, x_off

    def _draw_overlay(self, screen: pygame.Surface, y: int, view_w: int, screen_play_x: float) -> None:
        overlay = pygame.Surface((view_w, self.strip_height), pygame.SRCALPHA)
        rect_w = 30
        rect_h = self.strip_height
        rect_x = int(round(screen_play_x - rect_w // 2))
        rect_y = 0
        rect_color = (0, 200, 255, 128)
        try:
            pygame.draw.rect(overlay, rect_color, (rect_x, rect_y, rect_w, rect_h))
        except (pygame.error, TypeError):
            pass
        line_color = (0, 200, 255, 200)
        pygame.draw.line(overlay, line_color, (int(round(screen_play_x)), 0), (int(round(screen_play_x)), rect_h), 3)
        screen.blit(overlay, (0, y))

    def _draw_debug_lines(self, screen: pygame.Surface, y: int, x_off: int, view_w: int) -> None:
        if not self.debug:
            return
        positions = set()
        try:
            positions.update(int(x) for x in self.notehead_xs)
        except (TypeError, ValueError):
            pass
        for gx in sorted(positions):
            sx = gx - x_off
            if -4 <= sx <= view_w + 4:
                pygame.draw.line(screen, (90, 255, 120), (int(sx), y), (int(sx), y + self.strip_height), 1)

    def seek_to_index(self, index: int, animate: bool = True) -> None:
        """Seek directly to a visual note index (clamped).
        If animate is True, smoothly scrolls the view to the new target.
        If animate is False, jumps the view to the target immediately.
        """
        if self.notehead_xs:
            try:
                view_w, x_off = self._compute_view_and_xoff()
            except (AttributeError, TypeError, ValueError):
                view_w = self.screen_width
                x_off = int(round(self.view_x_off)) if hasattr(self, 'view_x_off') else 0

            if animate:
                self._init_screen_play_x(view_w, x_off)

            idx = max(0, min(int(index), len(self.notehead_xs) - 1))
            self._current_note_idx = idx
            self._compute_target_offset(self.screen_width, 0.0)

            if animate:
                try:
                    self._last_time_ms = pygame.time.get_ticks()
                except (pygame.error, AttributeError):
                    pass
                return
            else:
                self.view_x_off = float(self._target_x_off)
                self._screen_play_x = None
                try:
                    self._last_time_ms = pygame.time.get_ticks()
                except (pygame.error, AttributeError):
                    pass
                return
        self._current_note_idx = max(0, int(index))
