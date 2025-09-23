from pathlib import Path
import math
import pygame
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError
from sheet_music_renderer import midi_to_svg
from typing import Optional
import cairosvg
import json
import xml.etree.ElementTree as ElementTree
from save_system import SaveSystem
import io

_resampling = getattr(Image, "Resampling", Image)
RESAMPLE_LANCZOS = getattr(_resampling, "LANCZOS", getattr(_resampling, "BICUBIC", getattr(Image, "NEAREST", 0)))
_CAIROSVG_DPI = 1200
strip_png = "strip.png"
song_svg = "song.svg"

class SheetMusicRenderer:
    def __init__(self, midi_path: str | Path, screen_width: int, save_system: SaveSystem, height: int = 260, debug: bool = False) -> None:
        self.midi_path = Path(midi_path)
        self.screen_width = int(screen_width)
        self.strip_height = int(height)
        self.debug = debug
        self.save_system = save_system.sheet_music_cache

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

    def _load_cache_data(self) -> None:
        with self.save_system as s:
            self.notehead_xs = [float(x) for x in json.loads(s.load_file("note_xs.json")) if x is not None] if s.file_exists("note_xs.json") else []
            self.measure_data = [
                (float(tup[0]) if tup[0] is not None else None, float(tup[1]) if tup[1] is not None else None, int(tup[2] or 0))
                if tup is not None and len(tup) == 3 else (None, None, 0)
                for tup in json.loads(s.load_file("measure_data.json"))
            ] if s.file_exists("measure_data.json") else []

    def _call_midi_to_svg(self, cache_dir: Path, svg_out: Path) -> None:
        try:
            try:
                res_note_xs, res_measure_data = midi_to_svg(str(self.midi_path), str(cache_dir))
            except TypeError:
                res_note_xs, res_measure_data = midi_to_svg(str(self.midi_path), str(cache_dir), "mscore")
            
            self.measure_data = res_measure_data
            if isinstance(res_note_xs, list):
                self.notehead_xs = [float(x) for x in res_note_xs]
                try:
                    with self.save_system as s:
                        s.save_file("note_xs.json", json.dumps(self.notehead_xs))
                except (OSError, TypeError, ValueError) as e:
                    print("SheetMusicRenderer: failed to save note x positions:", e)

            # Save measure data
            if isinstance(res_measure_data, list):
                try:
                    with self.save_system as s:
                        serializable_measure_data = []
                        for tup in res_measure_data:
                            if tup is not None and len(tup) == 3:
                                sx, ex, n = tup
                                serializable_measure_data.append([sx, ex, n])
                            else:
                                serializable_measure_data.append([None, None, 0])
                        s.save_file("measure_data.json", json.dumps(serializable_measure_data))
                except (OSError, TypeError, ValueError) as e:
                    print(f"SheetMusicRenderer: failed to save measure data: {e}")

        except (RuntimeError, OSError, ValueError, TypeError) as e:
            print("SheetMusicRenderer: failed to render SVG via sheet_music_renderer:", e)
            return

        if not svg_out.exists():
            svgs = list(cache_dir.glob(f"{self.midi_path.stem}*.svg"))
            if svgs:
                svg_out = svgs[0]

        if not svg_out.exists():
            print("No SVG produced by sheet_music_renderer; aborting strip preparation.")
            return

    @staticmethod
    def _convert_svg_to_png(svg_out: Path, strip_png_path: Path) -> bool:
        try:
            cairosvg.svg2png(url=str(svg_out), write_to=str(strip_png_path), dpi=_CAIROSVG_DPI, scale=2.0, negate_colors=True)
            return True
        except (RuntimeError, OSError, ValueError) as e:
            print("Failed to convert SVG to PNG: ", e)
            return False

    def _open_and_process_image(self) -> tuple[Optional[Image.Image], Optional[float], int, int]:
        """Open PNG, crop whitespace on the left, resize to strip_height.
        Returns (pil_image, svg_width_if_found_or_None, cropped_w, left_crop)
        """
        with self.save_system as s:
            try:
                im = Image.open(io.BytesIO(s.load_file(strip_png))).convert("RGBA")
            except (FileNotFoundError, OSError, UnidentifiedImageError) as e:
                print("Failed to open generated PNG:", e)
                return None, None, 0, 0

            svg_width = None
            if s.file_exists(song_svg):
                try:
                    tree = ElementTree.parse(io.BytesIO(s.load_file(song_svg)))
                    root = tree.getroot()
                    w = root.get('width') or root.get('{http://www.w3.org/2000/svg}width')
                    if w:
                        if isinstance(w, str) and w.endswith('px'):
                            w = w[:-2]
                        svg_width = float(w)
                except (ElementTree.ParseError, OSError, ValueError) as e:
                    print("Failed to parse SVG width:", e)
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

            if self.measure_data:
                scaled_measure_data = []
                for tup in self.measure_data:
                    if tup is not None and len(tup) == 3:
                        sx, ex, n = tup
                        scaled = []
                        for item in [sx, ex]:
                            if item is not None:
                                try:
                                    item_px = float(item) * pixels_per_svg_unit
                                    item_adj_px = item_px - float(left_crop)
                                    if 0 <= item_adj_px <= float(cropped_w):
                                        scaled.append(int(round(item_adj_px * final_scale)))
                                    else:
                                        scaled.append(None)
                                except (TypeError, ValueError):
                                    scaled.append(None)
                            else:
                                scaled.append(None)
                        scaled_measure_data.append((scaled[0], scaled[1], int(n)))
                    else:
                        scaled_measure_data.append((None, None, 0))
                self.measure_data = scaled_measure_data
        except (TypeError, ValueError, ZeroDivisionError) as e:
            if self.debug:
                print("Failed to scale cached note x positions and measure data:", e)
            self.notehead_xs = []
            self.measure_data = []

    def _prepare_strip(self) -> None:
        with self.save_system as s:
            cache_dir = s.get_absolute_path()
            self._load_cache_data()

            if not s.file_exists(strip_png) or not self.notehead_xs or not self.measure_data:
                self._call_midi_to_svg(cache_dir, s.get_absolute_path(song_svg))
                if not s.file_exists(song_svg):
                    raise RuntimeError("No SVG produced by sheet_music_renderer; aborting strip preparation.")
                ok = self._convert_svg_to_png(s.get_absolute_path(song_svg), s.get_absolute_path(strip_png))
                if not ok:
                    return

        strip_resized, svg_width, cropped_w, left_crop = self._open_and_process_image()
        if strip_resized is None:
            return
        try:
            data = strip_resized.tobytes()
            self.full_surface = pygame.image.frombytes(data, strip_resized.size, "RGBA").convert_alpha()
        except (pygame.error, ValueError, TypeError) as e:
            print("Failed to create pygame surface from strip image:", e)
            return

        self.full_width = strip_resized.size[0]
        self._scale_cached_note_positions(svg_width=svg_width, cropped_w=cropped_w, left_crop=left_crop, strip_resized=strip_resized)

        if self.debug:
            print(f"Built strip from SVG: width={self.full_width}px, detected {len(self.notehead_xs)} visual noteheads")

    def draw(self, screen: pygame.Surface, y: int, progress: float, guided_teacher, alpha: float = 1.0) -> None:
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
        current_note_highlight = self._draw_overlay(overlay, view_w, screen_play_x)
        if guided_teacher is not None:
            self._draw_guided_highlight(overlay, guided_teacher, x_off, current_note_highlight)
        self._draw_debug_lines(overlay, 0, x_off, view_w)
        aa = max(0.0, min(1.0, float(alpha)))
        overlay.set_alpha(int(aa * 255))
        screen.blit(overlay, (0, y))

    def _draw_guided_highlight(self, screen, guided_teacher, x_off, current_note_highlight=None):
        if not guided_teacher.is_active or not guided_teacher.current_section_visual_info:
            return
        padding = 12
        animation_duration = 150  # ms
        rect_color = (0, 255, 0, 64)

        start_x = guided_teacher.current_section_visual_info[0]
        rect_x = start_x - padding
        rect_w = guided_teacher.current_section_visual_info[-1] - start_x + padding*2

        now = pygame.time.get_ticks()
        target = (rect_x, 0, rect_w, self.strip_height)
        if not hasattr(self, "_guided_anim_state"):
            self._guided_anim_state = {
                'prev': target,
                'target': target,
                'start_time': now,
                'animating': False
            }
        state = self._guided_anim_state
        if state['target'] != target:
            state['prev'] = state['prev'] if state['animating'] else state['target']
            state['target'] = target
            state['start_time'] = now
            state['animating'] = True
        if state['animating']:
            elapsed = now - state['start_time']
            t = min(1.0, elapsed / animation_duration)
            interp = lambda a, b: a + (b - a) * t
            cur_rect = tuple(int(interp(p, q)) for p, q in zip(state['prev'], state['target']))
            if t >= 1.0:
                state['animating'] = False
                state['prev'] = state['target']
        else:
            cur_rect = state['target']
        overlay = pygame.Surface((cur_rect[2], cur_rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(overlay, rect_color, (0, 0, cur_rect[2], cur_rect[3]), border_radius=10)
        overlay.blit(
            pygame.Surface((current_note_highlight[2], current_note_highlight[3]), pygame.SRCALPHA) if current_note_highlight else pygame.Surface((0, 0), pygame.SRCALPHA),
            (current_note_highlight[0]-cur_rect[0]+x_off, 0) if current_note_highlight else (0, 0),
            None,
            pygame.BLEND_RGBA_MULT
        )
        screen.blit(overlay, (cur_rect[0]-x_off, cur_rect[1]))

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

    def _draw_overlay(self, screen: pygame.Surface, view_w: int, screen_play_x: float):
        overlay = pygame.Surface((view_w, self.strip_height), pygame.SRCALPHA)
        rect_w = 30
        rect_h = self.strip_height
        rect_x = int(round(screen_play_x - rect_w // 2))
        rect_color = (0, 200, 255, 128)
        try:
            pygame.draw.rect(overlay, rect_color, (rect_x, 0, rect_w, rect_h))
        except (pygame.error, TypeError):
            pass
        line_color = (0, 200, 255, 200)
        pygame.draw.line(overlay, line_color, (int(round(screen_play_x)), 0), (int(round(screen_play_x)), rect_h), 3)
        screen.blit(overlay, (0, 0))
        return rect_x, 0, rect_w, rect_h

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
