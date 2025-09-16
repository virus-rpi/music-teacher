from pathlib import Path
import math
import pygame
from PIL import Image, ImageOps, ImageFilter
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
        self.note_to_xs: dict[int, list[int]] = {}
        self.system_boxes: list[tuple[int, int]] = []
        self.notehead_xs: list[int] = []
        self._notehead_xs_per_system: list[list[int]] = []
        self._current_note_idx: int = 0
        self._view_x_off: float = 0.0
        self._target_x_off: float = 0.0
        self._scroll_tau: float = 0.08
        self._screen_play_x: Optional[float] = None
        self._play_tau: float = 0.03
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

    def _load_notehead_cache(self, note_xs_cache: Path) -> None:
        if not note_xs_cache.exists():
            return
        try:
            with note_xs_cache.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
                self.notehead_xs = [float(x) for x in raw if x is not None]
        except Exception:
            self.notehead_xs = []

    def _call_midi_to_svg(self, cache_subdir: Path, svg_out: Path, note_xs_cache: Path) -> None:
        try:
            try:
                res = midi_to_svg(str(self.midi_path), str(cache_subdir))
            except TypeError:
                res = midi_to_svg(str(self.midi_path), str(cache_subdir), "mscore")
            if isinstance(res, list):
                try:
                    with note_xs_cache.open("w", encoding="utf-8") as fh:
                        json.dump([float(x) for x in res], fh)
                except Exception as e:
                    print("SheetMusicRenderer: failed to save note x positions:", e)
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

    @staticmethod
    def _convert_svg_to_png(svg_out: Path, strip_png: Path) -> bool:
        try:
            cairosvg.svg2png(url=str(svg_out), write_to=str(strip_png), dpi=_CAIROSVG_DPI, scale=2.0, negate_colors=True)
            return True
        except Exception as e:
            print("Failed to convert SVG to PNG: ", e)
            return False

    def _open_and_process_image(self, strip_png: Path, svg_path: Path) -> tuple[Optional[Image.Image], Optional[float], int, int]:
        """Open PNG, crop whitespace on left, resize to strip_height.
        Returns (pil_image, svg_width_if_found_or_None, cropped_w, left_crop)
        """
        try:
            im = Image.open(str(strip_png)).convert("RGBA")
        except Exception as e:
            print("Failed to open generated PNG:", e)
            return None, None, 0, 0

        svg_width = None
        if svg_path.exists():
            tree = ElementTree.parse(str(svg_path))
            root = tree.getroot()
            w = root.get('width') or root.get('{http://www.w3.org/2000/svg}width')
            if w:
                if isinstance(w, str) and w.endswith('px'):
                    w = w[:-2]
                svg_width = float(w)

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
                except Exception:
                    continue
                x_adj_px = x_px - float(left_crop)
                if x_adj_px < 0 or x_adj_px > float(cropped_w):
                    continue
                pxs.append(int(round(x_adj_px * final_scale)))
            pxs = sorted(set(pxs))
            self.notehead_xs = pxs
        except Exception as e:
            if self.debug:
                print("Failed to scale cached note x positions:", e)
            self.notehead_xs = []

    def _prepare_strip(self) -> None:
        cache_dir = self._get_cache_dir()
        cache_key = self._midi_cache_key()
        cache_subdir = cache_dir / cache_key
        cache_subdir.mkdir(parents=True, exist_ok=True)

        strip_png = cache_subdir / "strip.png"
        svg_out = cache_subdir / f"{self.midi_path.stem}.svg"
        note_xs_cache = cache_subdir / "note_xs.json"
        self._load_notehead_cache(note_xs_cache)
        if not strip_png.exists() or not self.notehead_xs:
            self._call_midi_to_svg(cache_subdir, svg_out, note_xs_cache)
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
        except Exception as e:
            print("Failed to create pygame surface from strip image:", e)
            return

        self.full_width = strip_resized.size[0]
        self._scale_cached_note_positions(svg_width=svg_width, cropped_w=cropped_w, left_crop=left_crop, strip_resized=strip_resized)

        if self.debug:
            print(f"Built strip from SVG: width={self.full_width}px, detected {len(self.notehead_xs)} visual noteheads")

    def draw(self, screen: pygame.Surface, y: int, progress: float) -> None:
        if self.full_surface is None:
            return
        view_w = self.screen_width

        self._compute_target_offset(view_w, progress)
        dt = self._update_view_smoothing()

        x_off = int(round(max(0.0, min(self._view_x_off, float(max(0, self.full_width - view_w))))))
        src_rect = pygame.Rect(x_off, 0, view_w, self.strip_height)
        screen.blit(self.full_surface, (0, y), src_rect)

        screen_play_x = self._compute_screen_play_x(view_w, x_off, dt)

        self._draw_overlay(screen, y, view_w, screen_play_x)
        self._draw_debug_lines(screen, y, x_off, view_w)

        pygame.draw.rect(screen, (20, 20, 20), (0, y, view_w, self.strip_height), 2)

    def _compute_target_offset(self, view_w: int, progress: float) -> None:
        """Compute and set self._target_x_off based on current note index or progress."""
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
            self._view_x_off += (self._target_x_off - self._view_x_off) * alpha
            if abs(self._target_x_off - self._view_x_off) < 0.5:
                self._view_x_off = float(self._target_x_off)
        return dt

    def _compute_screen_play_x(self, view_w: int, x_off: int, dt: float) -> float:
        """Compute and smooth the play-line X position relative to the view.

        The play-line is smoothed toward the desired screen coordinate where the
        current chord/note visual is expected. This mirrors the original logic
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

    def _draw_overlay(self, screen: pygame.Surface, y: int, view_w: int, screen_play_x: float) -> None:
        overlay = pygame.Surface((view_w, self.strip_height), pygame.SRCALPHA)
        rect_w = 30
        rect_h = self.strip_height
        rect_x = int(round(screen_play_x - rect_w // 2))
        rect_y = 0
        rect_color = (0, 200, 255, 128)
        try:
            pygame.draw.rect(overlay, rect_color, (rect_x, rect_y, rect_w, rect_h))
        except Exception:
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

    def advance_note(self, step: int = 1) -> None:
        """Advance the current note index by `step` and update target/playing positions.
        Keeps the smooth scrolling behavior but resets the play-line smoothing so it follows the new note.
        """
        try:
            idx = int(self._current_note_idx) + int(step)
        except Exception:
            idx = int(step)
        if self.notehead_xs:
            idx = max(0, min(idx, len(self.notehead_xs) - 1))
        else:
            idx = max(0, idx)
        self._current_note_idx = idx
        self._compute_target_offset(self.screen_width, 0.0)

        self._screen_play_x = None
        try:
            self._last_time_ms = pygame.time.get_ticks()
        except Exception:
            pass

    def reset_note_index(self) -> None:
        """Reset current note index to the start and jump view to that position."""
        self.seek_to_index(0)

    def seek_to_progress(self, progress: float) -> None:
        """Seek the sheet view to a relative progress (0.0-1.0).
        If visual note positions are available, map progress to a note index and seek to it.
        Otherwise jump the view offset directly.
        """
        p = min(max(float(progress), 0.0), 1.0)
        if self.notehead_xs:
            count = len(self.notehead_xs)
            idx = int(round(p * (count - 1))) if count > 0 else 0
            self.seek_to_index(idx)
            return
        max_off = max(0, self.full_width - self.screen_width)
        self._target_x_off = float(int(p * max_off))
        self._view_x_off = float(self._target_x_off)
        self._screen_play_x = None
        try:
            self._last_time_ms = pygame.time.get_ticks()
        except Exception:
            pass

    def seek_to_index(self, index: int) -> None:
        """Seek directly to a visual note index (clamped). Jumps the view to the target immediately."""
        # If chord_xs exists, index refers to chord index; otherwise it's a visual-note index
        if self.notehead_xs:
            idx = max(0, min(int(index), len(self.notehead_xs) - 1))
            self._current_note_idx = idx
            self._compute_target_offset(self.screen_width, 0.0)
            self._view_x_off = float(self._target_x_off)
            self._screen_play_x = None
            try:
                self._last_time_ms = pygame.time.get_ticks()
            except Exception:
                pass
            return
        else:
            self._current_note_idx = max(0, int(index))
