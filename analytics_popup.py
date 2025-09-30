import io
import json
import re
from collections import defaultdict
import matplotlib
import numpy as np
import pygame
from pygame_gui.core import ObjectID
from flexbox import FlexBox
from save_system import SaveSystem

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pygame_gui


def _render_background(surface):
    pygame.draw.rect(surface, (30, 30, 30), (0, 0, surface.get_width(), surface.get_height()), border_radius=16)
    pygame.draw.rect(surface, (200, 200, 200), (0, 0, surface.get_width(), surface.get_height()), 2, border_radius=16)


def _generate_tips(analytics) -> pygame_gui.elements.UITextBox:
    tips = []
    if not analytics:
        return pygame_gui.elements.UITextBox(
            html_text='<i>No analytics available.</i>',
            relative_rect=pygame.Rect(0, 0, 100, 10),
        )
    if analytics.get('worst_chord_idx') is not None:
        idx = analytics['worst_chord_idx'] + 1
        score = analytics['worst_chord_score']
        tips.append(f'Chord {idx} had the lowest accuracy: {score:.2f}')
    if analytics.get('worst_interval_idx') is not None:
        idx = analytics['worst_interval_idx'] + 1
        diff = analytics['worst_interval_diff']
        tips.append(f'Largest timing gap error between onsets {idx} and {idx + 1}: {diff:.2f}')
    if analytics.get('tempo_bias') is not None:
        bias = analytics['tempo_bias']
        if abs(bias) > 0.02:
            tips.append(f'Overall tempo was {"faster" if bias > 0 else "slower"} by {abs(bias) * 100:.1f}%')
    if analytics.get('legato_detected'):
        sev = analytics.get('legato_severity', 0)
        tips.append(f'Legato severity {sev * 100:.1f}% (lower is better)')
    if not tips:
        tips = ['No major issues detected.']
    list_items = ''.join([f'- {t} <br/>' for t in tips])
    return pygame_gui.elements.UITextBox(
        html_text=f'<b>Tips:</b><br/>{list_items}',
        relative_rect=pygame.Rect(0, 0, 100, 10000),
    )


def _render_overall_score(analytics, big_font, flex_box):
    score_val = analytics.get('score', None).get('overall', None) if analytics and analytics.get('score') else None
    if score_val is not None:
        score_text = f"Overall Score: {score_val * 100:.0f}%"
    else:
        score_text = "Overall Score: —"
    score_color = (80, 200, 120) if score_val and score_val >= 0.8 else (255, 120, 120)
    score_surf = big_font.render(score_text, True, score_color)
    flex_box.place_element(
        pygame_gui.elements.UIImage(
            image_surface=score_surf,
            relative_rect=pygame.Rect(0, 0, score_surf.get_width(), score_surf.get_height()),
        ),
        height_px=score_surf.get_height(),
        width_px=score_surf.get_width(),
    )


class UIManagerWithOffset(pygame_gui.UIManager):
    def __init__(self, window_resolution, offset, *args, **kwargs):
        super().__init__(window_resolution, *args, **kwargs)
        self.offset: tuple[int, int] = offset

    def calculate_scaled_mouse_position(
            self, position: tuple[int, int]
    ) -> tuple[int, int]:
        return (
            int(self.mouse_pos_scale_factor[0] * position[0]) - self.offset[0],
            int(self.mouse_pos_scale_factor[1] * position[1]) - self.offset[1],
        )

    def set_offset(self, offset):
        self.offset = offset


class ElementSurface:
    def __init__(self, surface):
        self.parent_surface = surface
        self.width = int(surface.get_width() * 0.9)
        self.height = int(surface.get_height() * 0.85)
        self.x = (surface.get_width() - self.width) // 2
        self.y = (surface.get_height() - self.height) // 4
        self.surface: pygame.Surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

    def __enter__(self):
        return self.surface, (self.x, self.y), (self.width, self.height)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.parent_surface.blit(self.surface, (self.x, self.y))


class AnalyticsPopup:
    def __init__(self, teacher, save_system: SaveSystem):
        self.teacher = teacher
        self.save_system = save_system
        self.visible = False

        self._margin = 32
        self._font = pygame.font.SysFont('Arial', 22)
        self._small_font = pygame.font.SysFont('Arial', 18)
        self._big_font = pygame.font.SysFont('Arial', 28, bold=True)

        self._ui_manager = UIManagerWithOffset((800, 600), (0, 0))

        self._selected_measure = None
        self._selected_section = None
        self._selected_pass = None
        self.pass_map = None
        self._measure_selector = None
        self._section_selector = None
        self._pass_selector = None

        self._dropdown_container: FlexBox | None = None
        self._main_container: FlexBox | None = None

        self._last_update_ms = pygame.time.get_ticks()

        self._analytics = None

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self._build_path_map()

    def draw(self, surface: pygame.Surface):
        if not self.visible:
            return

        with ElementSurface(surface) as (e, offset, dims):
            if self._ui_manager.window_resolution != e.get_size():
                self._ui_manager.set_window_resolution(e.get_size())
                self._ui_manager.set_offset(offset)
            _render_background(e)
            self._render_title(e)

            if not self._main_container:
                self._main_container = FlexBox(manager=self._ui_manager,
                                               relative_rect=pygame.Rect(self._margin, self._margin,
                                                                         dims[0] - self._margin * 2,
                                                                         dims[1] - self._margin * 2), gap=12)

                left_side = FlexBox(manager=self._ui_manager, relative_rect=pygame.Rect(0, 0, 0, 0), gap=12)
                self._main_container.place_element(left_side, width_percent=0.6, height_percent=1)
                _render_overall_score(self._analytics, self._big_font, left_side)
                self._main_container.place_element(_generate_tips(self._analytics), width_percent=0.4, height_percent=1)

            now = pygame.time.get_ticks()
            dt = max(0, now - self._last_update_ms) / 1000.0
            self._last_update_ms = now
            self._ui_manager.update(dt)
            self._ui_manager.draw_ui(e)

    def _update(self):
        self._measure_selector.kill()
        self._section_selector.kill()
        self._pass_selector.kill()
        self._dropdown_container.kill()
        self._dropdown_container = None
        self._main_container.kill()
        self._main_container = None
        self._analytics = self._get_selected_analytics()

    def handle_event(self, event):
        if not self.visible:
            return

        self._ui_manager.process_events(event)

        if event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self._measure_selector:
                selected = self._measure_selector.selected_option[0]
                m = re.match(r'Measure (\d+)', selected)
                if m:
                    self._selected_measure = int(m.group(1))
                    sections = sorted(self.pass_map[self._selected_measure].keys())
                    self._selected_section = sections[0] if sections else None
                    passes = list(self.pass_map[self._selected_measure][
                                      self._selected_section].keys()) if self._selected_section is not None else []
                    self._selected_pass = passes[0] if passes else None
                    self._update()
            elif event.ui_element == self._section_selector:
                selected = self._section_selector.selected_option[0]
                s = re.match(r'Section (\w+|\d+)', selected)
                if s:
                    self._selected_section = s.group(1)
                    passes = list(self.pass_map[self._selected_measure][
                                      self._selected_section].keys()) if self._selected_measure is not None and self._selected_section is not None else []
                    self._selected_pass = passes[0] if passes else None
                    self._update()
            elif event.ui_element == self._pass_selector:
                selected = self._pass_selector.selected_option[0]
                p = re.match(r'Pass (\d+)', selected)
                if p:
                    self._selected_pass = int(p.group(1))
                    self._update()

    def _render_title(self, surface):
        title_surf = self._font.render('Performance Analytics for', True, (255, 255, 255))
        surface.blit(title_surf, (self._margin, self._margin))
        self._render_dropdowns(self._margin + title_surf.get_width() + 12)

    def _render_dropdowns(self, x):
        if self._dropdown_container:
            return
        self._dropdown_container = FlexBox(
            manager=self._ui_manager,
            relative_rect=pygame.Rect(x, self._margin, 0, 32),
            gap=12,
        )

        def make_dropdown(options, starting_option):
            return pygame_gui.elements.UIDropDownMenu(
                options_list=options,
                starting_option=starting_option,
                relative_rect=pygame.Rect(0, 0, 0, 0),
            )

        measure_options = [f"Measure {m}" for m in sorted(self.pass_map.keys(), key=int)] if self.pass_map else ['—']
        self._measure_selector = make_dropdown(measure_options,
                                               f"Measure {self._selected_measure}" if self._selected_measure else
                                               measure_options[0])
        section_options = [f"Section {s}" for s in sorted(
            self.pass_map[self._selected_measure].keys())] if self._selected_measure is not None and self.pass_map.get(
            self._selected_measure) else ['—']
        self._section_selector = make_dropdown(section_options,
                                               f"Section {self._selected_section}" if self._selected_section else
                                               section_options[0])
        pass_options = [f"Pass {p}" for p in sorted(self.pass_map[self._selected_measure][
                                                        self._selected_section].keys())] if self._selected_measure is not None and self._selected_section is not None and self.pass_map.get(
            self._selected_measure, {}).get(self._selected_section) else ['—']

        self._pass_selector = make_dropdown(pass_options,
                                            f"Pass {self._selected_pass}" if self._selected_pass else pass_options[0])
        self._dropdown_container.place_element(self._measure_selector, height_percent=1, width_px=200)
        self._dropdown_container.place_element(self._section_selector, height_percent=1, width_px=220)
        self._dropdown_container.place_element(self._pass_selector, height_percent=1, width_px=140)

    def _build_path_map(self):
        self._selected_measure = None
        self._selected_section = None
        self._selected_pass = None
        self.pass_map = defaultdict(lambda: defaultdict(dict))
        results = self.save_system.search_index(module="guided_teacher_data", rel_path=['pass_', '.json'],
                                                sort_by='timestamp', ascending=False, rel_path_match="all")
        for entry in results:
            path = entry['rel_path']
            m = re.search(r'measure_(\d+)/section_(\w+|\d+)/pass_(\d+)\.json', path)
            if m:
                measure = int(m.group(1))
                section = str(m.group(2))
                pass_num = int(m.group(3))
                self.pass_map[measure][section][pass_num] = path
                if self._selected_measure is None:
                    self._selected_measure = measure
                    self._selected_section = section
                    self._selected_pass = pass_num

    def _get_selected_analytics(self):
        if not (self.pass_map and self.pass_map[self._selected_measure] and self.pass_map[self._selected_measure][
            self._selected_section] and self.pass_map[self._selected_measure][self._selected_section][
                    self._selected_pass]):
            return {}
        with self.save_system.guided_teacher_data as s:
            raw = json.loads(
                s.load_file(self.pass_map[self._selected_measure][self._selected_section][self._selected_pass]))
            analytics = raw.get('analytics', {})
            return analytics
