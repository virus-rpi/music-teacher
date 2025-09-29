import io
import json
import re
from collections import defaultdict
from pprint import pprint

import matplotlib
import numpy as np
import pygame
from pygame_gui.core import ObjectID

from save_system import SaveSystem

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pygame_gui

class AnalyticsPopup:
    def __init__(self, teacher, save_system: SaveSystem):
        self.teacher = teacher
        self.save_system = save_system
        self.visible = False
        self.margin = 30
        self.font = None
        self.small_font = None
        self.width = None
        self.height = None
        self.ui_manager = pygame_gui.UIManager((800, 600))
        self.dd_measure = None
        self.dd_section = None
        self.dd_pass = None
        self.img_spider = None
        self.img_pianoroll = None
        self.tips_box = None
        self._last_update_ms = pygame.time.get_ticks()
        self.pass_map = None
        self._init_selector()
        self._selected_measure = None
        self._selected_section = None
        self._selected_pass = None
        self.font = pygame.font.SysFont('Arial', 22)
        self.small_font = pygame.font.SysFont('Arial', 18)
        self.big_font = pygame.font.SysFont('Arial', 28, bold=True)

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self._init_selector()
        else:
            self._reset_ui_elements()

    def _reset_ui_elements(self):
        for attr in ['dd_measure', 'dd_section', 'dd_pass', 'img_spider', 'img_pianoroll', 'tips_box']:
            obj = getattr(self, attr, None)
            if obj is not None:
                if hasattr(obj, 'kill'):
                    obj.kill()
                setattr(self, attr, None)

    def _init_selector(self):
        self._selected_measure = None
        self._selected_section = None
        self._selected_pass = None
        self.pass_map = defaultdict(lambda: defaultdict(dict))
        results = self.save_system.search_index(module="guided_teacher_data", rel_path=['pass_', '.json'], sort_by='timestamp', ascending=False, rel_path_match="all")
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

    def _create_dropdown(self, options, starting_option, width, object_id):
        return pygame_gui.elements.UIDropDownMenu(
            options_list=options,
            starting_option=starting_option,
            relative_rect=pygame.Rect(0, 0, width, 32),
            manager=self.ui_manager,
            object_id=object_id
        )

    @staticmethod
    def _get_layout_dimensions(sw, sh):
        margin = 32
        width = int(sw * 0.9)
        height = int(sh * 0.85)
        left_w = int(width * 0.7)
        right_w = width - left_w - margin * 2
        tips_w = right_w
        tips_h = height - margin * 2
        heading_h = 48
        score_h = 40
        left_h = height - margin * 2
        # Use full left column width for radar and pianoroll
        radar_w = left_w
        radar_h = int(left_h * 0.5) - margin // 2
        pianoroll_w = left_w
        pianoroll_h = int(left_h * 0.5) - margin // 2
        return {
            'width': width,
            'height': height,
            'margin': margin,
            'left_w': left_w,
            'right_w': right_w,
            'tips_w': tips_w,
            'tips_h': tips_h,
            'heading_h': heading_h,
            'score_h': score_h,
            'radar_w': radar_w,
            'radar_h': radar_h,
            'pianoroll_w': pianoroll_w,
            'pianoroll_h': pianoroll_h
        }

    def _build_ui(self, sw, sh):
        dims = self._get_layout_dimensions(sw, sh)
        self.width = dims['width']
        self.height = dims['height']
        tips_w = dims['tips_w']
        tips_h = dims['tips_h']
        radar_w = dims['radar_w']
        radar_h = dims['radar_h']
        pianoroll_h = dims['pianoroll_h']
        dd_measure_w = 200
        dd_section_w = 220
        dd_pass_w = 140
        measure_options = [f"Measure {m}" for m in sorted(self.pass_map.keys(), key=int)] if self.pass_map else ['—']
        starting_measure = f"Measure {self._selected_measure}" if self._selected_measure else measure_options[0]
        self.dd_measure = self._create_dropdown(measure_options, starting_measure, dd_measure_w, '#analytics_dd_measure')
        section_options = [f"Section {s}" for s in sorted(self.pass_map[self._selected_measure].keys())] if self._selected_measure is not None and self.pass_map.get(self._selected_measure) else ['—']
        starting_section = f"Section {self._selected_section}" if self._selected_section else section_options[0]
        self.dd_section = self._create_dropdown(section_options, starting_section, dd_section_w, ObjectID(class_id='@section_dropdown'))
        pass_options = [f"Pass {p}" for p in sorted(self.pass_map[self._selected_measure][self._selected_section].keys())] if self._selected_measure is not None and self._selected_section is not None and self.pass_map.get(self._selected_measure, {}).get(self._selected_section) else ['—']
        starting_pass = f"Pass {self._selected_pass}" if self._selected_pass else pass_options[0]
        self.dd_pass = self._create_dropdown(pass_options, starting_pass, dd_pass_w, '#analytics_dd_pass')
        self.img_spider = pygame_gui.elements.UIImage(
            relative_rect=pygame.Rect(0, 0, radar_w, radar_h),
            image_surface=pygame.Surface((radar_w, radar_h), pygame.SRCALPHA),
            manager=self.ui_manager,
            object_id='#analytics_img_spider'
        )
        self.img_pianoroll = pygame_gui.elements.UIImage(
            relative_rect=pygame.Rect(0, 0, radar_w, pianoroll_h),
            image_surface=pygame.Surface((radar_w, pianoroll_h), pygame.SRCALPHA),
            manager=self.ui_manager,
            object_id='#analytics_img_pianoroll'
        )
        self.tips_box = pygame_gui.elements.UITextBox(
            html_text='',
            relative_rect=pygame.Rect(0, 0, tips_w, tips_h),
            manager=self.ui_manager,
            object_id='#analytics_tips_box'
        )

    def _rebuild_section_dropdown(self):
        if self.dd_section is not None:
            self.dd_section.kill()
        dd_section_w = 220
        section_options = [f"Section {s}" for s in sorted(self.pass_map[self._selected_measure].keys())] if self._selected_measure is not None and self.pass_map.get(self._selected_measure) else ['—']
        starting_section = f"Section {self._selected_section}" if self._selected_section else section_options[0]
        self.dd_section = self._create_dropdown(section_options, starting_section, dd_section_w, ObjectID(class_id='@section_dropdown'))
        self._rebuild_pass_dropdown()

    def _rebuild_pass_dropdown(self):
        if self.dd_pass is not None:
            self.dd_pass.kill()
        dd_pass_w = 140
        pass_options = [f"Pass {p}" for p in sorted(self.pass_map[self._selected_measure][self._selected_section].keys())] if self._selected_measure is not None and self._selected_section is not None and self.pass_map.get(self._selected_measure, {}).get(self._selected_section) else ['—']
        starting_pass = f"Pass {self._selected_pass}" if self._selected_pass else (pass_options[0] if pass_options else '—')
        self.dd_pass = self._create_dropdown(pass_options, starting_pass, dd_pass_w, '#analytics_dd_pass')

    def handle_event(self, event):
        if not self.visible:
            return
        self.ui_manager.process_events(event)
        if event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.dd_measure:
                selected = self.dd_measure.selected_option[0]
                m = re.match(r'Measure (\d+)', selected)
                if m:
                    self._selected_measure = int(m.group(1))
                    sections = sorted(self.pass_map[self._selected_measure].keys())
                    self._selected_section = sections[0] if sections else None
                    passes = list(self.pass_map[self._selected_measure][self._selected_section].keys()) if self._selected_section is not None else []
                    self._selected_pass = passes[0] if passes else None
                self._rebuild_section_dropdown()
                self._refresh_charts_and_tips()
            elif event.ui_element == self.dd_section:
                selected = self.dd_section.selected_option[0]
                s = re.match(r'Section (\w+|\d+)', selected)
                if s:
                    self._selected_section = s.group(1)
                    passes = list(self.pass_map[self._selected_measure][self._selected_section].keys()) if self._selected_measure is not None and self._selected_section is not None else []
                    self._selected_pass = passes[0] if passes else None
                self._rebuild_pass_dropdown()
                self._refresh_charts_and_tips()
            elif event.ui_element == self.dd_pass:
                selected = self.dd_pass.selected_option[0]
                p = re.match(r'Pass (\d+)', selected)
                if p:
                    self._selected_pass = int(p.group(1))
                self._refresh_charts_and_tips()

    def _get_selected_analytics(self):
        if not (self.pass_map and self.pass_map[self._selected_measure] and self.pass_map[self._selected_measure][self._selected_section] and self.pass_map[self._selected_measure][self._selected_section][self._selected_pass]):
            return {}
        with self.save_system.guided_teacher_data as s:
            raw = json.loads(s.load_file(self.pass_map[self._selected_measure][self._selected_section][self._selected_pass]))
            analytics = raw.get('analytics', {})
            return analytics

    def _refresh_charts_and_tips(self):
        analytics = self._get_selected_analytics()
        dims = self._get_layout_dimensions(self.width, self.height)
        radar_w = dims['left_w']
        radar_h = self.img_spider.rect.height
        pianoroll_w = dims['left_w']
        pianoroll_h = self.img_pianoroll.rect.height
        if analytics:
            spider_chart = self._matplotlib_spider_chart(analytics, width=radar_w, height=radar_h)
            self.img_spider.set_image(spider_chart)
            tips_html = self._generate_tips_html(analytics)
            self.tips_box.set_text(tips_html)
            pianoroll_rect = self.img_pianoroll.rect.copy()
            self._render_pianoroll(self.img_pianoroll.image, pianoroll_rect)
        else:
            self.img_spider.set_image(pygame.Surface((radar_w, radar_h), pygame.SRCALPHA))
            self.tips_box.set_text('<b>No analytics available yet.</b>')
            self.img_pianoroll.set_image(pygame.Surface((pianoroll_w, pianoroll_h), pygame.SRCALPHA))

    @staticmethod
    def _generate_tips_html(analytics):
        tips = []
        if not analytics:
            return '<i>No analytics available.</i>'
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
        return f'<b>Tips:</b><br/>{list_items}'

    def _draw_left_column(self, surface, dims, analytics):
        left_x = (surface.get_width() - dims['width']) // 2 + dims['margin']
        left_y = (surface.get_height() - dims['height']) // 4 + dims['margin']
        left_h = dims['height'] - 2 * dims['margin']
        # Title and dropdowns
        title_surf = self.font.render('Performance Analytics for', True, (255, 255, 255))
        surface.blit(title_surf, (left_x, left_y))
        dd_y = left_y
        dd_x = left_x + title_surf.get_width() + 12
        self.dd_measure.set_relative_position((dd_x, dd_y))
        dd_x += self.dd_measure.rect.width + 12
        self.dd_section.set_relative_position((dd_x, dd_y))
        dd_x += self.dd_section.rect.width + 12
        self.dd_pass.set_relative_position((dd_x, dd_y))
        # Score
        score_val = analytics.get('score', None).get('overall', None) if analytics and analytics.get('score') else None
        if score_val is not None:
            score_text = f"Overall Score: {score_val * 100:.0f}%"
        else:
            score_text = "Overall Score: —"
        score_color = (80, 200, 120) if score_val and score_val >= 0.8 else (255, 120, 120)
        score_surf = self.big_font.render(score_text, True, score_color)
        score_x = left_x
        score_y = left_y + title_surf.get_height() + 8
        surface.blit(score_surf, (score_x, score_y))
        # Calculate available height for radar and pianoroll
        used_height = title_surf.get_height() + 8 + score_surf.get_height() + 8
        available_h = left_h - used_height
        min_radar_h = 120
        min_pianoroll_h = 120
        # Split available space: 60% radar, 40% pianoroll, but respect minimums
        radar_h = max(min_radar_h, int(available_h * 0.6))
        pianoroll_h = max(min_pianoroll_h, available_h - radar_h)
        # If not enough space, shrink both equally but not below minimum
        total_needed = radar_h + pianoroll_h
        if total_needed > available_h:
            excess = total_needed - available_h
            shrink = excess // 2
            radar_h = max(min_radar_h, radar_h - shrink)
            pianoroll_h = max(min_pianoroll_h, pianoroll_h - shrink)
        radar_x = left_x
        radar_y = left_y + used_height
        self.img_spider.set_dimensions((dims['radar_w'], radar_h))
        self.img_spider.set_relative_position((radar_x, radar_y))
        pianoroll_x = left_x
        pianoroll_y = radar_y + radar_h
        self.img_pianoroll.set_dimensions((dims['pianoroll_w'], pianoroll_h))
        self.img_pianoroll.set_relative_position((pianoroll_x, pianoroll_y))

    def draw(self, surface):
        if not self.visible:
            return
        sw, sh = surface.get_size()
        if self.ui_manager.window_resolution != (sw, sh):
            self.ui_manager.set_window_resolution((sw, sh))
        dims = self._get_layout_dimensions(sw, sh)
        self.width = dims['width']
        self.height = dims['height']
        x = (sw - self.width) // 2
        y = (sh - self.height) // 4

        pygame.draw.rect(surface, (30, 30, 30), (x, y, self.width, self.height), border_radius=16)
        pygame.draw.rect(surface, (200, 200, 200), (x, y, self.width, self.height), 2, border_radius=16)

        ui_was_none = self.dd_measure is None or self.dd_section is None or self.dd_pass is None or self.img_spider is None or self.img_pianoroll is None or self.tips_box is None
        if ui_was_none:
            self._build_ui(sw, sh)
            self._refresh_charts_and_tips()
        analytics = self._get_selected_analytics()
        self._draw_left_column(surface, dims, analytics)

        tips_x = x + dims['left_w'] + dims['margin']
        tips_y = y + dims['margin']
        self.tips_box.set_relative_position((tips_x, tips_y))
        self.tips_box.set_dimensions((dims['tips_w'], dims['tips_h']))

        now = pygame.time.get_ticks()
        dt = max(0, now - self._last_update_ms) / 1000.0
        self._last_update_ms = now
        self.ui_manager.update(dt)
        self.ui_manager.draw_ui(surface)

    def _matplotlib_spider_chart(self, analytics, width=300, height=200):
        labels = np.array(['Accuracy', 'Relative', 'Absolute', 'Legato'])
        values = np.array([
            analytics.get('accuracy', 0),
            analytics.get('relative', 0),
            analytics.get('absolute', 0),
            1.0 - analytics.get('legato_severity', 0)
        ])
        values = np.append(values, values[0])
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.append(angles, angles[0])
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2, color='#50c878')
        ax.fill(angles, values, alpha=0.25, color='#50c878')
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics', y=1.1, color='white')
        ax.grid(True, color='white', alpha=0.3)
        ax.tick_params(colors='white')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color='white', fontsize=10)
        fig.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True)
        plt.close(fig)
        buf.seek(0)
        img = pygame.image.load(buf, 'spider_chart.png').convert_alpha()
        return img

    def _render_pianoroll(self, surface, rect):
        """Draws a pianoroll comparison of expected vs performed notes."""
        measure_data = self.teacher.midi_teacher.get_notes_for_measure(
            self._selected_measure
        )
        expected_chords, expected_times, _, _, _, expected_midi_msgs = measure_data
        performed_notes = self.teacher.midi_teacher.get_performed_notes_for_measure(
            self._selected_measure, self._selected_section, self._selected_pass
        )
        pprint(expected_midi_msgs)
        pprint(performed_notes)
        # Background
        pygame.draw.rect(surface, (30, 30, 30), rect)
        pygame.draw.rect(surface, (80, 80, 80), rect, 1)
        if not expected_midi_msgs and not performed_notes:
            return
        all_notes = [msg.note for msg in expected_midi_msgs if msg.type in ("note_on", "note_off")]
        if performed_notes:
            all_notes += [msg.note for msg in performed_notes if msg.type in ("note_on", "note_off")]
        if not all_notes:
            return
        min_pitch, max_pitch = min(all_notes), max(all_notes)
        pitch_range = max_pitch - min_pitch + 1
        def pitch_to_y(note):
            rel = (note - min_pitch) / max(1, pitch_range)
            return rect.bottom - rel * rect.height
        # --- Draw expected notes (green) ---
        expected_onsets = {}
        for msg in expected_midi_msgs:
            if msg.type == "note_on" and msg.velocity > 0:
                expected_onsets[msg.note] = getattr(msg, "time", 0)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                onset = expected_onsets.pop(msg.note, None)
                if onset is not None:
                    x1 = rect.left + (onset / max(1, rect.width))
                    x2 = rect.left + (getattr(msg, "time", 0) / max(1, rect.width))
                    y = pitch_to_y(msg.note)
                    pygame.draw.rect(surface, (0, 200, 0), pygame.Rect(x1, y - 4, max(1, x2 - x1), 8))
        # --- Draw performed notes (red) ---
        performed_onsets = {}
        for msg in performed_notes:
            if msg.type == "note_on" and msg.velocity > 0:
                performed_onsets[msg.note] = getattr(msg, "time", 0)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                onset = performed_onsets.pop(msg.note, None)
                if onset is not None:
                    x1 = rect.left + (onset / max(1, rect.width))
                    x2 = rect.left + (getattr(msg, "time", 0) / max(1, rect.width))
                    y = pitch_to_y(msg.note)
                    pygame.draw.rect(surface, (200, 0, 0), pygame.Rect(x1, y - 4, max(1, x2 - x1), 8))
