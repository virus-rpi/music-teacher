import json
from collections import defaultdict
import pygame
import io
import numpy as np
import matplotlib
from save_system import SaveSystem
import re
from pygame_gui.core import ObjectID
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
        self.window = None
        self.dd_measure = None
        self.dd_section = None
        self.dd_pass = None
        self.img_bar = None
        self.img_spider = None
        self.tips_box = None
        self._last_update_ms = pygame.time.get_ticks()
        self.pass_map = None
        self._init_selector()
        self._selected_measure = None
        self._selected_section = None
        self._selected_pass = None

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self._init_selector()
        else:
            self._reset_ui_elements()

    def _reset_ui_elements(self):
        for attr in ['window', 'dd_measure', 'dd_section', 'dd_pass', 'img_bar', 'img_spider', 'tips_box']:
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
        width = int(sw * 0.9)
        height = int(sh * 0.85)
        pad_side = int(width * 0.1)
        chart_w = int(width * 0.6)
        chart_h = int(height * 0.28)
        tips_w = width - chart_w - pad_side * 2 - 10
        tips_h = int(height * 0.6)
        return {
            'width': width,
            'height': height,
            'pad_side': pad_side,
            'chart_w': chart_w,
            'chart_h': chart_h,
            'tips_w': tips_w,
            'tips_h': tips_h
        }

    def _build_ui(self, sw, sh):
        dims = self._get_layout_dimensions(sw, sh)
        self.width = dims['width']
        self.height = dims['height']
        chart_w = dims['chart_w']
        chart_h = dims['chart_h']
        tips_w = dims['tips_w']
        tips_h = dims['tips_h']
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
        self.img_bar = pygame_gui.elements.UIImage(
            relative_rect=pygame.Rect(0, 0, chart_w, chart_h),
            image_surface=pygame.Surface((chart_w, chart_h), pygame.SRCALPHA),
            manager=self.ui_manager,
            object_id='#analytics_img_bar'
        )
        self.img_spider = pygame_gui.elements.UIImage(
            relative_rect=pygame.Rect(0, 0, chart_w, chart_h),
            image_surface=pygame.Surface((chart_w, chart_h), pygame.SRCALPHA),
            manager=self.ui_manager,
            object_id='#analytics_img_spider'
        )
        self.tips_box = pygame_gui.elements.UITextBox(
            html_text='',
            relative_rect=pygame.Rect(0, 0, tips_w, tips_h),
            manager=self.ui_manager,
            object_id='#analytics_tips_box'
        )
        self._refresh_charts_and_tips()

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
        if self.window is None:
            return
        bar_rect = self.img_bar.rect
        spider_rect = self.img_spider.rect
        content_w = bar_rect.width
        bar_h = bar_rect.height
        spider_h = spider_rect.height
        if analytics:
            spider_chart = self._matplotlib_spider_chart(analytics, width=content_w, height=spider_h)
            self.img_spider.set_image(spider_chart)
            tips_html = self._generate_tips_html(analytics)
            self.tips_box.set_text(tips_html)
        else:
            self.img_bar.set_image(pygame.Surface((content_w, bar_h), pygame.SRCALPHA))
            self.img_spider.set_image(pygame.Surface((content_w, spider_h), pygame.SRCALPHA))
            self.tips_box.set_text('<b>No analytics available yet.</b>')

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

    def draw(self, surface):
        if not self.visible:
            return
        if self.font is None:
            self.font = pygame.font.SysFont('Arial', 22)
            self.small_font = pygame.font.SysFont('Arial', 16)
        sw, sh = surface.get_size()
        if self.ui_manager.window_resolution != (sw, sh):
            self.ui_manager.set_window_resolution((sw, sh))
        dims = self._get_layout_dimensions(sw, sh)
        self.width = dims['width']
        self.height = dims['height']
        pad = 32
        pad_top = pad
        pad_side = pad
        x = (sw - self.width) // 2
        y = (sh - self.height) // 4
        pygame.draw.rect(surface, (30, 30, 30), (x, y, self.width, self.height), border_radius=16)
        pygame.draw.rect(surface, (200, 200, 200), (x, y, self.width, self.height), 2, border_radius=16)
        if self.dd_measure is None or self.dd_section is None or self.dd_pass is None or self.img_spider is None or self.tips_box is None:
            self._build_ui(sw, sh)
        title_surf = self.font.render('Performance Analytics for', True, (255, 255, 255))
        title_x = x + pad_side
        title_y = y + pad_top
        surface.blit(title_surf, (title_x, title_y))
        dd_y = title_y
        dd_x = title_x + title_surf.get_width() + 12
        self.dd_measure.set_relative_position((dd_x, dd_y))
        dd_x += self.dd_measure.rect.width + 12
        self.dd_section.set_relative_position((dd_x, dd_y))
        dd_x += self.dd_section.rect.width + 12
        self.dd_pass.set_relative_position((dd_x, dd_y))
        chart_pad_top = pad_top + 40
        chart_x = x + pad_side
        chart_w = dims['chart_w']
        chart_h = dims['chart_h']
        self.img_bar.set_dimensions((chart_w, chart_h))
        self.img_bar.set_relative_position((chart_x, y + chart_pad_top))
        spider_y = y + chart_pad_top + chart_h + 20
        self.img_spider.set_dimensions((chart_w, chart_h))
        self.img_spider.set_relative_position((chart_x, spider_y))
        tips_x = chart_x + chart_w + 24
        tips_y = y + chart_pad_top
        tips_w = dims['tips_w']
        tips_h = dims['tips_h']
        self.tips_box.set_relative_position((tips_x, tips_y))
        self.tips_box.set_dimensions((tips_w, tips_h))
        now = pygame.time.get_ticks()
        dt = max(0, now - self._last_update_ms) / 1000.0
        self._last_update_ms = now
        self.ui_manager.update(dt)
        self.ui_manager.draw_ui(surface)

    @staticmethod
    def _matplotlib_spider_chart(analytics, width=300, height=200):
        labels = np.array(['Accuracy', 'Relative', 'Absolute', 'Legato'])
        legato_score = 1.0 - analytics.get('legato_severity', 0)
        values = np.array([
            analytics.get('accuracy', 0),
            analytics.get('relative', 0),
            analytics.get('absolute', 0),
            legato_score
        ])
        values = np.append(values, values[0])
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.append(angles, angles[0])
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2, color='#50c878')
        ax.fill(angles, values, alpha=0.25, color='#50c878')
        ax.set_ylim(0, 1)
        ax.set_title('Radar (Spider) Graph', y=1.1, color='white')
        ax.grid(True, color='white', alpha=0.3)
        ax.tick_params(colors='white')
        fig.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True)
        plt.close(fig)
        buf.seek(0)
        img = pygame.image.load(buf, 'spider_chart.png').convert_alpha()
        return img

    def _draw_per_chord(self, surface, x, y, analytics, width=400, height=60):
        per = analytics.get('per_chord', []) if analytics else []
        interval_diffs = analytics.get('interval_diffs', []) if analytics else []
        n = len(per)
        if not per or n == 0:
            return
        max_w = width
        ref_bar_h = int(height * 0.7)
        ref_bar_w = max(8, min(30, max_w // max(1, n)))
        timing_accuracies = [1.0 - min(1.0, abs(d)) for d in interval_diffs] + [1.0]
        timing_accuracies = [max(0.0, min(1.0, t)) for t in timing_accuracies]
        for i in range(n):
            bx = x + i * (ref_bar_w + 2)
            by = y
            ref_rect = pygame.Rect(bx, by, ref_bar_w, ref_bar_h)
            pygame.draw.rect(surface, (180, 180, 180), ref_rect)
        for i, info in enumerate(per):
            acc = info.get('score', 0)
            timing = timing_accuracies[i] if i < len(timing_accuracies) else 1.0
            bar_h = int(acc * ref_bar_h)
            bar_w = int(ref_bar_w * timing)
            bx = x + i * (ref_bar_w + 2) + (ref_bar_w - bar_w) // 2
            by = y + ref_bar_h - bar_h
            user_surf = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
            color = (80, 200, 120, 50) if acc > 0.8 else (255, 120, 120, 50)
            user_surf.fill(color)
            surface.blit(user_surf, (bx, by))
            idx_txt = self.small_font.render(str(i + 1), True, (255, 255, 255))
            surface.blit(idx_txt, (x + i * (ref_bar_w + 2) + 2, y + ref_bar_h + 4))
        label = self.small_font.render('Per-chord accuracy (height=accuracy, width=timing)', True, (255, 255, 255))
        surface.blit(label, (x, y + ref_bar_h + 24))
