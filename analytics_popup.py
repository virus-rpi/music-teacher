import pygame
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Dropdown:
    def __init__(self, options, value=0, width=150, font=None, label_prefix=None):
        self.options = options
        self.value = value
        self.open = False
        self.rect = None
        self.option_rects = []
        self.width = width
        self.font = font or pygame.font.SysFont('Arial', 22)
        self.label_prefix = label_prefix
        self._last_options = list(options)

    def tick(self, event, close_others_callback=None):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if self.rect and self.rect.collidepoint(mouse_pos):
                if not self.open and close_others_callback:
                    close_others_callback(self)
                self.open = not self.open
                return True
            if self.open:
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(mouse_pos):
                        self.value = i
                        self.open = False
                        return True
                # Click outside closes
                self.open = False
        elif event.type == pygame.KEYDOWN and self.open:
            if event.key == pygame.K_ESCAPE:
                self.open = False
        return False

    def render_button(self, surface, pos):
        font = self.font
        dropdown_h = font.get_height() + 8
        dropdown_w = self.width
        rect = pygame.Rect(pos, (dropdown_w, dropdown_h))
        self.rect = rect
        pygame.draw.rect(surface, (40, 40, 40), rect, border_radius=6)
        pygame.draw.rect(surface, (80, 80, 80), rect, 2, border_radius=6)
        label = self.options[self.value] if self.options else '—'
        if self.label_prefix:
            label = f"{self.label_prefix} {label}"
        label_surf = font.render(label, True, (255, 255, 255))
        surface.blit(label_surf, (rect.x + 8, rect.y + (dropdown_h - label_surf.get_height()) // 2))
        chevron_x = rect.right - 20
        chevron_y = rect.centery
        pygame.draw.polygon(surface, (200, 200, 200), [
            (chevron_x, chevron_y - 5), (chevron_x + 10, chevron_y - 5), (chevron_x + 5, chevron_y + 4)
        ])

    def render_options(self, surface):
        if not self.open or not self.options or not self.rect:
            self.option_rects = []
            return
        font = self.font
        dropdown_h = font.get_height() + 8
        dropdown_w = self.width
        self.option_rects = []
        for i, option in enumerate(self.options):
            opt_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * dropdown_h, dropdown_w, dropdown_h)
            pygame.draw.rect(surface, (40, 40, 40), opt_rect, border_radius=6)
            pygame.draw.rect(surface, (80, 80, 80), opt_rect, 2, border_radius=6)
            opt_label = option
            if self.label_prefix:
                opt_label = f"{self.label_prefix} {option}"
            opt_surf = font.render(opt_label, True, (255, 255, 255))
            surface.blit(opt_surf, (opt_rect.x + 8, opt_rect.y + (dropdown_h - opt_surf.get_height()) // 2))
            self.option_rects.append(opt_rect)

    def render(self, surface, pos):
        self.render_button(surface, pos)
        self.render_options(surface)

class AnalyticsPopup:
    def __init__(self, teacher):
        self.teacher = teacher
        self.visible = False
        self.margin = 30
        self.font = None
        self.small_font = None
        self.width = None
        self.height = None
        self.measure_dropdown = None
        self.section_dropdown = None
        self.pass_dropdown = None
        self._init_selector()

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self._init_selector()

    def _init_selector(self):
        hist = self.teacher.evaluator_history
        measure_sections = {}
        for key in hist.keys():
            parts = key.split('_')
            measure = parts[0]
            measure_sections.setdefault(measure, []).append(key)
        self.measure_list = sorted(measure_sections.keys(), key=lambda m: int(m))
        self.section_dict = {m: sorted(measure_sections[m], key=lambda k: hist[k].get('scores', [-1])[-1] if hist[k].get('scores') else -1) for m in self.measure_list}
        if self.measure_list:
            measure_idx = len(self.measure_list) - 1
            measure = self.measure_list[measure_idx]
            section_idx = len(self.section_dict[measure]) - 1
        else:
            measure_idx = 0
            section_idx = 0
        self.measure_dropdown = Dropdown(self.measure_list, value=measure_idx, width=150, label_prefix="Measure")
        section_options = self.section_dict[self.measure_list[measure_idx]] if self.measure_list else []
        self.section_dropdown = Dropdown([f"Section {i+1}" for i in range(len(section_options))], value=section_idx, width=150, label_prefix=None)
        self.pass_dropdown = Dropdown(["Pass 1"], value=0, width=100, label_prefix=None)
        self._update_pass_dropdown()

    def _update_section_dropdown(self):
        measure_idx = self.measure_dropdown.value
        measure = self.measure_list[measure_idx] if self.measure_list else None
        section_options = self.section_dict[measure] if measure else []
        self.section_dropdown.options = [f"Section {i+1}" for i in range(len(section_options))]
        if self.section_dropdown.value >= len(self.section_dropdown.options):
            self.section_dropdown.value = max(0, len(self.section_dropdown.options)-1)

    def _update_pass_dropdown(self):
        measure_idx = self.measure_dropdown.value
        section_idx = self.section_dropdown.value
        measure = self.measure_list[measure_idx] if self.measure_list else None
        section_keys = self.section_dict[measure] if measure else []
        pass_count = 1
        if section_keys and section_idx < len(section_keys):
            key = section_keys[section_idx]
            hist = self.teacher.evaluator_history
            entry = hist.get(key, {})
            analytics_list = entry.get('analytics', [])
            pass_count = len(analytics_list)
        self.pass_dropdown.options = [f"Pass {i+1}" for i in range(pass_count)]
        if self.pass_dropdown.value >= pass_count:
            self.pass_dropdown.value = max(0, pass_count-1)

    def handle_event(self, event):
        if not self.visible or not self.measure_list:
            return
        def close_others(open_dropdown):
            for d in [self.measure_dropdown, self.section_dropdown, self.pass_dropdown]:
                if d is not open_dropdown:
                    d.open = False
        changed = self.measure_dropdown.tick(event, close_others)
        if changed:
            self._update_section_dropdown()
            self.section_dropdown.value = 0
            self._update_pass_dropdown()
            self.pass_dropdown.value = 0
            return
        changed = self.section_dropdown.tick(event, close_others)
        if changed:
            self._update_pass_dropdown()
            self.pass_dropdown.value = 0
            return
        self.pass_dropdown.tick(event, close_others)

    def _get_selected_key(self):
        if not self.measure_list:
            return None
        measure = self.measure_list[self.measure_dropdown.value]
        section_keys = self.section_dict[measure]
        if not section_keys:
            return None
        if self.section_dropdown.value >= len(section_keys):
            return None
        return section_keys[self.section_dropdown.value]

    def _get_selected_analytics(self):
        key = self._get_selected_key()
        if not key:
            return None
        hist = self.teacher.evaluator_history
        entry = hist.get(key, {})
        analytics_list = entry.get('analytics', [])
        if analytics_list:
            idx = min(self.pass_dropdown.value, len(analytics_list) - 1)
            return analytics_list[idx]
        return None

    def draw(self, surface):
        if not self.visible:
            return
        if self.font is None:
            self.font = pygame.font.SysFont('Arial', 22)
            self.small_font = pygame.font.SysFont('Arial', 16)
            self.measure_dropdown.font = self.font
            self.section_dropdown.font = self.font
            self.pass_dropdown.font = self.font
        overlay = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        sw, sh = overlay.get_size()
        self.width = int(sw * 0.95)
        self.height = int(sh * 0.85)
        pad = 32
        pad_top = pad
        pad_side = pad
        x = (sw - self.width) // 2
        y = (sh - self.height) // 4
        pygame.draw.rect(overlay, (0, 0, 0, 180), (0, 0, sw, sh))
        pygame.draw.rect(overlay, (30, 30, 30), (x, y, self.width, self.height), border_radius=16)
        pygame.draw.rect(overlay, (200, 200, 200), (x, y, self.width, self.height), 2, border_radius=16)
        left_w = int(self.width * 2 / 3)
        left_x = x + pad_side
        content_w = left_w - pad_side * 2
        content_h = self.height - pad_top * 2
        curr_y = y + pad_top
        curr_x = left_x
        title_txt = self.font.render('Performance Analytics for ', True, (255,255,255))
        title_h = title_txt.get_height()
        overlay.blit(title_txt, (curr_x, curr_y))
        curr_x += title_txt.get_width()
        self.measure_dropdown.render_button(overlay, (curr_x, curr_y))
        curr_x += self.measure_dropdown.rect.width + 10
        self.section_dropdown.render_button(overlay, (curr_x, curr_y))
        curr_x += self.section_dropdown.rect.width + 10
        if len(self.pass_dropdown.options) > 1:
            self.pass_dropdown.render_button(overlay, (curr_x, curr_y))
        analytics = self._get_selected_analytics()
        if analytics:
            bar_chart = self._matplotlib_bar_chart(analytics, width=content_w, height=content_h // 3)
            overlay.blit(bar_chart, (left_x, curr_y + title_h + 20))
            spider_chart = self._matplotlib_spider_chart(analytics, width=content_w, height=content_h // 3)
            overlay.blit(spider_chart, (left_x, curr_y + title_h + 20 + content_h // 3 + 10))
            self._draw_tips(overlay, left_x, curr_y + title_h + 20 + 2 * (content_h // 3) + 20, analytics, max_height=content_h // 3)
        # Render open dropdown options on top
        for dropdown in [self.measure_dropdown, self.section_dropdown, self.pass_dropdown]:
            if dropdown.open:
                dropdown.render_options(overlay)
        surface.blit(overlay, (0, 0))

    def _draw_explanations(self, surface, x, y):
        explanations = [
            ('Accuracy', 'How accurate you pressed the correct notes.'),
            ('Relative', 'How well your timing is aligned with the reference relative to your tempo.'),
            ('Absolute', 'How close your tempo is to the reference tempo.'),
            ('Legato', 'Ratio between the average note overlap difference to the reference compared to the average note length.'),
        ]
        for i, (label, desc) in enumerate(explanations):
            txt = self.small_font.render(f'{label}: {desc}', True, (255,255,255))
            surface.blit(txt, (x, y + i*20))

    def _get_last_analytics(self):
        hist = self.teacher.evaluator_history
        if not hist:
            return None
        last = None
        for k in sorted(hist.keys(), key=lambda k: hist[k].get('scores', [-1])[-1] if hist[k].get('scores') else -1):
            last = hist[k]
        if last and 'analytics' in last and last['analytics']:
            return last['analytics'][-1]
        return None

    def _matplotlib_bar_chart(self, analytics, width=300, height=200):
        labels = ['Accuracy', 'Relative', 'Absolute', 'Legato']
        legato_score = 1.0 - analytics.get('legato_severity', 0)
        values = [analytics.get('accuracy', 0), analytics.get('relative', 0), analytics.get('absolute', 0), legato_score]
        colors = ['#50c878', '#78b4ff', '#ffc850', '#c878ff']
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title('Score Breakdown')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=10, color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        fig.patch.set_alpha(0)
        ax.set_facecolor((0,0,0,0))
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True)
        plt.close(fig)
        buf.seek(0)
        img = pygame.image.load(buf, 'bar_chart.png').convert_alpha()
        return img

    def _matplotlib_spider_chart(self, analytics, width=300, height=200):
        labels = np.array(['Accuracy', 'Relative', 'Absolute', 'Legato'])
        legato_score = 1.0 - analytics.get('legato_severity', 0)
        values = np.array([
            analytics.get('accuracy', 0),
            analytics.get('relative', 0),
            analytics.get('absolute', 0),
            legato_score
        ])
        values = np.append(values, values[0])  # close the loop
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.append(angles, angles[0])
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2, color='#50c878')
        ax.fill(angles, values, alpha=0.25, color='#50c878')
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, color='white')
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

    def _draw_tips(self, surface, x, y, analytics, max_height=100):
        tips = []
        if not analytics:
            return
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
        line_h = self.small_font.get_height() + 2
        max_lines = max_height // line_h
        for i, tip in enumerate(tips[:max_lines]):
            tiptxt = self.small_font.render(tip, True, (255, 255, 255))
            surface.blit(tiptxt, (x, y + i * line_h))

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

    def _draw_dropdowns(self, surface):
        if not self.visible:
            return
        if self.font is None:
            self.font = pygame.font.SysFont('Arial', 22)
        dropdown_h = self.measure_dropdown_rect.height if hasattr(self, 'measure_dropdown_rect') else 32
        # Draw pass dropdown options
        if self.dropdown_open == 'pass' and self.pass_dropdown_rect and self.pass_option_rects:
            for i, opt_bg in enumerate(self.pass_option_rects):
                opt_label = f"Pass {i+1}"
                opt_surf = self.font.render(opt_label, True, (255,255,255))
                pygame.draw.rect(surface, (40,40,40), opt_bg, border_radius=6)
                pygame.draw.rect(surface, (80,80,80), opt_bg, 2, border_radius=6)
                surface.blit(opt_surf, (opt_bg.x+16, opt_bg.y + (self.pass_dropdown_rect.height - opt_surf.get_height())//2))
        # Draw measure dropdown options
        if self.dropdown_open == 'measure' and hasattr(self, 'measure_option_rects'):
            for i, m in enumerate(self.measure_list):
                opt_label = f"Measure {m}"
                opt_surf = self.font.render(opt_label, True, (255,255,255))
                opt_bg = self.measure_option_rects[i]
                pygame.draw.rect(surface, (40,40,40), opt_bg, border_radius=6)
                pygame.draw.rect(surface, (80,80,80), opt_bg, 2, border_radius=6)
                surface.blit(opt_surf, (opt_bg.x+16, opt_bg.y + (self.measure_dropdown_rect.height - opt_surf.get_height())//2))
        if self.dropdown_open == 'section' and hasattr(self, 'section_option_rects'):
            measure = self.measure_list[self.selected_measure_idx] if self.measure_list else '—'
            section_keys = self.section_dict[measure] if self.measure_list else []
            for i, key in enumerate(section_keys):
                opt_label = f"Section {i+1}"
                opt_surf = self.font.render(opt_label, True, (255,255,255))
                opt_bg = self.section_option_rects[i]
                pygame.draw.rect(surface, (40,40,40), opt_bg, border_radius=6)
                pygame.draw.rect(surface, (80,80,80), opt_bg, 2, border_radius=6)
                surface.blit(opt_surf, (opt_bg.x+16, opt_bg.y + (self.section_dropdown_rect.height - opt_surf.get_height())//2))
