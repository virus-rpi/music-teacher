import pygame
import math
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class AnalyticsPopup:
    def __init__(self, teacher):
        self.height = None
        self.width = None
        self.teacher = teacher
        self.visible = False
        self.margin = 30
        self.font = None
        self.small_font = None
        self.selected_key = None
        self.measure_keys = []
        self.selected_measure_idx = 0
        self.selected_section_idx = 0
        self.dropdown_open = None  # None, 'measure', or 'section'

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self._init_selector()

    def _init_selector(self):
        hist = self.teacher.evaluator_history
        # Parse measure/section from keys
        measure_sections = {}
        for key in hist.keys():
            parts = key.split('_')
            measure = parts[0]
            measure_sections.setdefault(measure, []).append(key)
        self.measure_list = sorted(measure_sections.keys(), key=lambda m: int(m))
        self.section_dict = {m: sorted(measure_sections[m], key=lambda k: hist[k].get('scores', [-1])[-1] if hist[k].get('scores') else -1) for m in self.measure_list}
        # Default to last measure/section
        if self.measure_list:
            self.selected_measure_idx = len(self.measure_list) - 1
            measure = self.measure_list[self.selected_measure_idx]
            self.selected_section_idx = len(self.section_dict[measure]) - 1
        else:
            self.selected_measure_idx = 0
            self.selected_section_idx = 0
        self.dropdown_open = None

    def _get_selected_key(self):
        if not self.measure_list:
            return None
        measure = self.measure_list[self.selected_measure_idx]
        section_keys = self.section_dict[measure]
        if not section_keys:
            return None
        return section_keys[self.selected_section_idx]

    def _get_selected_analytics(self):
        key = self._get_selected_key()
        if not key:
            return None
        hist = self.teacher.evaluator_history
        entry = hist.get(key, {})
        analytics_list = entry.get('analytics', [])
        if analytics_list:
            return analytics_list[-1]
        return None

    def handle_event(self, event):
        if not self.visible or not self.measure_list:
            return
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if hasattr(self, 'measure_dropdown_rect') and self.measure_dropdown_rect.collidepoint(mx, my):
                self.dropdown_open = 'measure' if self.dropdown_open != 'measure' else None
                return
            if hasattr(self, 'section_dropdown_rect') and self.section_dropdown_rect.collidepoint(mx, my):
                self.dropdown_open = 'section' if self.dropdown_open != 'section' else None
                return
            if self.dropdown_open == 'measure':
                for i, rect in enumerate(self.measure_option_rects):
                    if rect.collidepoint(mx, my):
                        self.selected_measure_idx = i
                        self.selected_section_idx = 0
                        self.dropdown_open = None
                        return
            if self.dropdown_open == 'section':
                measure = self.measure_list[self.selected_measure_idx]
                for i, rect in enumerate(self.section_option_rects):
                    if rect.collidepoint(mx, my):
                        self.selected_section_idx = i
                        self.dropdown_open = None
                        return
            self.dropdown_open = None
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.dropdown_open = None

    def draw(self, surface):
        if not self.visible:
            return
        if self.font is None:
            self.font = pygame.font.SysFont('Arial', 22)
            self.small_font = pygame.font.SysFont('Arial', 16)
        sw, sh = surface.get_size()
        self.width = int(sw * 0.75)
        self.height = int(sh * 0.75)
        pad = int(self.width * 0.05)
        pad_top = pad
        pad_side = pad
        x = (sw - self.width) // 2
        y = (sh - self.height) // 2
        # Draw background
        pygame.draw.rect(surface, (30, 30, 30), (x, y, self.width, self.height), border_radius=16)
        pygame.draw.rect(surface, (200, 200, 200), (x, y, self.width, self.height), 2, border_radius=16)

        # Layout
        left_w = int(self.width * 2 / 3)
        right_w = self.width - left_w
        left_x = x + pad_side
        right_x = x + left_w + pad_side // 2
        content_w = left_w - pad_side * 2
        content_h = self.height - pad_top * 2
        # Title with dropdowns
        curr_y = y + pad_top
        curr_x = left_x
        title_txt = self.font.render('Performance Analytics for ', True, (255,255,255))
        title_h = title_txt.get_height()
        surface.blit(title_txt, (curr_x, curr_y))
        curr_x += title_txt.get_width()
        # Measure dropdown (dark mode, wider, vertically centered)
        measure = self.measure_list[self.selected_measure_idx] if self.measure_list else '—'
        measure_label = f"Measure {measure}"
        measure_surf = self.font.render(measure_label, True, (255,255,255))
        dropdown_h = max(measure_surf.get_height(), title_h) + 8
        dropdown_w = measure_surf.get_width() + 36
        measure_bg = pygame.Rect(curr_x, curr_y + (title_h - dropdown_h)//2, dropdown_w, dropdown_h)
        pygame.draw.rect(surface, (40,40,40), measure_bg, border_radius=6)
        pygame.draw.rect(surface, (80,80,80), measure_bg, 2, border_radius=6)
        surface.blit(measure_surf, (curr_x+8, measure_bg.y + (dropdown_h - measure_surf.get_height())//2))
        # Down chevron
        chevron_x = curr_x + dropdown_w - 20
        chevron_y = measure_bg.y + dropdown_h//2
        pygame.draw.polygon(surface, (200,200,200), [
            (chevron_x, chevron_y-5), (chevron_x+10, chevron_y-5), (chevron_x+5, chevron_y+4)
        ])
        self.measure_dropdown_rect = measure_bg
        curr_x += dropdown_w + 10
        # Dash
        dash_surf = self.font.render('-', True, (255,255,255))
        surface.blit(dash_surf, (curr_x, curr_y))
        curr_x += dash_surf.get_width() + 10
        # Section dropdown (dark mode, wider, vertically centered)
        measure = self.measure_list[self.selected_measure_idx] if self.measure_list else '—'
        section_keys = self.section_dict[measure] if self.measure_list else []
        section_idx = self.selected_section_idx
        section_label = f"Section {section_idx+1}" if section_keys else '—'
        section_surf = self.font.render(section_label, True, (255,255,255))
        section_dropdown_w = section_surf.get_width() + 36
        section_bg = pygame.Rect(curr_x, curr_y + (title_h - dropdown_h)//2, section_dropdown_w, dropdown_h)
        pygame.draw.rect(surface, (40,40,40), section_bg, border_radius=6)
        pygame.draw.rect(surface, (80,80,80), section_bg, 2, border_radius=6)
        surface.blit(section_surf, (curr_x+8, section_bg.y + (dropdown_h - section_surf.get_height())//2))
        chevron_x = curr_x + section_dropdown_w - 20
        chevron_y = section_bg.y + dropdown_h//2
        pygame.draw.polygon(surface, (200,200,200), [
            (chevron_x, chevron_y-5), (chevron_x+10, chevron_y-5), (chevron_x+5, chevron_y+4)
        ])
        self.section_dropdown_rect = section_bg
        curr_x += section_dropdown_w + 10
        self.measure_option_rects = []
        self.section_option_rects = []
        if self.dropdown_open == 'measure':
            for i, m in enumerate(self.measure_list):
                opt_bg = pygame.Rect(self.measure_dropdown_rect.x, self.measure_dropdown_rect.bottom + i*dropdown_h, self.measure_dropdown_rect.width, dropdown_h)
                self.measure_option_rects.append(opt_bg)
        if self.dropdown_open == 'section':
            for i, key in enumerate(section_keys):
                opt_bg = pygame.Rect(self.section_dropdown_rect.x, self.section_dropdown_rect.bottom + i*dropdown_h, self.section_dropdown_rect.width, dropdown_h)
                self.section_option_rects.append(opt_bg)
        curr_y += title_h + 10
        analytics = self._get_selected_analytics()
        if analytics and 'accuracy' in analytics and 'relative' in analytics and 'absolute' in analytics:
            overall = 0.6 * analytics['accuracy'] + 0.25 * analytics['relative'] + 0.15 * analytics['absolute']
            score_txt = self.font.render(f"Overall Score: {overall*100:.1f}%", True, (255,255,255))
            surface.blit(score_txt, (left_x, curr_y))
            curr_y += score_txt.get_height() + int(self.height * 0.01)
        bar_h = int(content_h * 0.3)
        bar_chart_surf = self._matplotlib_bar_chart(analytics=analytics, width=content_w, height=bar_h)
        surface.blit(bar_chart_surf, (left_x, curr_y))
        curr_y += bar_h + int(self.height * 0.02)
        # Tips (fit in 15% of available height)
        tips_h = int(content_h * 0.15)
        self._draw_tips(surface, left_x, curr_y, analytics, max_height=tips_h)
        curr_y += tips_h + int(self.height * 0.01)
        # Per-chord accuracy (30% of available height)
        per_h = int(content_h * 0.3)
        self._draw_per_chord(surface, left_x, curr_y, analytics, width=content_w, height=per_h)
        curr_y += per_h + int(self.height * 0.01)
        # Explanations (rest of the space)
        self._draw_explanations(surface, left_x, curr_y)
        # Spider graph on right 1/3, all available space
        spider_h = self.height - pad_top * 2
        spider_w = right_w - pad_side
        spider_surf = self._matplotlib_spider_chart(analytics, width=spider_w, height=spider_h)
        surface.blit(spider_surf, (right_x, y + pad_top))
        # Close hint
        hint = self.small_font.render('Press TAB to close', True, (255, 255, 255))
        surface.blit(hint, (x + self.width - pad_side - 200, y + self.height - pad_top - 40))
        # Draw dropdowns last so they are above the rest
        self._draw_dropdowns(surface)

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
        if self.dropdown_open == 'measure' and hasattr(self, 'measure_option_rects'):
            for i, m in enumerate(self.measure_list):
                opt_label = f"Measure {m}"
                opt_surf = self.font.render(opt_label, True, (255,255,255))
                opt_bg = self.measure_option_rects[i]
                pygame.draw.rect(surface, (40,40,40), opt_bg, border_radius=6)
                pygame.draw.rect(surface, (80,80,80), opt_bg, 2, border_radius=6)
                surface.blit(opt_surf, (opt_bg.x+16, opt_bg.y + (dropdown_h - opt_surf.get_height())//2))
        # Draw section dropdown options
        if self.dropdown_open == 'section' and hasattr(self, 'section_option_rects'):
            measure = self.measure_list[self.selected_measure_idx] if self.measure_list else '—'
            section_keys = self.section_dict[measure] if self.measure_list else []
            for i, key in enumerate(section_keys):
                opt_label = f"Section {i+1}"
                opt_surf = self.font.render(opt_label, True, (255,255,255))
                opt_bg = self.section_option_rects[i]
                pygame.draw.rect(surface, (40,40,40), opt_bg, border_radius=6)
                pygame.draw.rect(surface, (80,80,80), opt_bg, 2, border_radius=6)
                surface.blit(opt_surf, (opt_bg.x+16, opt_bg.y + (dropdown_h - opt_surf.get_height())//2))
