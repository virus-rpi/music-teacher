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

    def toggle(self):
        self.visible = not self.visible

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
        # Title
        title = self.font.render('Performance Analytics', True, (255, 255, 255))
        surface.blit(title, (left_x, y + pad_top))
        curr_y = y + pad_top + title.get_height() + int(self.height * 0.01)
        # Overall score text
        analytics = self._get_last_analytics()
        if analytics and 'accuracy' in analytics and 'relative' in analytics and 'absolute' in analytics:
            # Use same weights as in evaluator.py
            overall = 0.6 * analytics['accuracy'] + 0.25 * analytics['relative'] + 0.15 * analytics['absolute']
            score_txt = self.font.render(f"Overall Score: {overall*100:.1f}%", True, (255,255,255))
            surface.blit(score_txt, (left_x, curr_y))
            curr_y += score_txt.get_height() + int(self.height * 0.01)
        # Score breakdown bar chart (30% of available height)
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
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2, color='#50c878')
        ax.fill(angles, values, alpha=0.25, color='#50c878')
        ax.set_thetagrids(angles[:-1] * 180/np.pi, labels, color='white')
        ax.set_ylim(0, 1)
        ax.set_title('Radar (Spider) Graph', y=1.1, color='white')
        ax.grid(True, color='white', alpha=0.3)
        ax.tick_params(colors='white')
        fig.patch.set_alpha(0)
        ax.set_facecolor((0,0,0,0))
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
            tips.append(f'Largest timing gap error between onsets {idx} and {idx+1}: {diff:.2f}')
        if analytics.get('tempo_bias') is not None:
            bias = analytics['tempo_bias']
            if abs(bias) > 0.02:
                tips.append(f'Overall tempo was {"faster" if bias>0 else "slower"} by {abs(bias)*100:.1f}%')
        if analytics.get('legato_detected'):
            sev = analytics.get('legato_severity', 0)
            tips.append(f'Legato severity {sev*100:.1f}% (lower is better)')
        if not tips:
            tips = ['No major issues detected.']
        line_h = self.small_font.get_height() + 2
        max_lines = max_height // line_h
        for i, tip in enumerate(tips[:max_lines]):
            tiptxt = self.small_font.render(tip, True, (255, 255, 255))
            surface.blit(tiptxt, (x, y + i*line_h))

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
            pygame.draw.rect(surface, (180,180,180), ref_rect)
        for i, info in enumerate(per):
            acc = info.get('score', 0)
            timing = timing_accuracies[i] if i < len(timing_accuracies) else 1.0
            bar_h = int(acc * ref_bar_h)
            bar_w = int(ref_bar_w * timing)
            bx = x + i * (ref_bar_w + 2) + (ref_bar_w - bar_w)//2
            by = y + ref_bar_h - bar_h
            user_surf = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
            color = (80,200,120, 50) if acc > 0.8 else (255,120,120, 50)
            user_surf.fill(color)
            surface.blit(user_surf, (bx, by))
            idx_txt = self.small_font.render(str(i+1), True, (255,255,255))
            surface.blit(idx_txt, (x + i * (ref_bar_w + 2) + 2, y+ref_bar_h+4))
        label = self.small_font.render('Per-chord accuracy (height=accuracy, width=timing)', True, (255,255,255))
        surface.blit(label, (x, y+ref_bar_h+24))
