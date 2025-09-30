import io
import json
import re
from collections import defaultdict
import matplotlib
import numpy as np
import pygame
from flexbox import FlexBox
from save_system import SaveSystem

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pygame_gui

# TODO: let user playback correct vs recorded midi to hear the difference
# TODO: add toggle for relative or absolute timing in the piano roll (separate or merged x normalization)

def _render_background(surface):
    pygame.draw.rect(surface, (30, 30, 30), (0, 0, surface.get_width(), surface.get_height()), border_radius=16)
    pygame.draw.rect(surface, (200, 200, 200), (0, 0, surface.get_width(), surface.get_height()), 2, border_radius=16)


def _generate_tips(analytics) -> pygame_gui.elements.UITextBox:
    tips = []
    technique_tips = []

    if not analytics:
        return pygame_gui.elements.UITextBox(
            html_text='<i>No analytics available.</i>',
            relative_rect=pygame.Rect(0, 0, 100, 10),
        )

    # Existing diagnostic tips
    if analytics.get('worst_chord_idx') is not None:
        idx = analytics['worst_chord_idx'] + 1
        score = analytics['worst_chord_score']
        tips.append(f'Chord {idx} had the lowest accuracy: {score:.2f}')

        # Add technique tip for chord accuracy
        if score < 0.7:
            technique_tips.append(f'For chord {idx}: Practice slowly with strong finger independence. Press all notes simultaneously and hold until the sound blends completely.')
        elif score < 0.9:
            technique_tips.append(f'For chord {idx}: Focus on even finger pressure and wrist stability. Practice the chord in isolation 10 times before playing in context.')

    if analytics.get('worst_interval_idx') is not None:
        idx = analytics['worst_interval_idx'] + 1
        diff = analytics['worst_interval_diff']
        tips.append(f'Largest timing gap error between onsets {idx} and {idx + 1}: {diff:.2f}')

        # Add technique tip for timing
        if abs(diff) > 0.1:
            technique_tips.append(f'Timing technique: Use a metronome and practice counting subdivisions. Try playing with exaggerated staccato first, then gradually connect the notes.')
        else:
            technique_tips.append(f'Fine timing: Practice with different rhythmic patterns (dotted, syncopated) to improve internal pulse accuracy.')

    if analytics.get('tempo_bias') is not None:
        bias = analytics['tempo_bias']
        if abs(bias) > 0.02:
            tips.append(f'Overall tempo was {"faster" if bias > 0 else "slower"} by {abs(bias) * 100:.1f}%')

            # Add technique tips for tempo control
            if bias > 0.05:  # rushing
                technique_tips.append('Tempo control: Practice with a metronome at 70% target tempo. Focus on feeling the "space" between beats. Try conducting with your free hand.')
            elif bias < -0.05:  # dragging
                technique_tips.append('Tempo energy: Imagine the music moving forward. Practice with slight accent on beat 1 of each measure. Keep your body engaged and avoid tension.')
            else:  # minor tempo issues
                technique_tips.append('Tempo stability: Record yourself and play back to develop tempo awareness. Practice starting pieces at different tempos without a metronome.')

    if analytics.get('legato_detected'):
        sev = analytics.get('legato_severity', 0)
        tips.append(f'Legato severity {sev * 100:.1f}% (lower is better)')

        # Add technique tips for legato
        if sev > 0.3:
            technique_tips.append('Legato technique: Practice finger substitution exercises. Connect notes by transferring weight smoothly between fingers while keeping wrist flexible.')
        elif sev > 0.1:
            technique_tips.append('Refined legato: Focus on "singing" through your fingers. Practice scales with different finger combinations to improve connection smoothness.')

    # General technique tips based on overall performance
    accuracy = analytics.get('accuracy', 0)
    relative_timing = analytics.get('relative', 0)
    absolute_timing = analytics.get('absolute', 0)

    if accuracy < 0.8:
        technique_tips.append('Accuracy foundation: Practice hands separately first. Use slow practice (50% tempo) with perfect accuracy before increasing speed.')
    elif accuracy < 0.95:
        technique_tips.append('Precision technique: Practice with mental preparation - visualize each note before playing. Use firm, deliberate finger actions.')

    if relative_timing < 0.8:
        technique_tips.append('Rhythmic precision: Clap the rhythm while singing note names. Practice with different articulations (staccato, legato, accented) to internalize the pattern.')
    elif relative_timing < 0.95:
        technique_tips.append('Advanced rhythm: Practice with displaced accents and cross-rhythms to develop rock-solid internal timing.')

    if absolute_timing < 0.8:
        technique_tips.append('Steady pulse: Practice with a drone or sustained chord. Count aloud while playing. Use body movement (tapping foot, swaying) to internalize the beat.')
    elif absolute_timing < 0.95:
        technique_tips.append('Pulse refinement: Practice with polyrhythms (2 against 3, etc.). Record yourself with a metronome and analyze timing deviations.')

    # Score-based encouragement and advanced tips
    overall_score = analytics.get('score', {}).get('overall', 0) if analytics.get('score') else 0

    if overall_score > 0.95:
        technique_tips.append('Mastery level: Focus on musical expression. Experiment with subtle tempo variations (rubato) and dynamic shading while maintaining technical precision.')
    elif overall_score > 0.85:
        technique_tips.append('Advanced practice: Try practicing in different keys or with altered rhythms. Focus on musical phrasing and breath-like flow between sections.')
    elif overall_score > 0.7:
        technique_tips.append('Building consistency: Practice the piece at 80% tempo until you can play it perfectly 3 times in a row before increasing speed.')

    # Mental and physical technique tips
    if len(tips) > 2:  # If there are multiple issues
        technique_tips.append('Practice strategy: Break the piece into small sections (2-4 measures). Master each section before combining. Always practice problem spots in isolation.')
        technique_tips.append('Physical technique: Check your posture and hand position. Tension often causes timing and accuracy issues. Practice relaxation exercises between repetitions.')

    if not tips and not technique_tips:
        tips = ['No major issues detected.']
        technique_tips = ['Excellent performance! Focus on musical expression and exploring different interpretations of the piece.']

    # Combine diagnostic and technique tips
    all_tips = tips + technique_tips
    list_items = ''.join([f'- {t} <br/>' for t in all_tips])
    return pygame_gui.elements.UITextBox(
        html_text=f'<b>Performance Analysis & Technique Tips:</b><br/>{list_items}',
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


def _matplotlib_spider_chart(analytics, width, height):
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


def _filter_midi_messages_by_section(midi_msgs, chord_times, start_chord_idx, end_chord_idx):
    """Filter MIDI messages to only include those within the specified chord index range."""
    if not midi_msgs or not chord_times or start_chord_idx >= len(chord_times):
        return {}
    start_time = chord_times[start_chord_idx] if start_chord_idx < len(chord_times) else 0
    if end_chord_idx < len(chord_times):
        if end_chord_idx + 1 < len(chord_times):
            end_time = chord_times[end_chord_idx + 1]
        else:
            end_time = float('inf')
    else:
        end_time = float('inf')

    filtered_msgs = {}
    for track_idx, messages in midi_msgs.items():
        filtered_track = []
        notes_started_in_section = set()

        for msg in messages:
            msg_time = getattr(msg, 'time', 0)

            if msg.type == "note_on" and msg.velocity > 0:
                if start_time <= msg_time < end_time:
                    filtered_track.append(msg)
                    notes_started_in_section.add(msg.note)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in notes_started_in_section or (start_time <= msg_time < end_time):
                    filtered_track.append(msg)
                    notes_started_in_section.discard(msg.note)
            else:
                if start_time <= msg_time < end_time:
                    filtered_track.append(msg)

        if filtered_track:
            filtered_msgs[track_idx] = filtered_track

    return filtered_msgs


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

        self._ui_manager.preload_fonts([{'name': 'noto_sans', 'point_size': 14, 'style': 'italic', 'antialiased': '1'}, {'name': 'noto_sans', 'point_size': 14, 'style': 'bold', 'antialiased': '1'}])

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self._build_path_map()
            self._analytics = self._get_selected_analytics()

    def draw(self, surface: pygame.Surface):
        if not self.visible:
            return

        with ElementSurface(surface) as (e, offset, dims):
            if self._ui_manager.window_resolution != e.get_size():
                self._ui_manager.set_window_resolution(e.get_size())
                self._ui_manager.set_offset(offset)
            _render_background(e)

            if not self._main_container:
                self._main_container = FlexBox(manager=self._ui_manager,
                                               relative_rect=pygame.Rect(0, 0, dims[0], dims[1]), gap=12, padding=self._margin)

                left_side = FlexBox(manager=self._ui_manager, relative_rect=pygame.Rect(0, 0, 0, 0), gap=12,
                                    align_y="start", direction="vertical")
                self._main_container.place_element(left_side, width_percent=0.6, height_percent=1)
                self._render_title(left_side)
                _render_overall_score(self._analytics, self._big_font, left_side)

                if self._analytics:
                    radar_chart = pygame_gui.elements.UIImage(
                        image_surface=pygame.Surface((100, 100), pygame.SRCALPHA),
                        relative_rect=pygame.Rect(0, 0, 100, 100),
                    )
                    left_side.place_element(radar_chart, width_percent=1, height_percent="max")
                    self._render_pianoroll(left_side)
                    radar_chart.set_image(_matplotlib_spider_chart(self._analytics, radar_chart.relative_rect[2],
                                                                   radar_chart.relative_rect[3]))

                self._main_container.place_element(_generate_tips(self._analytics), width_percent="max", height_percent=1)

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

    def _render_title(self, parent):
        title_surf = self._font.render('Performance Analytics for', True, (255, 255, 255))
        container = FlexBox(manager=self._ui_manager,
                            relative_rect=pygame.Rect(self._margin, self._margin, 0, title_surf.get_height()), gap=12)
        parent.place_element(container, height_px=title_surf.get_height(), width_percent=1)
        container.place_element(pygame_gui.elements.UIImage(
            image_surface=title_surf,
            relative_rect=pygame.Rect(0, 0, title_surf.get_width(), title_surf.get_height()),
        ), height_px=title_surf.get_height(), width_px=title_surf.get_width())
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

    def _get_expected_notes_for_section(self):
        """Get expected notes for the current section, handling transition sections specially."""
        if self._selected_section == "transition":
            from_measure = self._selected_measure
            to_measure = self._selected_measure + 1
            from_chords, from_times, from_xs, _, (from_start_idx, _), from_midi_msgs = self.teacher.midi_teacher.get_notes_for_measure(from_measure)
            to_chords, to_times, to_xs, _, (to_start_idx, _), to_midi_msgs = self.teacher.midi_teacher.get_notes_for_measure(to_measure)

            if not from_chords or not to_chords:
                return {}

            start_chord_idx, end_chord_idx = self._get_section_bounds()
            start_chord_idx -= from_start_idx
            end_chord_idx -= from_start_idx
            from_midi_msgs = _filter_midi_messages_by_section(from_midi_msgs, from_times, start_chord_idx, len(from_times) - 1)
            to_midi_msgs = _filter_midi_messages_by_section(to_midi_msgs, to_times, 0, end_chord_idx - len(from_times))
            max_from_time = 0
            for track in from_midi_msgs.values():
                for msg in track:
                    max_from_time = max(max_from_time, getattr(msg, 'time', 0))
            time_offset = max_from_time + 100
            filtered_midi_msgs = defaultdict(list)
            for track_idx, messages in from_midi_msgs.items():
                filtered_midi_msgs[track_idx].extend(messages)
            for track_idx, messages in to_midi_msgs.items():
                for msg in messages:
                    adjusted_msg = msg.copy()
                    adjusted_msg.time = getattr(msg, 'time', 0) + time_offset
                    filtered_midi_msgs[track_idx].append(adjusted_msg)

            return filtered_midi_msgs
        else:
            measure_data = self.teacher.midi_teacher.get_notes_for_measure(
                self._selected_measure,
                unpacked=False,
            )
            expected_chords, expected_times, expected_midi_msgs = measure_data.chords, measure_data.times, measure_data.midi_msgs

            try:
                start_chord_idx, end_chord_idx = self._get_section_bounds()
                measure_start_idx = measure_data.start_index
                relative_start_idx = start_chord_idx - measure_start_idx
                relative_end_idx = end_chord_idx - measure_start_idx

                expected_midi_msgs = _filter_midi_messages_by_section(
                    expected_midi_msgs, expected_times, relative_start_idx, relative_end_idx
                )
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass

            return expected_midi_msgs

    def _render_pianoroll(self, flexbox: FlexBox):
        surface_element = pygame_gui.elements.UIImage(
            image_surface=pygame.Surface((100, 100), pygame.SRCALPHA),
            relative_rect=pygame.Rect(0, 0, 100, 100),
        )
        flexbox.place_element(surface_element, width_percent=1, height_percent="max")
        rect = pygame.Rect(0, 0, surface_element.relative_rect[2]-8, surface_element.relative_rect[3])
        surface = pygame.Surface((rect.width+8, rect.height), pygame.SRCALPHA)

        pygame.draw.rect(surface, (20, 20, 20), rect, border_radius=8)

        expected_midi_msgs = self._get_expected_notes_for_section()

        performed_notes = self.teacher.midi_teacher.get_performed_notes_for_measure(
            self._selected_measure, self._selected_section, self._selected_pass
        )
        if not expected_midi_msgs and not performed_notes:
            return
        all_notes = [msg.note for track_index in range(len(expected_midi_msgs)) for msg in expected_midi_msgs[track_index] if msg.type in ("note_on", "note_off")]
        if performed_notes:
            all_notes += [msg.note for msg in performed_notes if msg.type in ("note_on", "note_off")]
        if not all_notes:
            return

        min_pitch, max_pitch = min(all_notes), max(all_notes)
        pitch_range = max_pitch - min_pitch + 1
        vertical_padding = 32
        padded_top = rect.top + vertical_padding
        padded_bottom = rect.bottom - vertical_padding
        padded_height = padded_bottom - padded_top
        bar_height = padded_height / pitch_range

        def pitch_to_y(note):
            rel = (note - min_pitch) / max(1, pitch_range)
            return padded_bottom - rel * padded_height
        expected_times_list = []
        for track in expected_midi_msgs.values() if expected_midi_msgs else []:
            for msg in track:
                if hasattr(msg, "time"):
                    expected_times_list.append(getattr(msg, "time", 0))
        performed_notes_absolute = []
        current_time = 0
        for msg in performed_notes:
            current_time += getattr(msg, 'time', 0)
            abs_msg = msg.copy()
            abs_msg.time = current_time
            performed_notes_absolute.append(abs_msg)

        performed_times_list = [msg.time for msg in performed_notes_absolute if hasattr(msg, "time")]

        min_expected_time = min(expected_times_list) if expected_times_list else 0
        max_expected_time = max(expected_times_list) if expected_times_list else 1
        expected_time_range = max_expected_time - min_expected_time if max_expected_time > min_expected_time else 1

        min_performed_time = min(performed_times_list) if performed_times_list else 0
        max_performed_time = max(performed_times_list) if performed_times_list else 1
        performed_time_range = max_performed_time - min_performed_time if max_performed_time > min_performed_time else 1
        def expected_time_to_x(time):
            norm = (time - min_expected_time) / expected_time_range if expected_time_range > 0 else 0
            return (rect.left + 12) + norm * (rect.width - 32)

        def performed_time_to_x(time):
            norm = (time - min_performed_time) / performed_time_range if performed_time_range > 0 else 0
            return (rect.left + 12) + norm * (rect.width - 32)

        expected_onsets = {}
        left_hand_color = (80, 150, 255)
        right_hand_color = (255, 80, 80)
        for track_index, track in enumerate(expected_midi_msgs.values() if expected_midi_msgs else []):
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    expected_onsets[msg.note] = getattr(msg, "time", 0)
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    onset = expected_onsets.pop(msg.note, None)
                    if onset is not None:
                        x1 = expected_time_to_x(onset)
                        x2 = expected_time_to_x(getattr(msg, "time", 0))
                        y = pitch_to_y(msg.note)
                        note_rect = pygame.Rect(x1, y - bar_height / 2, max(1, x2 - x1), bar_height)
                        color = right_hand_color if track_index == 0 else left_hand_color
                        pygame.draw.rect(surface, (*color, 180), note_rect, border_radius=8)
                        pygame.draw.rect(surface, color, note_rect.inflate(-4, -4), 4, border_radius=6)

        performed_onsets = {}
        performed_color = (0, 140, 40)
        for msg in performed_notes_absolute:
            if msg.type == "note_on" and msg.velocity > 0:
                performed_onsets[msg.note] = msg.time
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                onset = performed_onsets.pop(msg.note, None)
                if onset is not None:
                    x1 = performed_time_to_x(onset)
                    x2 = performed_time_to_x(msg.time)
                    y = pitch_to_y(msg.note)
                    note_rect = pygame.Rect(x1, y - (bar_height*.7) / 2, max(2, x2 - x1), (bar_height*.7))
                    pygame.draw.rect(surface, (*performed_color, 180), note_rect, border_radius=8)
                    pygame.draw.rect(surface, performed_color, note_rect.inflate(-2, -2), 2, border_radius=8)
        surface_element.set_image(surface)

    def _get_section_bounds(self):
        with self.save_system.guided_teacher_data as s:
            if not s.file_exists(f"measure_{self._selected_measure}/section_{self._selected_section}/section.json"):
                raise FileNotFoundError(f"Section {self._selected_section} not found in measure {self._selected_measure}")
            info = json.loads(s.load_file(f"measure_{self._selected_measure}/section_{self._selected_section}/section.json"))["section"]
            return info["start_idx"], info["end_idx"]
