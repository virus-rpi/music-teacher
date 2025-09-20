from collections import deque
from midi_teach import MidiTeacher
from synth import Synth
from abc import ABC, abstractmethod

NOTES_PER_SECTION = 4

class Task(ABC):
    def __init__(self, teacher: 'GuidedTeacher'):
        self.teacher = teacher

    @abstractmethod
    def on_start(self):
        pass

    @abstractmethod
    def on_end(self):
        pass

    @abstractmethod
    def on_tick(self, pressed_notes, pressed_note_events):
        pass

class PlaybackMeasureTask(Task):
    def __init__(self, teacher, measure):
        super().__init__(teacher)
        self.measure = measure
        self.not_played_for = 0
        self.played = False

    def on_start(self):
        _, _, _, (start_x, end_x) = self.teacher.midi_teacher.get_notes_for_measure(self.measure)
        self.teacher.current_section_visual_info = [start_x, end_x]

    def on_end(self):
        self.teacher.current_section_visual_info = None

    def on_tick(self, pressed_notes, pressed_note_events):
        if self.not_played_for > 10 and not self.played: # give it a bit of time to render highlight
            self.teacher.synth.play_measure(self.measure, self.teacher.midi_teacher)
            self.played = True
        elif self.played:
            self.teacher.next_task()
        else:
            self.not_played_for += 1

class PracticeSectionTask(Task):
    def __init__(self, teacher, section, measure):
        super().__init__(teacher)
        self.section = section
        self.measure = measure
        self._section_progress = None
        self._section_loop_info = None

    def on_start(self):
        self.teacher.current_section_visual_info = self.section[2]
        measure_chords, _, _, _ = self.teacher.midi_teacher.get_notes_for_measure(self.measure)
        section_chords = self.section[0]
        start_idx = next(i for i in range(len(measure_chords)) if measure_chords[i] == section_chords[0])
        end_idx = start_idx + len(section_chords) - 1
        self.teacher.midi_teacher.loop_enabled = True
        self.teacher.midi_teacher.loop_start = start_idx
        self.teacher.midi_teacher.loop_end = end_idx
        self.teacher.midi_teacher.seek_to_index(start_idx)
        self._section_progress = None
        self._section_loop_info = None

    def on_end(self):
        self.teacher.midi_teacher.loop_enabled = False
        self.teacher.current_section_visual_info = None

    def on_tick(self, pressed_notes, pressed_note_events):
        teacher = self.teacher
        section_chords, section_times, _ = self.section
        if not pressed_note_events:
            return
        if not self._section_loop_info or self._section_loop_info.get('section_id') != id(self.section):
            measure_chords, _, _, _ = teacher.midi_teacher.get_notes_for_measure(self.measure)
            start_idx = next(i for i in range(len(measure_chords)) if measure_chords[i] == section_chords[0])
            end_idx = start_idx + len(section_chords) - 1
            teacher.midi_teacher.set_loop_start_index(start_idx)
            teacher.midi_teacher.set_loop_end_index(end_idx)
            teacher.midi_teacher.loop_enabled = True
            teacher.midi_teacher.seek_to_index(start_idx)
            teacher.synth.play_notes(section_chords, section_times)
            self._section_loop_info = {'section_id': id(self.section), 'start_idx': start_idx, 'end_idx': end_idx}
            self._section_progress = {
                'section_id': id(self.section),
                'chord_idx': 0,
                'matched_events': []
            }
            return
        if teacher.midi_teacher.did_wrap_and_clear():
            matched_events = self._section_progress['matched_events']
            eval_pressed_notes = set(n for notes, _ in matched_events for n in (notes if isinstance(notes, (list, tuple, set)) else [notes]))
            eval_pressed_note_events = [(notes, ts) for notes, ts in matched_events]
            teacher.last_score = evaluate_performance(section_chords, section_times, eval_pressed_notes, eval_pressed_note_events)
            if teacher.last_score >= 0.8:
                teacher.history.append(self)
                self._section_progress = {}
                self._section_loop_info = {}
                teacher.midi_teacher.loop_enabled = False
                teacher.next_task()
                return
            else:
                self._section_progress = {
                    'section_id': id(self.section),
                    'chord_idx': 0,
                    'matched_events': []
                }
            return
        chord_idx = self._section_progress['chord_idx']
        matched_events = self._section_progress['matched_events']
        while chord_idx < len(section_chords) and pressed_note_events:
            expected_chord = set(note for note, _ in section_chords[chord_idx])
            for i, (notes, timestamp) in enumerate(pressed_note_events):
                if isinstance(notes, int):
                    played_set = {notes}
                else:
                    played_set = set(notes)
                if expected_chord.issubset(played_set):
                    matched_events.append((notes, timestamp))
                    chord_idx += 1
                    pressed_note_events = pressed_note_events[i+1:]
                    break
            else:
                break
        self._section_progress['chord_idx'] = chord_idx
        self._section_progress['matched_events'] = matched_events
        if chord_idx == len(section_chords):
            eval_pressed_notes = set(n for notes, _ in matched_events for n in (notes if isinstance(notes, (list, tuple, set)) else [notes]))
            eval_pressed_note_events = [(notes, ts) for notes, ts in matched_events]
            teacher.last_score = evaluate_performance(section_chords, section_times, eval_pressed_notes, eval_pressed_note_events)
            if teacher.last_score >= 0.8:
                teacher.history.append(self)
                self._section_progress = {}
                teacher.next_task()
            else:
                self._section_progress = {}

class PracticeMeasureTask(PracticeSectionTask):
    def __init__(self, teacher, measure):
        cords, times, note_xs, _ = teacher.midi_teacher.get_notes_for_measure(measure)
        super().__init__(teacher, (cords, times, note_xs), measure)

    def on_start(self):
        super().on_start()

    def on_end(self):
        super().on_end()

    def on_tick(self, pressed_notes, pressed_note_events):
        super().on_tick(pressed_notes, pressed_note_events)

class PracticeTransitionTask(PracticeSectionTask):
    def __init__(self, teacher, from_measure, to_measure):
        from_chords, from_times, from_xs, _ = teacher.midi_teacher.get_notes_for_measure(from_measure)
        to_chords, to_times, to_xs, _ = teacher.midi_teacher.get_notes_for_measure(to_measure)
        super().__init__(teacher, (from_chords[-2:] + to_chords[:2], from_times[-2:] + to_times[:2], from_xs[-2:] + to_xs[:2]), from_measure)

    def on_start(self):
        super().on_start()

    def on_end(self):
        super().on_end()

    def on_tick(self, pressed_notes, pressed_note_events):
        super().on_tick(pressed_notes, pressed_note_events)

class GuidedTeacher:
    def __init__(self, midi_teacher: MidiTeacher, synth: Synth):
        self._section_progress = None
        self.midi_teacher = midi_teacher
        self.synth = synth
        self.is_active = False
        self.tasks = deque()
        self.history = []
        self.current_task = None
        self.measure_sections = []
        self.current_measure_index = 0
        self.current_section_visual_info = None
        self.last_score = 0.0
        self._loop_info_before = None

    def get_last_score(self):
        return self.last_score

    def start(self):
        self.is_active = True
        self._loop_info_before = (self.midi_teacher.loop_enabled, self.midi_teacher.loop_start, self.midi_teacher.loop_end)
        self.current_measure_index = 0
        self.generate_tasks_for_measure(self.current_measure_index)

    def stop(self):
        self.is_active = False
        self.tasks.clear()
        self.history.clear()
        self.current_task = None
        self.midi_teacher.loop_enabled, self.midi_teacher.loop_start, self.midi_teacher.loop_end = self._loop_info_before

    def update(self, pressed_notes, pressed_note_events):
        if not self.is_active:
            return
        if not self.current_task and self.tasks:
            self.next_task()
        self.current_task.on_tick(pressed_notes, pressed_note_events)

    def generate_tasks_for_measure(self, measure_index):
        self.tasks.clear()
        self.measure_sections = self.split_measure_into_sections(measure_index)

        self.tasks.append(PlaybackMeasureTask(self, measure_index))
        for section in self.measure_sections:
            self.tasks.append(PracticeSectionTask(self, section, measure_index))
        self.tasks.append(PracticeMeasureTask(self, measure_index))
        if measure_index + 1 < len(self.midi_teacher.measures):
            self.tasks.append(PracticeTransitionTask(self, measure_index, measure_index + 1))

    def next_task(self):
        if self.current_task:
            self.current_task.on_end()
        if self.tasks:
            self.current_task = self.tasks.popleft()
            self.current_task.on_start()
        else:
            self.current_task = None
            self.stop()

    def split_measure_into_sections(self, measure_index):
        measure_chords, measure_times, measure_xs, _ = self.midi_teacher.get_notes_for_measure(measure_index)
        if not measure_chords:
            return []

        sections = []
        notes_per_section = min(NOTES_PER_SECTION, len(measure_chords))

        for i in range(0, len(measure_chords) - notes_per_section + 1, notes_per_section - 1):
            section_chords = measure_chords[i:i + notes_per_section]
            section_times = measure_times[i:i + notes_per_section]
            section_xs = measure_xs[i:i + notes_per_section]
            sections.append((section_chords, section_times, section_xs))

        print(f"[Debug] Split measure {measure_index} into {len(sections)} sections.")

        return sections

    def repeat_measure(self):
        if self.current_task and hasattr(self.current_task, 'measure'):
            self.synth.play_measure(self.current_task.measure, self.midi_teacher)

    def get_current_task_info(self):
        return self.current_task


def evaluate_performance(expected_chords, expected_times, pressed_notes, pressed_note_events):
        if not expected_chords:
            return 1.0
        if not pressed_note_events:
            return 0.0

        expected_note_set = {note for chord in expected_chords for note, hand in chord}
        # Accuracy
        accuracy = len(expected_note_set.intersection(pressed_notes)) / len(expected_note_set)

        # Relative Timing
        expected_intervals = [expected_times[i+1] - expected_times[i] for i in range(len(expected_times)-1)]
        pressed_intervals = [pressed_note_events[i+1][1] - pressed_note_events[i][1] for i in range(len(pressed_note_events)-1)]
        relative_timing_score = 0.0
        if expected_intervals and pressed_intervals:
            sum_expected = sum(expected_intervals)
            sum_pressed = sum(pressed_intervals)
            if sum_expected > 0 and sum_pressed > 0:
                expected_intervals_norm = [i / sum_expected for i in expected_intervals]
                pressed_intervals_norm = [i / sum_pressed for i in pressed_intervals]
                min_len = min(len(expected_intervals_norm), len(pressed_intervals_norm))
                correlation = sum(expected_intervals_norm[i] * pressed_intervals_norm[i] for i in range(min_len))
                relative_timing_score = correlation

        # Absolute Timing
        absolute_timing_score = 0.0
        if expected_times and pressed_note_events:
            time_diff = pressed_note_events[0][1] - expected_times[0]
            total_diff = 0
            for i in range(min(len(expected_times), len(pressed_note_events))):
                total_diff += abs((pressed_note_events[i][1] - time_diff) - expected_times[i])
            avg_diff = total_diff / min(len(expected_times), len(pressed_note_events))
            absolute_timing_score = max(0.0, 1 - (avg_diff / 1000.0)) # 1 second diff = 0 score

        final_score = (accuracy * 1.0) + (relative_timing_score * 0.6) + (absolute_timing_score * 0.2)
        return final_score / (1.0 + 0.6 + 0.2)