import time
from collections import deque
from dataclasses import dataclass
import pygame
from midi_teach import MidiTeacher
from synth import Synth
from abc import ABC, abstractmethod

CHORDS_PER_SECTION = (3, 4)

@dataclass(frozen=True)
class MeasureSection:
    chords: list
    times: list
    xs: list
    start_idx: int
    end_idx: int

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
    def on_tick(self, pressed_notes, pressed_note_events, pygame_events):
        pass

class PlaybackMeasureTask(Task):
    def __init__(self, teacher, measure):
        super().__init__(teacher)
        self.measure = measure
        self.notes_x = []
        self.not_played_for = 0
        self.played = False

    def on_start(self):
        _, _, self.notes_x, _, _ = self.teacher.midi_teacher.get_notes_for_measure(self.measure)
        self.teacher.current_section_visual_info = [self.notes_x[0], self.notes_x[-1]]

    def on_end(self):
        self.teacher.current_section_visual_info = None

    def on_tick(self, pressed_notes, pressed_note_events, _):
        if self.not_played_for > 5 and not self.played: # give it a bit of time to render highlight
            self.teacher.synth.play_measure(self.measure, self.teacher.midi_teacher, self.teacher.midi_teacher.seek_to_index, self.teacher.midi_teacher.get_current_index())
            self.played = True
        elif self.played:
            self.teacher.next_task()
        else:
            self.not_played_for += 1

class PracticeSectionTask(Task):
    def __init__(self, teacher: 'GuidedTeacher', section: MeasureSection, measure):
        super().__init__(teacher)
        self.section = section
        self.measure = measure
        self.start_idx = section.start_idx
        self.end_idx = section.end_idx

    def on_start(self):
        self.teacher.current_section_visual_info = self.section.xs
        self.teacher.midi_teacher.loop_enabled = True
        self.teacher.midi_teacher.loop_start = self.start_idx
        self.teacher.midi_teacher.loop_end = self.end_idx
        self.teacher.current_section_visual_info = self.section.xs
        self.teacher.midi_teacher.seek_to_index(self.start_idx)

    def on_end(self):
        self.teacher.midi_teacher.loop_enabled = False
        self.teacher.current_section_visual_info = None

    def on_tick(self, pressed_notes, pressed_note_events, pygame_events):
        self._handle_pygame_events(pygame_events)

    def _handle_pygame_events(self, pygame_events):
        for event in pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.teacher.synth.play_measure(self.measure, self.teacher.midi_teacher, self.teacher.midi_teacher.seek_to_index, self.teacher.midi_teacher.get_current_index())
                elif event.key == pygame.K_SPACE:
                    self.teacher.synth.play_notes(self.section.chords, self.section.times, self.start_idx, self.teacher.midi_teacher.seek_to_index, self.teacher.midi_teacher.get_current_index())
                elif event.key == pygame.K_RETURN:
                    self.teacher.next_task()

class PracticeMeasureTask(PracticeSectionTask):
    def __init__(self, teacher: 'GuidedTeacher', measure):
        chords, times, xs, _, (measure_start_index, _) = teacher.midi_teacher.get_notes_for_measure(measure)
        start_idx = measure_start_index
        end_idx = start_idx + len(chords) - 1 if chords else start_idx
        section = MeasureSection(chords, times, xs, start_idx, end_idx)
        super().__init__(teacher, section, measure)

class PracticeTransitionTask(PracticeSectionTask):
    def __init__(self, teacher, from_measure, to_measure):
        from_chords, from_times, from_xs, _, (from_start_idx, _) = teacher.midi_teacher.get_notes_for_measure(from_measure)
        to_chords, to_times, to_xs, _, (to_start_idx, _) = teacher.midi_teacher.get_notes_for_measure(to_measure)
        section_chords = from_chords[-2:] + to_chords[:2]
        section_times = [0, from_times[-1] - from_times[-2]]  + [(from_times[-1] - from_times[-2])*2, to_times[1] + (from_times[-1] - from_times[-2])*2]
        section_xs = from_xs[-2:] + to_xs[:2]
        start_idx = from_start_idx + max(0, len(from_chords) - 2)
        end_idx = to_start_idx + min(1, len(to_chords) - 1) if to_chords else start_idx
        section = MeasureSection(section_chords, section_times, section_xs, start_idx, end_idx)
        super().__init__(teacher, section, from_measure)

    def _handle_pygame_events(self, pygame_events):
        for event in pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.teacher.synth.play_measure(self.measure, self.teacher.midi_teacher, self.teacher.midi_teacher.seek_to_index, self.teacher.midi_teacher.get_current_index())
                    time.sleep(.1)
                    self.teacher.synth.play_measure(self.measure+1, self.teacher.midi_teacher,
                                                    self.teacher.midi_teacher.seek_to_index,
                                                    self.teacher.midi_teacher.get_current_index())
                elif event.key == pygame.K_SPACE:
                    self.teacher.synth.play_notes(self.section.chords, self.section.times, self.start_idx, self.teacher.midi_teacher.seek_to_index, self.teacher.midi_teacher.get_current_index())
                elif event.key == pygame.K_RETURN:
                    self.teacher.next_task()

class GenerateNextMeasureTasks(Task):
    def on_start(self):
        self.teacher.current_measure_index += 1
        self.teacher.current_measure_index %= len(self.teacher.midi_teacher.measures)
        self.teacher.generate_tasks_for_measure(self.teacher.current_measure_index)
        self.teacher.next_task()

    def on_end(self):
        pass

    def on_tick(self, pressed_notes, pressed_note_events, pygame_events):
        pass

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

    def update(self, pressed_notes, pressed_note_events, pygame_events):
        if not self.is_active:
            return
        if not self.current_task and self.tasks:
            self.next_task()
        self.current_task.on_tick(pressed_notes, pressed_note_events, pygame_events)

    def generate_tasks_for_measure(self, measure_index):
        self.tasks.clear()
        self.measure_sections = self.split_measure_into_sections(measure_index)

        self.tasks.append(PlaybackMeasureTask(self, measure_index))
        for section in self.measure_sections:
            self.tasks.append(PracticeSectionTask(self, section, measure_index))
        self.tasks.append(PracticeMeasureTask(self, measure_index))
        if measure_index + 1 < len(self.midi_teacher.measures):
            self.tasks.append(PracticeTransitionTask(self, measure_index, measure_index + 1))
        self.tasks.append(GenerateNextMeasureTasks(self))

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
        measure_chords, measure_times, measure_xs, _, (measure_start_index, _) = self.midi_teacher.get_notes_for_measure(measure_index)
        if not measure_chords:
            return []

        min_chords, max_chords = CHORDS_PER_SECTION
        n = len(measure_chords)
        sections = []
        i = 0
        while i < n:
            end = min(i + max_chords, n)
            next_i = i + max_chords - 1  # overlap 1
            next_remaining = n - next_i
            if next_remaining < min_chords and (i + max_chords - 2 >= 0):
                penult_len = n - (i + max_chords - 2)
                curr_len = end - i
                if penult_len >= min_chords and curr_len > min_chords:
                    end = min(i + max_chords, n)
                    section_chords = measure_chords[i:end]
                    section_times = measure_times[i:end]
                    section_xs = measure_xs[i:end]
                    start_index = measure_start_index + i
                    end_index = start_index + len(section_chords) - 1
                    sections.append(MeasureSection(section_chords, section_times, section_xs, start_index, end_index))
                    i = n - penult_len
                    end = n
                    section_chords = measure_chords[i:end]
                    section_times = measure_times[i:end]
                    section_xs = measure_xs[i:end]
                    start_index = measure_start_index + i
                    end_index = start_index + len(section_chords) - 1
                    sections.append(MeasureSection(section_chords, section_times, section_xs, start_index, end_index))
                    break
            section_chords = measure_chords[i:end]
            section_times = measure_times[i:end]
            section_xs = measure_xs[i:end]
            start_index = measure_start_index + i
            end_index = start_index + len(section_chords) - 1
            sections.append(MeasureSection(section_chords, section_times, section_xs, start_index, end_index))
            if end == n:
                break
            i += max_chords - 1  # overlap 1

        print(f"[Debug] Split measure {measure_index} into {len(sections)} sections.")

        return sections

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