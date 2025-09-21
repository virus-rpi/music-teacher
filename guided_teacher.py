import time
from collections import deque
from dataclasses import dataclass, asdict
import mido
import pygame
from evaluator import Evaluator
from midi_teach import MidiTeacher
from synth import Synth
from abc import ABC, abstractmethod
import json
import os
from analytics_popup import AnalyticsPopup

CHORDS_PER_SECTION = (3, 4)
STATE_FILE = 'guided_teacher_state.json'

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

        self.timer_start = None
        self.recording = mido.MidiTrack()

        self._prev_pressed_notes = set()
        self._last_record_time = None

        self._waiting_for_wrap_release = False
        self._wrap_pressed_notes = set()

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
        self._handle_midi_events(pressed_notes, pressed_note_events)

        if self._waiting_for_wrap_release:
            if not (self._wrap_pressed_notes & pressed_notes):
                self._waiting_for_wrap_release = False
                self._wrap_pressed_notes.clear()
                self._handle_evaluate()

                self.recording.clear()
                self._prev_pressed_notes.clear()
                self.timer_start = None
                self._last_record_time = None
            return

        if self.teacher.midi_teacher.did_wrap_and_clear():
            wrap_pressed = pressed_notes.copy()
            if not wrap_pressed:
                self._handle_evaluate()

                self.recording.clear()
                self._prev_pressed_notes.clear()
                self.timer_start = None
                self._last_record_time = None
            else:
                self._waiting_for_wrap_release = True
                self._wrap_pressed_notes = wrap_pressed

    def _handle_pygame_events(self, pygame_events):
        for event in pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.teacher.synth.play_measure(self.measure, self.teacher.midi_teacher, self.teacher.midi_teacher.seek_to_index, self.teacher.midi_teacher.get_current_index())
                elif event.key == pygame.K_SPACE:
                    self.teacher.synth.play_notes(self.section.chords, self.section.times, self.start_idx, self.teacher.midi_teacher.seek_to_index, self.teacher.midi_teacher.get_current_index())

    def _handle_midi_events(self, pressed_notes: set[int], pressed_note_events):
        if len(self.recording) == 0 and len(pressed_note_events) > 0:
            times = [ev.time for ev in pressed_note_events if hasattr(ev, 'time') and ev.note in [n[0] for n in self.section.chords[0]]]
            if not times:
                return
            earliest = min(times)
            self.timer_start = earliest

        if self._last_record_time is None and self.timer_start is not None:
            self._last_record_time = self.timer_start

        current = pressed_notes.copy()
        new_notes = current - self._prev_pressed_notes
        released_notes = self._prev_pressed_notes - current

        def append_msg(msg: mido.Message, timestamp_s: float):
            if self._last_record_time is None:
                delta_ms = 0
            else:
                delta_ms = max(int((timestamp_s - self._last_record_time) * 1000), 0)
            msg.time = delta_ms
            self.recording.append(msg)
            self._last_record_time = timestamp_s

        for note in sorted(new_notes):
            ev = next((ev for ev in pressed_note_events if ev.note == note), None)
            if ev is not None and hasattr(ev, 'time'):
                t = ev.time
                vel = getattr(ev, 'velocity', 127)
            else:
                t = time.time()
                vel = 127
            append_msg(mido.Message('note_on', note=note, velocity=vel), t)

        now = time.time()
        for note in sorted(released_notes):
            append_msg(mido.Message('note_off', note=note, velocity=0), now)

        self._prev_pressed_notes = current

    def _handle_evaluate(self):
        print(self.recording)
        evaluator = Evaluator(self)
        score = evaluator.score
        guidance_text = evaluator.generate_guidance(score)

        self.teacher.last_score = score.overall
        idx = getattr(self, 'measure', None)
        section = getattr(self, 'section', None)
        if idx is not None and section is not None:
            section_key = f"{idx}_{section.start_idx}_{section.end_idx}"
            hist = self.teacher.evaluator_history.setdefault(section_key, {'scores': [], 'best_score': 0.0, 'analytics': [], 'recording': None})
            hist['scores'].append(score.overall)
            hist['best_score'] = max(hist['best_score'], score.overall)
            hist['analytics'].append(getattr(evaluator, 'analytics', {}))
            hist['recording'] = self.recording.copy()
            self.teacher.save_state()

        print(f"Eval: accuracy={score.accuracy:.3f} rel={score.relative_timing:.3f} abs={score.absolute_timing:.3f} -> score={score.overall:.3f}")
        if score.overall > 0.95 and self.teacher.auto_advance:
            self.teacher.synth.play_success_sound()
            self.teacher.next_task()
            self.teacher.guide_text = "Great! " + guidance_text
        else:
            self.teacher.guide_text = guidance_text

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
        self.auto_advance = True
        self.tasks = deque()
        self.history = []
        self.current_task = None
        self.measure_sections = []
        self.current_measure_index = 0
        self.current_section_visual_info = None
        self.last_score = 0.0
        self.guide_text = None
        self._loop_info_before = None
        self.evaluator_history = {}
        self._state_file = STATE_FILE
        self.analytics_popup = AnalyticsPopup(self)
        self.load_state()

    def get_last_score(self):
        return self.last_score

    def get_guide_text(self):
        return self.guide_text

    def start(self):
        self.is_active = True
        self._loop_info_before = (self.midi_teacher.loop_enabled, self.midi_teacher.loop_start, self.midi_teacher.loop_end)
        self.load_state()
        if not self.tasks and not self.current_task:
            self.current_measure_index = 0
            self.generate_tasks_for_measure(self.current_measure_index)

    def stop(self):
        self.is_active = False
        self.save_state()
        self.tasks.clear()
        self.history.clear()
        self.current_task = None
        self.midi_teacher.loop_enabled, self.midi_teacher.loop_start, self.midi_teacher.loop_end = self._loop_info_before

    def update(self, pressed_notes, pressed_note_events, pygame_events):
        if not self.is_active:
            return

        for event in pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    self.analytics_popup.toggle()
                    return
        if self.analytics_popup.visible:
            for event in pygame_events:
                self.analytics_popup.handle_event(event)
            return

        if not self.current_task and self.tasks:
            self.next_task()
        self.current_task.on_tick(pressed_notes, pressed_note_events, pygame_events)

        for event in pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and (event.mod & pygame.KMOD_SHIFT):
                    self.previous_task()
                elif event.key == pygame.K_RETURN:
                    self.next_task()
                elif event.key == pygame.K_a:
                    self.auto_advance = not self.auto_advance
                    print(f"Auto advance: {self.auto_advance}")
        self.save_state()

    def render(self, surface):
        if hasattr(self, 'analytics_popup'):
            self.analytics_popup.draw(surface)

    def generate_tasks_for_measure(self, measure_index):
        self.tasks.clear()
        self.measure_sections = self.split_measure_into_sections(measure_index)

        self.tasks.append(PlaybackMeasureTask(self, measure_index))
        for section in self.measure_sections:
            self.tasks.append(PracticeSectionTask(self, section, measure_index))
        self.tasks.append(PracticeMeasureTask(self, measure_index))
        if measure_index + 1 < len(self.midi_teacher.measures):
            self.tasks.append(PracticeTransitionTask(self, measure_index, measure_index + 1))
            # TODO: if its the last one let the use play the whole song and evaluate it and based on the evaluation add more tasks to practice specific areas
        self.tasks.append(GenerateNextMeasureTasks(self))
        self.save_state()

    def next_task(self):
        if self.current_task:
            self.current_task.on_end()
            if not isinstance(self.current_task, (GenerateNextMeasureTasks, PlaybackMeasureTask)):
                self.history.append(self.current_task)
        if self.tasks:
            self.current_task = self.tasks.popleft()
            self.current_task.on_start()
        else:
            self.current_task = None
            self.stop()
        self.save_state()

    def previous_task(self):
        if self.history:
            if self.current_task:
                self.current_task.on_end()
                self.tasks.insert(0, self.current_task)
            self.current_task = self.history.pop()
            self.current_task.on_start()
        self.save_state()

    def split_measure_into_sections(self, measure_index):
        measure_chords, measure_times, measure_xs, _, (measure_start_index, _) = self.midi_teacher.get_notes_for_measure(measure_index)
        if not measure_chords:
            return []

        min_chords, max_chords = CHORDS_PER_SECTION
        n = len(measure_chords)
        sections = []
        i = 0
        def add_section(section_start, section_end):
            section_chords = measure_chords[section_start:section_end]
            section_times = measure_times[section_start:section_end]
            section_xs = measure_xs[section_start:section_end]
            start_index = measure_start_index + section_start
            end_index = start_index + len(section_chords) - 1
            sections.append(MeasureSection(section_chords, section_times, section_xs, start_index, end_index))
        while i < n:
            end = min(i + max_chords, n)
            next_i = i + max_chords - 1  # overlap 1
            next_remaining = n - next_i
            if next_remaining < min_chords and (i + max_chords - 2 >= 0):
                penult_len = n - (i + max_chords - 2)
                curr_len = end - i
                if penult_len >= min_chords and curr_len > min_chords:
                    add_section(i, end)
                    i = n - penult_len
                    add_section(i, n)
                    break
            add_section(i, end)
            if end == n:
                break
            i += max_chords - 1  # overlap 1
        return sections

    def to_dict(self):
        def task_to_dict(task):
            if task is None:
                return None
            section = getattr(task, 'section', None)
            if section is not None and isinstance(section, MeasureSection):
                section = asdict(section)
            return {
                'type': type(task).__name__,
                'measure': getattr(task, 'measure', None),
                'section': section,
                'start_idx': getattr(task, 'start_idx', None),
                'end_idx': getattr(task, 'end_idx', None),
            }
        tasks_list = list(self.tasks)
        if self.current_task is not None:
            tasks_list = [self.current_task] + tasks_list
        return {
            'current_measure_index': self.current_measure_index,
            'auto_advance': self.auto_advance,
            'tasks': [task_to_dict(t) for t in tasks_list],
            'history': [task_to_dict(t) for t in self.history],
            'current_task': None,
            'last_score': self.last_score,
            'guide_text': self.guide_text,
            'evaluator_history': self.evaluator_history,
        }

    def from_dict(self, d):
        def dict_to_task(td):
            if td is None:
                return None
            ttype = td.get('type')
            measure = td.get('measure')
            section = td.get('section')
            if ttype == 'PlaybackMeasureTask':
                return PlaybackMeasureTask(self, measure)
            elif ttype == 'PracticeSectionTask':
                if section and isinstance(section, dict):
                    sec = MeasureSection(**section)
                else:
                    sec = None
                return PracticeSectionTask(self, sec, measure)
            elif ttype == 'PracticeMeasureTask':
                return PracticeMeasureTask(self, measure)
            elif ttype == 'PracticeTransitionTask':
                return PracticeTransitionTask(self, measure, measure+1)
            elif ttype == 'GenerateNextMeasureTasks':
                return GenerateNextMeasureTasks(self)
            return None
        self.current_measure_index = d.get('current_measure_index', 0)
        self.auto_advance = d.get('auto_advance', True)
        self.tasks = deque([dict_to_task(td) for td in d.get('tasks', []) if td])
        self.history = [dict_to_task(td) for td in d.get('history', []) if td]
        self.current_task = dict_to_task(d.get('current_task'))
        self.last_score = d.get('last_score', 0.0)
        self.guide_text = d.get('guide_text', None)
        self.evaluator_history = d.get('evaluator_history', {})

    def save_state(self):
        try:
            with open(self._state_file, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save state: {e}")

    def load_state(self):
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, 'r') as f:
                    d = json.load(f)
                self.from_dict(d)
            except Exception as e:
                print(f"Failed to load state: {e}")
