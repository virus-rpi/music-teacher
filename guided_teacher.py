import time
from collections import deque
from dataclasses import asdict
import mido
import pygame
from evaluator import Evaluator
from mt_types import PerformanceEvaluation, MeasureSection
from midi_teach import MidiTeacher
from synth import Synth
from abc import ABC, abstractmethod
import json
import os
from analytics_popup import AnalyticsPopup
import threading
from save_system import SaveSystem

CHORDS_PER_SECTION = (3, 4)
SAVE_THROTTLE_SECONDS = 2.0

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
    def on_tick(self, pressed_notes: set[int], midi_events: list[mido.Message], pygame_events: list[pygame.event.Event]):
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
        self.teacher.current_section_visual_info = [self.notes_x[0], self.notes_x[-1]] if self.notes_x else None

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
        self.section_index = next((i for i, sec in enumerate(self.teacher.split_measure_into_sections(self.measure)) if sec.start_idx == self.section.start_idx and sec.end_idx == self.section.end_idx), None)

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

    def on_tick(self, pressed_notes, midi_events, pygame_events):
        self._handle_pygame_events(pygame_events)
        self._handle_midi_events(pressed_notes, midi_events)

        if self._waiting_for_wrap_release:
            if not (self._wrap_pressed_notes & pressed_notes) and len(midi_events) == 0:
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

    def _handle_midi_events(self, pressed_notes: set[int], midi_events: list[mido.Message]):
        if len(self.recording) == 0 and len(midi_events) > 0:
            first_chord = [n[0] for n in self.section.chords[0]]
            times = [getattr(ev, 'time', None) for ev in midi_events if hasattr(ev, 'time') and hasattr(ev, 'note') and getattr(ev, 'note', None) in first_chord]
            if not times:
                return
            earliest = min(times)
            self.timer_start = earliest

        if self._last_record_time is None and self.timer_start is not None:
            self._last_record_time = self.timer_start

        if self.timer_start is not None:
            for msg in midi_events:
                t = getattr(msg, 'time', time.time())
                delta_ms = max(int((t - self._last_record_time) * 1000), 0) if self._last_record_time is not None else 0
                self.recording.append(msg.copy(time=delta_ms))
                self._last_record_time = t

        self._prev_pressed_notes = pressed_notes.copy()

    def _handle_evaluate(self):
        print(self.recording)
        evaluator = Evaluator(
            self.recording,
            self.teacher.midi_teacher.query_notes_and_pedals(self.start_idx, self.end_idx),
        )
        score = evaluator.score
        guidance_text = evaluator.tip

        self.teacher.last_score = score
        self._save_pass(evaluator)
        if score > 0.95 and self.teacher.auto_advance:
            self.teacher.synth.play_success_sound()
            self.teacher.next_task()
            self.teacher.guide_text = "Great! " + guidance_text
        else:
            self.teacher.guide_text = guidance_text

    def _save_pass(self, evaluator):
        if self.section_index is not None:
            self.teacher.save_pass(str(self.measure), str(self.section_index), self.teacher.pass_index.get(str(self.measure), {}).get(str(self.section_index), 0), evaluator.full_evaluation,
                                   self.recording, self.section)
        else:
            print(f"Could not find section {self.section.start_idx} {self.section.end_idx} in measure {self.measure}")

class PracticeMeasureTask(PracticeSectionTask):
    def __init__(self, teacher: 'GuidedTeacher', measure):
        chords, times, xs, _, (measure_start_index, _) = teacher.midi_teacher.get_notes_for_measure(measure)
        start_idx = measure_start_index
        end_idx = start_idx + len(chords) - 1 if chords else start_idx
        section = MeasureSection(chords, times, xs, start_idx, end_idx)
        super().__init__(teacher, section, measure)
        self.section_index = "measure"

class PracticeTransitionTask(PracticeSectionTask):
    def __init__(self, teacher, from_measure, to_measure):
        from_chords, from_times, from_xs, _, (from_start_idx, _) = teacher.midi_teacher.get_notes_for_measure(from_measure)
        to_chords, to_times, to_xs, _, (to_start_idx, _) = teacher.midi_teacher.get_notes_for_measure(to_measure)
        section_chords = from_chords[-2:] + to_chords[:2]
        if len(from_times) >= 2 and len(to_times) >= 2:
            dt = from_times[-1] - from_times[-2]
            section_times = [0, dt] + [2 * dt, to_times[1] + 2 * dt]
        else:
            section_times = [0] * len(section_chords)
        section_xs = from_xs[-2:] + to_xs[:2]
        start_idx = from_start_idx + max(0, len(from_chords) - 2)
        end_idx = to_start_idx + min(1, len(to_chords) - 1) if to_chords else start_idx
        section = MeasureSection(section_chords, section_times, section_xs, start_idx, end_idx)
        super().__init__(teacher, section, from_measure)
        self.section_index = "transition"

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
    def __init__(self, midi_teacher: MidiTeacher, synth: Synth, save_system: SaveSystem = None):
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
        self.analytics_popup = AnalyticsPopup(self, save_system)
        self._save_lock = threading.Lock()
        self._last_save_time = 0
        self.save_system = save_system.guided_teacher_data
        self._save_root = os.path.join(save_system.save_root, save_system.guided_teacher_data_dir)
        self._unzipped_this_run = False
        self.pass_index = {}
        self.load_state()

    def save_state(self, force=False):
        now = time.time()
        if not force and now - self._last_save_time < SAVE_THROTTLE_SECONDS:
            return
        with self._save_lock:
            os.makedirs(self._save_root, exist_ok=True)
            try:
                state_dict = self.to_dict()
                state_dict['midi_teacher_index'] = self.midi_teacher.get_current_index()
                with self.save_system as s:
                    s.save_state(state_dict)
                self._last_save_time = now
            except Exception as e:
                print(f"Failed to save state: {e}")

    def load_state(self):
        state = self.save_system.load_state()
        if state:
            self.from_dict(state)
            idx = state.get('midi_teacher_index')
            if idx is not None:
                self.midi_teacher.seek_to_index(idx)

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
        if self.analytics_popup.visible: self.analytics_popup.toggle()
        self.is_active = False
        self.save_state()
        self.tasks.clear()
        self.history.clear()
        self.current_task = None
        self.midi_teacher.loop_enabled, self.midi_teacher.loop_start, self.midi_teacher.loop_end = self._loop_info_before

    def update(self, pressed_notes, midi_events, pygame_events):
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
        if self.current_task:
            self.current_task.on_tick(pressed_notes, midi_events, pygame_events)

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
        self.save_state(force=True)

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
        self.save_state(force=True)

    def previous_task(self):
        if self.history:
            if self.current_task:
                self.current_task.on_end()
                self.tasks.insert(0, self.current_task)
            self.current_task = self.history.pop()
            self.current_task.on_start()
        self.save_state(force=True)

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
            'pass_index': self.pass_index,
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
        self.pass_index = d.get('pass_index', {})

    def save_pass(self, measure_idx: str, section_idx: str, pass_idx: int, evaluation: PerformanceEvaluation, recording: mido.MidiTrack, section: MeasureSection):
        with self.save_system as s:
            if pass_idx == 0:
                s.save_file(f"measure_{measure_idx}/section_{section_idx}/section.json", json.dumps({'section': asdict(section), 'timestamp': time.time()}, indent=2, default=str))
            s.save_file(f"measure_{measure_idx}/section_{section_idx}/pass_{pass_idx}.json", json.dumps({'evaluation': asdict(evaluation), 'timestamp': time.time()}, indent=2, default=str))
            if recording is not None and isinstance(recording, mido.MidiTrack):
                mid = mido.MidiFile(ticks_per_beat=1000)
                track = mido.MidiTrack()
                track.append(mido.MetaMessage('set_tempo', tempo=1000000, time=0))
                track.extend(recording)
                mid.tracks.append(track)
                mid.save(s.get_absolute_path(f"measure_{measure_idx}/section_{section_idx}/pass_{pass_idx}.mid"))
        self.pass_index[measure_idx] = self.pass_index.get(measure_idx, {})
        self.pass_index[measure_idx][section_idx] = self.pass_index[measure_idx].get(section_idx, 0) + 1
