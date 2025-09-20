import time
from collections import deque
from dataclasses import dataclass

import mido
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

        self.timer_start = None
        self.recording = mido.MidiTrack()
        # tracking to avoid repeated on/off messages while a key remains held
        self._prev_pressed_notes = set()
        self._last_record_time = None

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

        if self.teacher.midi_teacher.did_wrap_and_clear():
            self._handle_evaluate()

            self.recording.clear()
            self._prev_pressed_notes.clear()
            self.timer_start = None
            self._last_record_time = None

    def _handle_pygame_events(self, pygame_events):
        for event in pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.teacher.synth.play_measure(self.measure, self.teacher.midi_teacher, self.teacher.midi_teacher.seek_to_index, self.teacher.midi_teacher.get_current_index())
                elif event.key == pygame.K_SPACE:
                    self.teacher.synth.play_notes(self.section.chords, self.section.times, self.start_idx, self.teacher.midi_teacher.seek_to_index, self.teacher.midi_teacher.get_current_index())
                elif event.key == pygame.K_RETURN and (event.mod & pygame.KMOD_SHIFT):
                    self.teacher.previous_task()
                elif event.key == pygame.K_RETURN:
                    self.teacher.next_task()

    def _handle_midi_events(self, pressed_notes: set[int], pressed_note_events):
        if len(self.recording) == 0 and len(pressed_note_events) > 0:
            earliest = min(ev.time for ev in pressed_note_events if hasattr(ev, 'time'))
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
        print("Evaluating...")
        print(self.recording)

        events = []
        abs_ms = 0
        for msg in self.recording:
            abs_ms += getattr(msg, 'time', 0)
            if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0:
                events.append((abs_ms, msg.note))

        threshold_ms = 50
        recorded_onsets = []
        for t, note in events:
            if not recorded_onsets or t - recorded_onsets[-1][0] > threshold_ms:
                recorded_onsets.append([t, {note}])
            else:
                recorded_onsets[-1][1].add(note)
        # Normalize to (time_ms, set)
        recorded_onsets = [(o[0], o[1]) for o in recorded_onsets]

        mt = self.teacher.midi_teacher
        if not self.section.chords:
            score = 1.0
            self.teacher.last_score = score
            if score > 0.8:
                self.teacher.next_task()
            return

        chord_indices = list(range(self.start_idx, self.end_idx + 1))
        chord_ticks = []
        for idx in chord_indices:
            try:
                chord_ticks.append(mt.chord_times[idx])
            except Exception:
                chord_ticks.append(None)

        def tick_to_seconds(tick):
            if tick is None:
                return 0.0
            mid = mido.MidiFile(mt.midi_path)
            ticks_per_beat = mid.ticks_per_beat
            tempo_changes = []
            for track in mid.tracks:
                abs_tick = 0
                for msg in track:
                    abs_tick += msg.time
                    if msg.type == 'set_tempo':
                        tempo_changes.append((abs_tick, msg.tempo))
            tempo_changes.sort()
            cur_tempo = 500000
            last_tick = 0
            seconds = 0.0
            for tc_tick, tc_tempo in tempo_changes:
                if tc_tick >= tick:
                    break
                delta = tc_tick - last_tick
                seconds += (delta * cur_tempo) / (ticks_per_beat * 1e6)
                cur_tempo = tc_tempo
                last_tick = tc_tick
            delta = tick - last_tick
            seconds += (delta * cur_tempo) / (ticks_per_beat * 1e6)
            return seconds

        ref_abs_secs = [tick_to_seconds(t) for t in chord_ticks]
        if any(v is None for v in chord_ticks):
            measure_start_tick = mt.chord_times[self.start_idx] if self.start_idx < len(mt.chord_times) else 0
            ref_abs_secs = [tick_to_seconds(measure_start_tick + t) for t in self.section.times]

        ref_start = ref_abs_secs[0]
        ref_onsets_ms = [int((s - ref_start) * 1000.0) for s in ref_abs_secs]
        ref_chords = [set(n for n, _hand in chord) for chord in self.section.chords]

        if not recorded_onsets:
            score = 0.0
            self.teacher.last_score = score
            if score > 0.8:
                self.teacher.next_task()
            return

        n_ref = len(ref_onsets_ms)
        n_rec = len(recorded_onsets)
        m = min(n_ref, n_rec)

        acc_scores = []
        for i in range(m):
            ref_set = ref_chords[i]
            rec_set = recorded_onsets[i][1]
            if not ref_set and not rec_set:
                acc_scores.append(1.0)
                continue
            inter = len(ref_set & rec_set)
            denom = (len(ref_set) + len(rec_set))
            if denom == 0:
                acc_scores.append(1.0)
            else:
                acc_scores.append((2.0 * inter) / denom)
        if n_rec < n_ref:
            acc_scores.extend([0.0] * (n_ref - n_rec))
        accuracy_score = sum(acc_scores) / max(1, n_ref)

        def intervals(xs):
            return [xs[i+1] - xs[i] for i in range(len(xs)-1)] if len(xs) > 1 else []

        ref_intervals = intervals(ref_onsets_ms[:m])
        rec_intervals = intervals([t for t, _ in recorded_onsets[:m]])
        if not ref_intervals or not rec_intervals:
            relative_score = 1.0
        else:
            mean_ref = sum(ref_intervals) / len(ref_intervals)
            mean_rec = sum(rec_intervals) / len(rec_intervals)
            if mean_ref == 0 or mean_rec == 0:
                relative_score = 0.0
            else:
                nref = [ri / mean_ref for ri in ref_intervals]
                nrec = [ri / mean_rec for ri in rec_intervals]
                L = min(len(nref), len(nrec))
                diffs = [abs(nref[i] - nrec[i]) for i in range(L)]
                avg_diff = sum(diffs) / L
                relative_score = max(0.0, 1.0 - avg_diff)

        ref_total = (ref_onsets_ms[-1] - ref_onsets_ms[0]) if len(ref_onsets_ms) > 1 else 0
        rec_total = (recorded_onsets[m-1][0] - recorded_onsets[0][0]) if m > 1 else 0
        if ref_total == 0:
            absolute_score = 1.0 if rec_total == 0 else 0.0
        else:
            rel_error = abs(rec_total - ref_total) / float(ref_total)
            absolute_score = max(0.0, 1.0 - min(rel_error, 1.0))

        score = (0.7 * accuracy_score) + (0.2 * relative_score) + (0.1 * absolute_score)
        score = max(0.0, min(1.0, score))

        self.teacher.last_score = score
        print(f"Eval: accuracy={accuracy_score:.3f} rel={relative_score:.3f} abs={absolute_score:.3f} -> score={score:.3f}")
        if score > 0.9:
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
                elif event.key == pygame.K_RETURN and (event.mod & pygame.KMOD_SHIFT):
                    self.teacher.previous_task()
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
            if not isinstance(self.current_task, (GenerateNextMeasureTasks, PlaybackMeasureTask)):
                self.history.append(self.current_task)
        if self.tasks:
            self.current_task = self.tasks.popleft()
            self.current_task.on_start()
        else:
            self.current_task = None
            self.stop()

    def previous_task(self):
        if self.history:
            if self.current_task:
                self.current_task.on_end()
                self.tasks.insert(0, self.current_task)
            self.current_task = self.history.pop()
            self.current_task.on_start()

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

    def get_current_task_info(self):
        return self.current_task
