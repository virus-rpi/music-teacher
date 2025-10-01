from collections import defaultdict
from dataclasses import dataclass
import mido
from sheet_music import SheetMusicRenderer
from save_system import SaveSystem

@dataclass
class MeasureData:
    chords: tuple[list[tuple[int, str]], ...]= ()
    times: tuple[int, ...] = ()
    xs: tuple[int, ...] = ()
    start_x: int = 0
    end_x: int = 0
    start_index: int = 0
    end_index: int = 0
    midi_msgs: dict[int, list[mido.Message]] = ()

class MidiTeacher:
    def __init__(self, midi_path, sheet_music_renderer: SheetMusicRenderer, save_system: SaveSystem = None):
        self.save_system = save_system or SaveSystem()
        self.midi_path = self.save_system.load_midi_path() or midi_path
        self.sheet_music_renderer = sheet_music_renderer
        self.midi = mido.MidiFile(self.midi_path)
        self.chords, self.chord_times = self._extract_chords()
        self._current_index = 0
        self._pending_notes = set()
        self.loop_enabled = False
        self.loop_start = 0
        self.loop_end = max(0, len(self.chords) - 1)
        self._last_wrapped = False
        self.measures: list[MeasureData] = []
        self._set_measure_data()

    def _extract_chords(self):
        events = []
        note_on_times = {}
        note_durations = []
        for track_idx, track in enumerate(self.midi.tracks):
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0:
                    events.append((abs_time, msg.note, track_idx))
                    note_on_times[(msg.note, track_idx, abs_time)] = abs_time
                elif (msg.type == 'note_off') or (msg.type == 'note_on' and getattr(msg, 'velocity', 0) == 0):
                    candidates = [(k, v) for k, v in note_on_times.items() if k[0] == msg.note and k[1] == track_idx]
                    if candidates:
                        (note, t_idx, onset_tick), _ = max(candidates, key=lambda x: x[1])
                        note_durations.append((msg.note, track_idx, onset_tick, abs_time))
                        del note_on_times[(note, t_idx, onset_tick)]
        events.sort()
        chords_dict = {}
        for t, note, track_idx in events:
            chords_dict.setdefault(t, []).append((note, track_idx))
        sorted_times = sorted(chords_dict.keys())
        track_note_counts = {}
        for _, note, track_idx in events:
            track_note_counts[track_idx] = track_note_counts.get(track_idx, 0) + 1
        sorted_tracks = sorted(track_note_counts, key=track_note_counts.get, reverse=True)
        if len(sorted_tracks) < 2:
            hand_map = {sorted_tracks[0]: 'R'} if sorted_tracks else {}
        else:
            hand_map = {sorted_tracks[0]: 'R', sorted_tracks[1]: 'L'}
        chords = []
        times = []
        chords_with_durations = []
        for t in sorted_times:
            notes = chords_dict[t]
            chord = [(note, hand_map.get(track_idx, 'R')) for note, track_idx in notes]
            chords.append(chord)
            times.append(t)
            chord_notes_with_durations = []
            for note, track_idx in notes:
                match = next(((n, hand_map.get(track_idx, 'R'), onset, offset)
                              for n, t_idx, onset, offset in note_durations
                              if n == note and t_idx == track_idx and onset == t), None)
                if match:
                    chord_notes_with_durations.append(match)
                else:
                    chord_notes_with_durations.append((note, hand_map.get(track_idx, 'R'), t, t))
            chords_with_durations.append(chord_notes_with_durations)
        self.chord_notes_with_durations = chords_with_durations
        return chords, times

    def get_next_notes(self):
        if self.current_index < len(self.chords):
            return {note: hand for note, hand in self.chords[self.current_index]}
        return {}

    def reset(self):
        self.current_index = 0
        self._pending_notes = set()
        self._last_wrapped = False

    def get_progress(self):
        total = len(self.chords)
        if total == 0:
            return 1.0
        return self.current_index / max(1, total - 1)

    def advance_if_pressed(self, pressed_notes):
        self._last_wrapped = False
        next_notes = self.get_next_notes()
        if next_notes and set(next_notes.keys()).issubset(pressed_notes):
            self.current_index += 1
            if self.loop_enabled and self.current_index > self.loop_end:
                self.current_index = self.loop_start
                self._last_wrapped = True

    def advance_one(self):
        self._last_wrapped = False
        if not self.chords:
            return False
        if self.current_index >= len(self.chords) - 1 and not self.loop_enabled:
            return False
        candidate = self.current_index + 1
        if self.loop_enabled and candidate > self.loop_end:
            self.current_index = self.loop_start
            self._last_wrapped = True
        else:
            self.current_index = min(candidate, len(self.chords) - 1)
        return True

    @property
    def current_index(self):
        return self._current_index

    @current_index.setter
    def current_index(self, value):
        if not self.chords:
            self._current_index = 0
        else:
            self._current_index = max(0, min(int(value), len(self.chords) - 1))
        if self.sheet_music_renderer is not None:
            self.sheet_music_renderer.seek_to_index(self._current_index)

    def seek_to_index(self, index: int):
        if not self.chords:
            self.current_index = 0
            return
        self.current_index = index
        self._last_wrapped = False

    def seek_relative(self, delta: int):
        self.current_index = self.current_index + int(delta)

    def seek_to_progress(self, progress: float):
        if not self.chords:
            return
        p = max(0.0, min(1.0, float(progress)))
        idx = int(round(p * (len(self.chords) - 1)))
        self.seek_to_index(idx)

    def set_loop_start(self):
        self.set_loop_start_index(self.current_index)

    def set_loop_end(self):
        self.set_loop_end_index(self.current_index)

    def set_loop_start_index(self, index: int):
        print(f"Set loop start to {index}")
        if not self.chords:
            self.loop_start = 0
            self.loop_end = 0
            return
        idx = max(0, min(int(index), len(self.chords) - 1))
        self.loop_start = idx
        if self.loop_end < self.loop_start:
            self.loop_end = self.loop_start
        self._last_wrapped = False

    def set_loop_end_index(self, index: int):
        if not self.chords:
            self.loop_start = 0
            self.loop_end = 0
            return
        idx = max(0, min(int(index), len(self.chords) - 1))
        self.loop_end = idx
        if self.loop_start > self.loop_end:
            self.loop_start = self.loop_end
        self._last_wrapped = False

    def toggle_loop(self):
        self.loop_enabled = not self.loop_enabled
        self._last_wrapped = False

    def get_loop_range(self):
        return self.loop_start, self.loop_end, self.loop_enabled

    def get_total_chords(self):
        return len(self.chords)

    def get_current_index(self):
        return int(self.current_index)

    def did_wrap_and_clear(self) -> bool:
        """Returns True if it looped around and then clears the flag."""
        w = bool(self._last_wrapped)
        self._last_wrapped = False
        return w

    def _get_measure_tick_boundaries(self):
        """Returns a list of (start_tick, end_tick) for each measure in the MIDI file."""
        ticks_per_beat = self.midi.ticks_per_beat
        numerator = 4
        denominator = 4
        time_sig_changes = []
        for track in self.midi.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == 'time_signature':
                    time_sig_changes.append((abs_tick, msg.numerator, msg.denominator))
        time_sig_changes.sort()
        measure_boundaries: list[tuple[int, int]] = []
        current_tick = 0
        current_numerator = numerator
        current_denominator = denominator
        next_time_sig_idx = 0
        last_tick = self.chord_times[-1] if self.chord_times else 0
        while current_tick <= last_tick + ticks_per_beat:
            if (next_time_sig_idx < len(time_sig_changes) and
                current_tick >= time_sig_changes[next_time_sig_idx][0]):
                _, current_numerator, current_denominator = time_sig_changes[next_time_sig_idx]
                next_time_sig_idx += 1
            beats_per_measure = current_numerator
            beat_length = ticks_per_beat * 4 // current_denominator
            measure_length = beats_per_measure * beat_length
            start_tick = current_tick
            end_tick = current_tick + measure_length
            measure_boundaries.append((start_tick, end_tick))
            current_tick = end_tick
        return measure_boundaries

    def _set_measure_data(self):
        if not self.sheet_music_renderer:
            return
        measure_data = self.sheet_music_renderer.measure_data
        notehead_xs = self.sheet_music_renderer.notehead_xs

        measure_tick_boundaries = self._get_measure_tick_boundaries()
        self.measures: list[MeasureData] = []
        chord_times = self.chord_times
        chords = self.chords
        chord_idx = 0
        for i, measure_info in enumerate(measure_data):
            if measure_info is None or len(measure_info) != 3:
                self.measures.append(MeasureData())
                continue
            start_x, end_x, _ = measure_info
            if i >= len(measure_tick_boundaries):
                self.measures.append(MeasureData())
                continue
            start_tick, end_tick = measure_tick_boundaries[i]
            measure_chord_indices: list[int] = []
            while chord_idx < len(chord_times) and chord_times[chord_idx] < end_tick:
                if chord_times[chord_idx] >= start_tick:
                    measure_chord_indices.append(chord_idx)
                chord_idx += 1
            measure_chords = [chords[j] for j in measure_chord_indices]
            measure_chord_times = [chord_times[j] - start_tick for j in measure_chord_indices]
            measure_note_xs: list[int] = [notehead_xs[j] for j in measure_chord_indices] if notehead_xs else []
            if measure_chord_indices:
                start_index = measure_chord_indices[0]
                end_index = measure_chord_indices[-1] + 1
            else:
                start_index = end_index = chord_idx

            measure_midi_msgs = {i: track for i, track in enumerate(self.get_midi_messages_between_indices(start_index, end_index))}

            self.measures.append(MeasureData(
                chords=tuple(measure_chords),
                times=tuple(measure_chord_times),
                xs=tuple(measure_note_xs),
                start_x=start_x,
                end_x=end_x,
                start_index=start_index,
                end_index=end_index,
                midi_msgs=dict(measure_midi_msgs),
            ))

    def get_notes_for_measure(self, measure_index, unpacked=True) -> tuple[tuple, tuple, tuple, tuple[int, int], tuple[int, int], dict[int, list[mido.Message]]] | MeasureData:
        """Returns (chords, times, note_xs, (start_x, end_x), (start_index, end_index), midi_msgs) for the given measure index."""
        if 0 <= measure_index < len(self.measures):
            if unpacked:
                measure_data = self.measures[measure_index]
                return measure_data.chords, measure_data.times, measure_data.xs, (measure_data.start_x, measure_data.end_x), (measure_data.start_index, measure_data.end_index), measure_data.midi_msgs
            else:
                return self.measures[measure_index]
        return (), (), (), (0, 0), (0, 0), {} if unpacked else MeasureData()

    def get_performed_notes_for_measure(self, measure_index, section, pass_num):
        """Load performed notes from the corresponding MIDI file for a specific measure, section, and pass."""
        if not self.save_system:
            return []
        try:
            with self.save_system.guided_teacher_data as s:
                mid = mido.MidiFile(s.get_absolute_path(f"measure_{measure_index}/section_{section}/pass_{pass_num}.mid"))
                return list(mid.tracks[0]) if mid.tracks else []

        except (FileNotFoundError, OSError, Exception) as e:
            print(f"Failed to load performed notes for measure {measure_index}, section {section}, pass {pass_num}: {e}")
            return []

    def get_midi_messages_between_indices(self, start_idx: int, end_idx: int) -> tuple[mido.MidiTrack, mido.MidiTrack]:
        """
        Returns two mido.MidiTrack objects (right_hand_track, left_hand_track) containing all MIDI messages
        between the absolute times of the given chord indices (inclusive start, exclusive end).
        """
        if not self.chord_times or start_idx >= len(self.chord_times) or end_idx > len(self.chord_times):
            return mido.MidiTrack(), mido.MidiTrack()
        start_tick = self.chord_times[start_idx]
        if end_idx >= len(self.chord_times):
            end_tick = self.chord_times[-1] + 1
        else:
            end_tick = self.chord_times[end_idx]

        measure_midi_msgs = defaultdict(mido.MidiTrack)
        for track_index, track in enumerate(self.midi.tracks):
            abs_tick = 0
            for msg in track:
                abs_tick += getattr(msg, 'time', 0)
                if start_tick <= abs_tick < end_tick and msg.type in ('note_on', 'note_off'):
                    measure_midi_msgs[track_index].append(msg.copy(time=abs_tick - start_tick))
        return measure_midi_msgs[0], measure_midi_msgs[1]
