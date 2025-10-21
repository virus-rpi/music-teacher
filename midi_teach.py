import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import mido
from save_system import SaveSystem
from sheet_music import SheetMusicRenderer
from mt_types import Note, PedalEvent
from midi_utils import extract_notes_and_pedal

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
        """
        Initialize the MidiTeacher, load and preprocess the MIDI file, and prepare playback and measure state.
        
        Initializes internal playback state (current index, pending notes, loop settings), loads the MIDI file (using the provided SaveSystem if it supplies an override path), extracts chord events and their times, builds the tempo map, preprocesses per-track message and note/pedal indices for fast time-range queries, and populates measure data from the sheet music renderer.
        
        Parameters:
            midi_path (str): Path to the MIDI file to load; overridden if `save_system.load_midi_path()` returns a non-empty value.
            sheet_music_renderer (SheetMusicRenderer): Renderer used to compute measure layout and positions.
            save_system (SaveSystem, optional): Persistence helper used to load/save user data and (optionally) a MIDI path. If omitted, a default SaveSystem is created.
        """
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

        self._build_tempo_map()
        self._preprocess_track_indices()
        self._preprocess_notes_and_pedals()
        self._set_measure_data()

    def _extract_chords(self):
        """
        Extracts chords and their onset ticks from the loaded MIDI file.
        
        Scans all tracks for note on/off events, groups simultaneous onsets into chords, and assigns each note a hand label based on track prominence. Also records per-note onset and offset ticks when available.
        
        Returns:
            tuple: (chords, times)
                - chords (list[list[tuple[int, str]]]): List of chords; each chord is a list of (note, hand) tuples where `note` is the MIDI note number and `hand` is 'L' or 'R'.
                - times (list[int]): List of absolute onset ticks corresponding to each chord.
        
        Side effects:
            Populates self.chord_notes_with_durations with a list per chord of (note, hand, onset_tick, offset_tick).
        """
        timer = time.perf_counter()
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
        print(f"Extracted {len(chords)} chords in {time.perf_counter() - timer:.3f} seconds")
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

    def _build_tempo_map(self):
        """Build a global tempo map from all tracks: list of segments with start_tick, ms_per_tick, and cumulative ms."""
        timer = time.perf_counter()
        ticks_per_beat = self.midi.ticks_per_beat
        default_tempo_us_per_beat = 500000  # 120 BPM
        tempo_changes: list[tuple[int, int]] = []
        for track in self.midi.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += getattr(msg, 'time', 0)
                if msg.type == 'set_tempo':
                    tempo_changes.append((abs_tick, int(msg.tempo)))
        tempo_changes.sort(key=lambda x: x[0])
        if not tempo_changes or tempo_changes[0][0] != 0:
            tempo_changes = [(0, default_tempo_us_per_beat)] + tempo_changes
        else:
            pass
        collapsed: list[tuple[int, int]] = []
        for tick, tempo in tempo_changes:
            if collapsed and collapsed[-1][0] == tick:
                collapsed[-1] = (tick, tempo)
            else:
                collapsed.append((tick, tempo))
        self._tempo_changes = collapsed
        self._tempo_segment_starts: list[int] = []
        self._tempo_segments: list[tuple[int, float, float]] = []
        cum_ms = 0.0
        for i, (start_tick, tempo_us) in enumerate(collapsed):
            ms_per_tick = (tempo_us / 1000.0) / float(ticks_per_beat)
            self._tempo_segment_starts.append(start_tick)
            self._tempo_segments.append((start_tick, ms_per_tick, cum_ms))
            if i + 1 < len(collapsed):
                next_tick = collapsed[i + 1][0]
                if next_tick > start_tick:
                    cum_ms += (next_tick - start_tick) * ms_per_tick
        print(f"Built tempo map with {len(self._tempo_segments)} segments in {time.perf_counter() - timer:.3f} seconds")

    def _tick_to_ms(self, tick: int) -> float:
        """Convert an absolute tick to absolute milliseconds using tempo segments."""
        if not self._tempo_segments:
            return 0.0
        idx = bisect_right(self._tempo_segment_starts, tick) - 1
        if idx < 0:
            idx = 0
        start_tick, ms_per_tick, cum_ms_at_start = self._tempo_segments[idx]
        return cum_ms_at_start + (tick - start_tick) * ms_per_tick

    def _tick_duration_to_ms(self, tick: int, duration: int) -> float:
        """Convert an absolute tick and duration to absolute milliseconds using tempo segments."""
        if not self._tempo_segments:
            return 0.0
        return  self._tick_to_ms(tick + duration) -  self._tick_to_ms(tick)

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
        """
        Populate self.measures with MeasureData objects describing sheet-music measures and the chords that lie within each measure.
        
        For each renderer-provided measure, associates chord indices whose onset ticks fall inside the measure's tick boundaries and stores:
        - chords: tuple of chord entries for that measure,
        - times: tuple of chord onset times expressed as ticks relative to the measure start,
        - xs: tuple of notehead x positions when available,
        - start_x / end_x: renderer-provided horizontal bounds for the measure,
        - start_index / end_index: inclusive start index and exclusive end index into the global chord list for the measure.
        
        If no sheet_music_renderer is available, or a measure entry is missing/invalid, an empty MeasureData is appended for that measure. Measures that contain no chords yield start_index == end_index positioned at the current chord scan location.
        """
        timer = time.perf_counter()
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

            self.measures.append(MeasureData(
                chords=tuple(measure_chords),
                times=tuple(measure_chord_times),
                xs=tuple(measure_note_xs),
                start_x=start_x,
                end_x=end_x,
                start_index=start_index,
                end_index=end_index,
            ))
        print(f"Built measure data in {time.perf_counter() - timer:.3f} seconds")

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

    def _preprocess_track_indices(self):
        """
        Builds per-track indices mapping absolute message times to message lists for fast time-range queries.
        
        For each MIDI track, computes absolute millisecond timestamps for every message and produces a parallel list of message copies whose `time` field is the message's delta time expressed in milliseconds. Stores a list of (times_ms, msg_list) tuples on `self._track_msg_indices`, where `times_ms` is a list of absolute times in milliseconds and `msg_list` is the corresponding list of copied messages. Also prints the preprocessing duration.
        """
        timer = time.perf_counter()
        self._track_msg_indices = []
        for track in self.midi.tracks:
            abs_tick = 0
            times_ms = []
            msg_list = []
            for msg in track:
                delta_ms = int(self._tick_duration_to_ms(abs_tick, getattr(msg, 'time', 0)))
                abs_tick += getattr(msg, 'time', 0)
                times_ms.append(int(self._tick_to_ms(abs_tick)))
                msg_list.append(msg.copy(time=delta_ms))
            self._track_msg_indices.append((times_ms, msg_list))
        print(f"Preprocessed track message indices in {time.perf_counter() - timer:.3f} seconds")

    def _preprocess_notes_and_pedals(self):
        """
        Precomputes per-track note and pedal timing in milliseconds and stores indexable structures for fast range queries.
        
        Converts extracted Note.onset_ms and PedalEvent.time_ms from ticks to integer milliseconds for every MIDI track, and stores:
        - self._preprocessed_notes: list of per-track lists of Note objects with onset_ms in ms
        - self._preprocessed_pedals: list of per-track lists of PedalEvent objects with time_ms in ms
        - self._note_onset_indices: list of per-track lists of integer onset_ms values (for bisect queries)
        - self._pedal_time_indices: list of per-track lists of integer time_ms values (for bisect queries)
        
        This method mutates the above attributes and does not return a value.
        """
        timer = time.perf_counter()
        self._preprocessed_notes: list[list[Note]] = []
        self._preprocessed_pedals: list[list[PedalEvent]] = []
        self._note_onset_indices: list[list[int]] = []
        self._pedal_time_indices: list[list[int]] = []

        for track_idx, track in enumerate(self.midi.tracks):
            notes, pedals = extract_notes_and_pedal(track, mark="rh" if track_idx == 0 else "lh" if track_idx == 1 else "unknown")
            notes = list(map(lambda n: n.copy(onset_ms=int(self._tick_to_ms(n.onset_ms))), notes))
            pedals = list(map(lambda p: p.copy(time_ms=int(self._tick_to_ms(p.time_ms))), pedals))
            self._preprocessed_notes.append(notes)
            self._preprocessed_pedals.append(pedals)
            self._note_onset_indices.append([note.onset_ms for note in notes])
            self._pedal_time_indices.append([pedal.time_ms for pedal in pedals])

        print(f"Preprocessed notes and pedals in {time.perf_counter() - timer:.3f} seconds")

    def query_notes_and_pedals(self, start_idx: int, end_idx: int) -> tuple[list[Note], list[PedalEvent]]:
        """
        Return notes and sustain pedal events whose onsets fall between two chord indices.
        
        The window includes events with onset time greater than or equal to the start chord's onset and strictly less than the end chord's onset. If end_idx is greater than or equal to the last chord index, the window extends to the end of the MIDI. Results are aggregated across all tracks and returned sorted by onset time.
        
        Parameters:
            start_idx (int): Index of the starting chord (inclusive).
            end_idx (int): Index of the ending chord (exclusive).
        
        Returns:
            tuple[list[Note], list[PedalEvent]]: A pair where the first element is a list of Note objects and the second is a list of PedalEvent objects; both lists are sorted by their `onset_ms` / `time_ms` fields.
        """
        if not hasattr(self, '_preprocessed_notes'):
            return [], []
        start_ms = int(round(self._tick_to_ms(self.chord_times[start_idx])))
        end_ms = int(round(self._tick_to_ms(
            self.chord_times[-1] + 1 if end_idx >= len(self.chord_times) else self.chord_times[end_idx])))
        notes = []
        pedals = []
        for track_idx in range(len(self._preprocessed_notes)):
            onset_times = self._note_onset_indices[track_idx]
            notes.extend(self._preprocessed_notes[track_idx][bisect_left(onset_times, start_ms):bisect_left(onset_times, end_ms)])
            pedal_times = self._pedal_time_indices[track_idx]
            pedals.extend(self._preprocessed_pedals[track_idx][bisect_left(pedal_times, start_ms):bisect_left(pedal_times, end_ms)])
        return sorted(notes, key=lambda n: n.onset_ms), sorted(pedals, key=lambda p: p.time_ms)