import mido
from sheet_music import SheetMusicRenderer


class MidiTeacher:
    def __init__(self, midi_path, sheet_music_renderer: SheetMusicRenderer):
        self.midi_path = midi_path
        self.sheet_music_renderer = sheet_music_renderer
        self.chords, self.chord_times = self._extract_chords()
        self._current_index = 0
        self._pending_notes = set()
        self.loop_enabled = False
        self.loop_start = 0
        self.loop_end = max(0, len(self.chords) - 1)
        self._last_wrapped = False
        self.measures = []
        self._set_measure_data()

    def _extract_chords(self):
        mid = mido.MidiFile(self.midi_path)
        events = []
        for track_idx, track in enumerate(mid.tracks):
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0:
                    events.append((abs_time, msg.note, track_idx))
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
        for t in sorted_times:
            notes = chords_dict[t]
            chord = [(note, hand_map.get(track_idx, 'R')) for note, track_idx in notes]
            chords.append(chord)
            times.append(t)
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
        w = bool(self._last_wrapped)
        self._last_wrapped = False
        return w

    def get_chord_times(self):
        return list(self.chord_times)

    def _set_measure_data(self):
        if not self.sheet_music_renderer:
            return
        measure_data = self.sheet_music_renderer.measure_data
        notehead_xs = self.sheet_music_renderer.notehead_xs

        self.measures = []
        current_chord_idx = 0

        for measure_info in measure_data:
            if measure_info is None or len(measure_info) != 3:
                self.measures.append(([], [], [], (None, None)))
                continue
            start_x, end_x, n_notes = measure_info
            count = int(n_notes) if n_notes is not None else 0
            if count == 0:
                self.measures.append(([], [], [], (start_x, end_x)))
                continue
            end_idx = min(current_chord_idx + count, len(self.chords))
            measure_chords = self.chords[current_chord_idx:end_idx]
            measure_chord_times = [measure_time - self.chord_times[current_chord_idx:end_idx][0] for measure_time in self.chord_times[current_chord_idx:end_idx]]
            measure_note_xs = notehead_xs[current_chord_idx:end_idx] if notehead_xs else []
            self.measures.append((measure_chords, measure_chord_times, measure_note_xs, (start_x, end_x), (current_chord_idx, end_idx)))
            current_chord_idx = end_idx

    def get_notes_for_measure(self, measure_index) -> tuple[list, list, list, tuple[int, int]]:
        """Returns (chords, times, note_xs, (start_x, end_x), (start_index, end_index)) for the given measure index."""
        if 0 <= measure_index < len(self.measures):
            return self.measures[measure_index]
        return [], [], [], (0, 0), (0, 0)
