import mido

class MidiTeacher:
    def __init__(self, midi_path):
        self.midi_path = midi_path
        self.chords, self.chord_times = self._extract_chords()
        self.current_index = 0
        self._pending_notes = set()
        self.loop_enabled = False
        self.loop_start = 0
        self.loop_end = max(0, len(self.chords) - 1)
        self._last_wrapped = False

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

    def advance_if_pressed(self, pressed_notes):
        self._last_wrapped = False
        next_notes = self.get_next_notes()
        if next_notes and set(next_notes.keys()).issubset(pressed_notes):
            return self.advance_one()
        return False

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

    def seek_to_index(self, index: int):
        if not self.chords:
            self.current_index = 0
            return
        self.current_index = max(0, min(int(index), len(self.chords) - 1))
        self._last_wrapped = False

    def seek_relative(self, delta: int):
        self.seek_to_index(self.current_index + int(delta))

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

    def get_chords_segment(self, start_idx: int, count: int):
        if not self.chords:
            return [], []
        start = max(0, min(int(start_idx), len(self.chords) - 1))
        end = max(start, min(start + int(count), len(self.chords)))
        return self.chords[start:end], self.chord_times[start:end]

    def get_remaining_chords_count(self, start_idx: int):
        if not self.chords:
            return 0
        return max(0, len(self.chords) - max(0, int(start_idx)))
