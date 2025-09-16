import mido

class MidiTeacher:
    def __init__(self, midi_path):
        self.midi_path = midi_path
        self.chords = self._extract_chords()
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
                if msg.type == 'note_on' and msg.velocity > 0:
                    events.append((abs_time, msg.note, track_idx))
        events.sort()
        chords_dict = {}
        for t, note, track_idx in events:
            chords_dict.setdefault(t, []).append((note, track_idx))
        track_note_counts = {}
        for _, note, track_idx in events:
            track_note_counts[track_idx] = track_note_counts.get(track_idx, 0) + 1
        sorted_tracks = sorted(track_note_counts, key=track_note_counts.get, reverse=True)
        if len(sorted_tracks) < 2:
            # Fallback: all notes right hand
            hand_map = {sorted_tracks[0]: 'R'}
        else:
            hand_map = {sorted_tracks[0]: 'R', sorted_tracks[1]: 'L'}
        chords = []
        for notes in chords_dict.values():
            chord = [(note, hand_map.get(track_idx, 'R')) for note, track_idx in notes]
            chords.append(chord)
        return chords

    def get_next_notes(self):
        if self.current_index < len(self.chords):
            return {note: hand for note, hand in self.chords[self.current_index]}
        return {}

    def advance_if_pressed(self, pressed_notes):
        self._last_wrapped = False
        next_notes = self.get_next_notes()
        if next_notes and set(next_notes.keys()).issubset(pressed_notes):
            self.current_index += 1
            if self.loop_enabled and self.current_index > self.loop_end:
                self.current_index = self.loop_start
                self._last_wrapped = True
            return True
        return False

    def reset(self):
        self.current_index = 0
        self._pending_notes = set()
        self._last_wrapped = False

    def get_progress(self):
        if not self.chords:
            return 1.0
        return self.current_index / len(self.chords)

    # Debug helper: force-advance by one chord regardless of pressed notes
    def advance_one(self):
        self._last_wrapped = False
        if self.current_index < len(self.chords):
            self.current_index += 1
            if self.loop_enabled and self.current_index > self.loop_end:
                self.current_index = self.loop_start
                self._last_wrapped = True
            return True
        return False

    def seek_to_index(self, index: int):
        """Seek directly to a chord index (clamped)."""
        if not self.chords:
            self.current_index = 0
            return
        self.current_index = max(0, min(int(index), len(self.chords) - 1))
        self._last_wrapped = False

    def seek_relative(self, delta: int):
        """Move forward/backward by delta chords."""
        self.seek_to_index(self.current_index + int(delta))

    def seek_to_progress(self, progress: float):
        """Seek to a relative progress (0.0-1.0) mapping to chord index."""
        if not self.chords:
            return
        p = max(0.0, min(1.0, float(progress)))
        idx = int(round(p * (len(self.chords) - 1)))
        self.seek_to_index(idx)

    def set_loop_start(self):
        """Set the loop/practice start to current position."""
        self.set_loop_start_index(self.current_index)

    def set_loop_end(self):
        """Set the loop/practice end to current position."""
        self.set_loop_end_index(self.current_index)

    def set_loop_start_index(self, index: int):
        """Set loop start to a specific chord index (clamped)."""
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
        """Set loop end to a specific chord index (clamped)."""
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
        """Return True if the most recent advance wrapped back to loop_start; clears the flag."""
        w = bool(self._last_wrapped)
        self._last_wrapped = False
        return w

