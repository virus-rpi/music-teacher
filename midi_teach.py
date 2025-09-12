import mido

class MidiTeacher:
    def __init__(self, midi_path):
        self.midi_path = midi_path
        self.chords = self._extract_chords()
        self.current_index = 0
        self._pending_notes = set()

    def _extract_chords(self):
        mid = mido.MidiFile(self.midi_path)
        events = []
        # Collect note_on events with track index
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
            # Return dict: {note: hand}
            return {note: hand for note, hand in self.chords[self.current_index]}
        return {}

    def advance_if_pressed(self, pressed_notes):
        next_notes = self.get_next_notes()
        if next_notes and set(next_notes.keys()).issubset(pressed_notes):
            self.current_index += 1
            return True
        return False

    def reset(self):
        self.current_index = 0
        self._pending_notes = set()

    def get_progress(self):
        if not self.chords:
            return 1.0
        return self.current_index / len(self.chords)
