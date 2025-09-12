import mido

class MidiTeacher:
    def __init__(self, midi_path):
        self.midi_path = midi_path
        self.chords = self._extract_chords()
        self.current_index = 0
        self._pending_notes = set()

    def _extract_chords(self):
        mid = mido.MidiFile(self.midi_path)
        chords = []
        events = []
        abs_time = 0
        for track in mid.tracks:
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    events.append((abs_time, msg.note))
        events.sort()
        chords_dict = {}
        for t, note in events:
            chords_dict.setdefault(t, set()).add(note)
        chords = [list(notes) for t, notes in sorted(chords_dict.items())]
        return chords

    def get_next_notes(self):
        if self.current_index < len(self.chords):
            return set(self.chords[self.current_index])
        return set()

    def advance_if_pressed(self, pressed_notes):
        next_notes = self.get_next_notes()
        if next_notes and next_notes.issubset(pressed_notes):
            self.current_index += 1
            return True
        return False

    def reset(self):
        self.current_index = 0
        self._pending_notes = set()

    def get_progress(self):
        """Returns progress as a float between 0 and 1."""
        if not self.chords:
            return 1.0
        return self.current_index / len(self.chords)
