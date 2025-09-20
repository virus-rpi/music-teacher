import fluidsynth

PEDAL_CC = {"sustain": 64, "sostenuto": 66, "soft": 67}

class Synth:
    def __init__(self, soundfont_path):
        self.fs = fluidsynth.Synth(gain=0.8)
        self.fs.start(driver="alsa")
        self.sfid = self.fs.sfload(soundfont_path)
        if self.sfid < 0:
            print(f"Error: Could not load SoundFont at {soundfont_path}")
        else:
            print(f"Loaded SoundFont {soundfont_path} with id {self.sfid}")
        self.fs.program_select(0, self.sfid, 0, 0)
        print(f"Selected program 0 on channel 0, bank 0, preset 0")

    def note_on(self, note, velocity):
        self.fs.noteon(0, note, velocity)

    def note_off(self, note):
        self.fs.noteoff(0, note)

    def pedal_cc(self, control, value):
        self.fs.cc(0, control, value)

    def play_error_sound(self):
        self.fs.program_select(9, self.sfid, 128, 0)
        self.fs.noteon(9, 81, 127)
        self.fs.noteon(9, 71, 127)
        self.fs.program_select(0, self.sfid, 0, 0)

    def play_measure(self, measure_index, midi_teacher):
        chords, times, _, _ = midi_teacher.get_notes_for_measure(measure_index)
        if not chords:
            return
        
        self.play_notes(chords, times)

    def play_notes(self, chords, times):
        if not chords or not times:
            return

        import time
        start_time = time.time()
        time_offset = times[0]

        for i, chord in enumerate(chords):
            chord_time = times[i] - time_offset
            while (time.time() - start_time)*1000 < chord_time:
                time.sleep(0.001)
            for note, hand in chord:
                self.note_on(note, 100)

    def delete(self):
        self.fs.delete()
