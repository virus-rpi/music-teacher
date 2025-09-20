import fluidsynth
import time
import threading

PEDAL_CC = {"sustain": 64, "sostenuto": 66, "soft": 67}

class Synth:
    def __init__(self, soundfont_path, render_callback):
        self.fs = fluidsynth.Synth(gain=0.8)
        self.fs.start(driver="alsa")
        self.sfid = self.fs.sfload(soundfont_path)
        if self.sfid < 0:
            print(f"Error: Could not load SoundFont at {soundfont_path}")
        else:
            print(f"Loaded SoundFont {soundfont_path} with id {self.sfid}")
        self.fs.program_select(0, self.sfid, 0, 0)
        print(f"Selected program 0 on channel 0, bank 0, preset 0")
        self.render_callback = render_callback

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

    def play_measure(self, measure_index, midi_teacher, set_index_callback=None, reset_to_index=None):
        chords, times, _, _, (start_index, _) = midi_teacher.get_notes_for_measure(measure_index)
        
        self.play_notes(chords, times, start_index=start_index if set_index_callback else None, set_index_callback=set_index_callback, reset_to_index=reset_to_index)

    def play_notes(self, chords, times, start_index=None, set_index_callback=None, reset_to_index=None):
        """Play a sequence of chords with their corresponding times. The times are in milliseconds and say when a cord should be played relative to the first chord.
        If notes_xs and set_index_callback are provided, set_index_callback is called with the index in notes_xs corresponding to the current chord as each chord is played."""
        if not chords or not times:
            raise ValueError("Error: chords and times must be non-empty")
        if len(chords) != len(times):
            raise ValueError("Error: chords and times must have the same length")
        if start_index is not None and set_index_callback is None:
            raise ValueError("Error: set_index_callback must be provided if start_index is provided")
        if reset_to_index is not None and set_index_callback is None:
            raise ValueError("Error: set_index_callback must be provided if reset_to_index is provided")

        stop_event = threading.Event()
        def player():
            start_time = time.time()
            time_offset = times[0]

            for i, chord in enumerate(chords):
                chord_time = times[i] - time_offset
                while (time.time() - start_time) * 1000 < chord_time:
                    time.sleep(0.001)
                if start_index is not None and set_index_callback is not None:
                    set_index_callback(start_index + i)
                for note, hand in chord:
                    self.note_on(note, 100)
            time.sleep(0.1)
            if reset_to_index is not None:
                set_index_callback(reset_to_index)
            stop_event.set()
        play_tread = threading.Thread(target=player)
        play_tread.start()

        while not stop_event.is_set():
            self.render_callback()
            time.sleep(1 / 60)

        play_tread.join()

    def delete(self):
        self.fs.delete()
