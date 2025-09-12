import sys
import fluidsynth

PEDAL_CC = {"sustain": 64, "sostenuto": 66, "soft": 67}


def get_fluidsynth_driver():
    if sys.platform.startswith("linux"):
        for driver in ["alsa", "pulseaudio", "portaudio", "oss", "jack", "pipewire", "sdl3"]:
            try:
                test_fs = fluidsynth.Synth()
                test_fs.start(driver=driver)
                test_fs.delete()
                print(f"Using FluidSynth audio driver: {driver}")
                return driver
            except Exception:
                continue
        raise RuntimeError("No suitable FluidSynth audio driver found on Linux.")
    elif sys.platform.startswith("win"):
        return "dsound"
    elif sys.platform.startswith("darwin"):
        return "coreaudio"
    else:
        return None


class Synth:
    def __init__(self, soundfont_path):
        self.driver = get_fluidsynth_driver()
        self.fs = fluidsynth.Synth()
        if self.driver:
            self.fs.start(driver=self.driver)
        else:
            self.fs.start()
        self.sfid = self.fs.sfload(soundfont_path)
        self.fs.program_select(0, self.sfid, 0, 0)

    def note_on(self, note, velocity):
        self.fs.noteon(0, note, velocity)

    def note_off(self, note):
        self.fs.noteoff(0, note)

    def pedal_cc(self, control, value):
        self.fs.cc(0, control, value)

    def delete(self):
        self.fs.delete()

