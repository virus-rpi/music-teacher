import pygame
import mido
import threading
from visual import draw_piano
from synth import Synth, PEDAL_CC

LOWEST_NOTE = 21   # A0
HIGHEST_NOTE = 108 # C8
TOTAL_KEYS = HIGHEST_NOTE - LOWEST_NOTE + 1
SOUNDFONT_PATH = "/home/u200b/Music/Sound fonts/GeneralUser-GS.sf2"

pygame.init()
info = pygame.display.Info()
SCREEN_WIDTH, SCREEN_HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("MIDI Piano Visualizer")
clock = pygame.time.Clock()

WHITE_KEY_WIDTH = SCREEN_WIDTH / 52
WHITE_KEY_HEIGHT = int(WHITE_KEY_WIDTH * 7)
BLACK_KEY_WIDTH = WHITE_KEY_WIDTH * 0.6
BLACK_KEY_HEIGHT = WHITE_KEY_HEIGHT * 0.65
PEDAL_WIDTH = int(WHITE_KEY_WIDTH * 1.2)
PEDAL_HEIGHT = int(WHITE_KEY_HEIGHT * 0.6)
PEDAL_SPACING = int(WHITE_KEY_WIDTH * 0.2)
PEDAL_Y = 20 + WHITE_KEY_HEIGHT + 30

pressed_keys = {}
pressed_fade_keys = {}
pedals = {"soft": False, "sostenuto": False, "sustain": False}
synth_enabled = True

dims = {
    'SCREEN_WIDTH': SCREEN_WIDTH,
    'SCREEN_HEIGHT': SCREEN_HEIGHT,
    'WHITE_KEY_WIDTH': WHITE_KEY_WIDTH,
    'WHITE_KEY_HEIGHT': WHITE_KEY_HEIGHT,
    'BLACK_KEY_WIDTH': BLACK_KEY_WIDTH,
    'BLACK_KEY_HEIGHT': BLACK_KEY_HEIGHT,
    'PEDAL_WIDTH': PEDAL_WIDTH,
    'PEDAL_HEIGHT': PEDAL_HEIGHT,
    'PEDAL_SPACING': PEDAL_SPACING,
    'PEDAL_Y': PEDAL_Y,
    'LOWEST_NOTE': LOWEST_NOTE,
    'HIGHEST_NOTE': HIGHEST_NOTE
}

synth = Synth(SOUNDFONT_PATH)

def midi_listener():
    try:
        port_name = mido.get_input_names()[1]
        print(f"Opening MIDI input: {port_name}")
    except IndexError:
        print("No MIDI input found.")
        return
    with mido.open_input(port_name) as inport:
        for msg in inport:
            print(msg)
            if msg.type == "note_on" and msg.velocity > 0:
                pressed_keys[msg.note] = True
                pressed_fade_keys[msg.note] = pygame.time.get_ticks()
                if synth_enabled:
                    synth.note_on(msg.note, msg.velocity)
            elif msg.type in ("note_off", "note_on"):
                pressed_keys[msg.note] = False
                pressed_fade_keys.pop(msg.note, None)
                if synth_enabled:
                    synth.note_off(msg.note)
            elif msg.type == "control_change":
                for pedal, cc in PEDAL_CC.items():
                    if msg.control == cc:
                        pedals[pedal] = msg.value >= 64
                        if synth_enabled:
                            synth.pedal_cc(msg.control, msg.value)

midi_thread = threading.Thread(target=midi_listener, daemon=True)
midi_thread.start()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                synth_enabled = not synth_enabled
                print(f"Synth enabled: {synth_enabled}")
    draw_piano(screen, pressed_keys, pressed_fade_keys, pedals, dims)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
synth.delete()

