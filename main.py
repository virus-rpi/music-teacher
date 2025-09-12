import pygame
import mido
import threading
from visual import draw_piano, draw_progress_bar
from synth import Synth, PEDAL_CC
from midi_teach import MidiTeacher
from sheet_music import SheetMusicRenderer

LOWEST_NOTE = 21   # A0
HIGHEST_NOTE = 108 # C8
TOTAL_KEYS = HIGHEST_NOTE - LOWEST_NOTE + 1
SOUNDFONT_PATH = "/home/u200b/Music/Sound fonts/GeneralUser-GS.sf2"
MIDI_TEACH_PATH = "/home/u200b/Music/Credits Song For My Death.mid"

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
PIANO_Y_OFFSET = 24 + 28 + 16  # progress bar margin + bar height + extra spacing
PEDAL_Y = PIANO_Y_OFFSET + WHITE_KEY_HEIGHT + 30

pressed_keys = {}
pressed_fade_keys = {}
pedals = {"soft": False, "sostenuto": False, "sustain": False}
synth_enabled = True
teaching_mode = True  # Set to True to enable teaching mode
pressed_notes_set = set()

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
    'HIGHEST_NOTE': HIGHEST_NOTE,
    'PIANO_Y_OFFSET': PIANO_Y_OFFSET
}

synth = Synth(SOUNDFONT_PATH)
midi_teacher = MidiTeacher(MIDI_TEACH_PATH)
sheet_music_renderer = SheetMusicRenderer(MIDI_TEACH_PATH, SCREEN_WIDTH)

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
                pressed_notes_set.add(msg.note)
                if teaching_mode:
                    next_notes = midi_teacher.get_next_notes()
                    if msg.note in next_notes:
                        if synth_enabled:
                            synth.note_on(msg.note, msg.velocity)
                    else:
                        if synth_enabled:
                            synth.play_error_sound()
                else:
                    if synth_enabled:
                        synth.note_on(msg.note, msg.velocity)
                if teaching_mode:
                    midi_teacher.advance_if_pressed(pressed_notes_set)
            elif msg.type in ("note_off", "note_on"):
                pressed_keys[msg.note] = False
                pressed_fade_keys.pop(msg.note, None)
                pressed_notes_set.discard(msg.note)
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
            if event.key == pygame.K_t:
                teaching_mode = not teaching_mode
                print(f"Teaching mode: {teaching_mode}")
                midi_teacher.reset()
                pressed_notes_set.clear()
            # Debug: advance teacher by one chord when pressing 'd'
            if event.key == pygame.K_d and teaching_mode:
                advanced = midi_teacher.advance_one()
                if advanced:
                    print("[Debug] Advanced teacher by one chord.")
                else:
                    print("[Debug] Already at end; cannot advance.")
    highlighted_notes = midi_teacher.get_next_notes() if teaching_mode else set()
    draw_piano(screen, pressed_keys, pressed_fade_keys, pedals, dims, highlighted_notes)
    draw_progress_bar(screen, midi_teacher.get_progress(), dims) if teaching_mode else None
    sheet_music_renderer.draw(screen, dims['PEDAL_Y'] + dims['PEDAL_HEIGHT'] + 32, midi_teacher.get_progress())
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
synth.delete()
