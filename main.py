import pygame
import mido
import threading
from visual import draw_piano, draw_ui_overlay, BG_COLOR, draw_guided_mode_overlay
from synth import Synth, PEDAL_CC
from midi_teach import MidiTeacher
from sheet_music import SheetMusicRenderer
from guided_teacher import GuidedTeacher
import math

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
SHEET_Y = 24 + 28 + 64
PIANO_Y_OFFSET = SHEET_Y + WHITE_KEY_HEIGHT + PEDAL_HEIGHT
PEDAL_Y = PIANO_Y_OFFSET + WHITE_KEY_HEIGHT + 30

pressed_keys = {}
pressed_fade_keys = {}
pedals = {"soft": False, "sostenuto": False, "sustain": False}
synth_enabled = True
teaching_mode = True
guided_mode = False
pressed_notes_set = set()
pressed_note_events = []

piano_y_default = PIANO_Y_OFFSET
piano_y_center = (SCREEN_HEIGHT - WHITE_KEY_HEIGHT - PEDAL_HEIGHT) // 2
piano_y_current = float(piano_y_default)
piano_y_target = float(piano_y_default)

overlay_alpha_current = 1.0 if teaching_mode else 0.0
overlay_alpha_target = overlay_alpha_current
sheet_alpha_current = 1.0 if teaching_mode else 0.0
sheet_alpha_target = sheet_alpha_current
piano_tau = 0.12
alpha_tau = 0.18

last_time_ms = pygame.time.get_ticks()

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
    'PIANO_Y_OFFSET': PIANO_Y_OFFSET,
    'SHEET_Y': SHEET_Y
}

font_small = pygame.font.SysFont("Segoe UI", 16)
font_medium = pygame.font.SysFont("Segoe UI", 20, bold=True)

def render():
    global last_time_ms, piano_y_current, piano_y_target, overlay_alpha_current, overlay_alpha_target, sheet_alpha_current, sheet_alpha_target
    now_ms = pygame.time.get_ticks()
    dt = max(0.0, (now_ms - last_time_ms) / 1000.0)
    last_time_ms = now_ms

    screen.fill(BG_COLOR)

    piano_y_target = piano_y_default if teaching_mode else piano_y_center
    overlay_alpha_target = 1.0 if teaching_mode else 0.0
    sheet_alpha_target = 1.0 if teaching_mode else 0.0
    if dt > 0.0:
        a = 1.0 - math.exp(-dt / max(1e-6, piano_tau))
        piano_y_current += (piano_y_target - piano_y_current) * a
        b = 1.0 - math.exp(-dt / max(1e-6, alpha_tau))
        overlay_alpha_current += (overlay_alpha_target - overlay_alpha_current) * b
        sheet_alpha_current += (sheet_alpha_target - sheet_alpha_current) * b
    dims['PIANO_Y_OFFSET'] = piano_y_current
    dims['PEDAL_Y'] = int(piano_y_current + WHITE_KEY_HEIGHT + 30)
    draw_piano(screen, pressed_keys, pressed_fade_keys, pedals, dims,
               midi_teacher.get_next_notes() if teaching_mode else set())
    draw_ui_overlay(screen, midi_teacher, dims, font_small, font_medium, alpha=overlay_alpha_current)
    if guided_mode:
        draw_guided_mode_overlay(screen, guided_teacher, sheet_music_renderer, dims)
    sheet_music_renderer.draw(screen, dims.get('SHEET_Y', 0), midi_teacher.get_progress(), guided_teacher, sheet_alpha_current)

    pygame.display.flip()


synth = Synth(SOUNDFONT_PATH, render)
sheet_music_renderer = SheetMusicRenderer(MIDI_TEACH_PATH, SCREEN_WIDTH)
midi_teacher = MidiTeacher(MIDI_TEACH_PATH, sheet_music_renderer)
guided_teacher = GuidedTeacher(midi_teacher, synth)

def midi_listener():
    try:
        port_name = mido.get_input_names()[1]
        print(f"Opening MIDI input: {port_name}")
    except IndexError:
        print("No MIDI input found.")
        return
    with mido.open_input(port_name) as in_port:
        for msg in in_port:
            print(msg)
            if msg.type == "note_on" and msg.velocity > 0:
                pressed_keys[msg.note] = True
                pressed_fade_keys[msg.note] = pygame.time.get_ticks()
                pressed_notes_set.add(msg.note)
                pressed_note_events.append((msg.note, pygame.time.get_ticks()))
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
    if guided_mode:
        guided_teacher.update(pressed_notes_set, pressed_note_events)
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
            if event.key == pygame.K_g:
                guided_mode = not guided_mode
                if guided_mode:
                    guided_teacher.start()
                else:
                    guided_teacher.stop()
                print(f"Guided mode: {guided_mode}")
            if event.key == pygame.K_d and teaching_mode:
                advanced = midi_teacher.advance_one()
                if advanced:
                    print("[Debug] Advanced teacher by one chord.")
                else:
                    print("[Debug] Already at end; cannot advance.")

            # Seeking controls
            if event.key in (pygame.K_RIGHT, pygame.K_LEFT):
                # compute step based on modifiers
                step = 1
                mods = pygame.key.get_mods()
                if mods & pygame.KMOD_SHIFT:
                    step = 10
                elif mods & pygame.KMOD_CTRL:
                    step = 5
                if event.key == pygame.K_LEFT:
                    step = -step
                midi_teacher.seek_relative(step)

            # loop controls: set start/end and toggle
            if event.key == pygame.K_COMMA:  # ','
                midi_teacher.set_loop_start()
                ls, le, _ = midi_teacher.get_loop_range()
                print(f"Set loop start to {ls}")
            if event.key == pygame.K_PERIOD:  # '.'
                midi_teacher.set_loop_end()
                ls, le, _ = midi_teacher.get_loop_range()
                print(f"Set loop end to {le}")
            if event.key == pygame.K_l:
                midi_teacher.toggle_loop()
                print(f"Loop enabled: {midi_teacher.loop_enabled}")

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            bar_height = 28
            bar_margin = 24
            bar_width = int(SCREEN_WIDTH * 0.7)
            bx = int((SCREEN_WIDTH - bar_width) / 2)
            by = bar_margin
            if bx <= mx <= bx + bar_width and by <= my <= by + bar_height:
                rel = (mx - bx) / float(bar_width)
                rel = max(0.0, min(1.0, rel))
                total = midi_teacher.get_total_chords()
                if total > 0:
                    idx = int(round(rel * (total - 1)))
                else:
                    idx = 0

                mods = pygame.key.get_mods()
                if event.button == 1 and not (mods & (pygame.KMOD_SHIFT | pygame.KMOD_CTRL)):
                    midi_teacher.seek_to_progress(rel)
                elif event.button == 1 and (mods & pygame.KMOD_SHIFT):
                    # shift+left-click: set loop start
                    midi_teacher.set_loop_start_index(idx)
                    ls, le, _ = midi_teacher.get_loop_range()
                    print(f"Set loop start to {ls}")
                elif event.button == 1 and (mods & pygame.KMOD_CTRL):
                    # ctrl+left-click: set loop end
                    midi_teacher.set_loop_end_index(idx)
                    ls, le, _ = midi_teacher.get_loop_range()
                    print(f"Set loop end to {le}")
                elif event.button == 2:
                    # middle click: set loop start
                    midi_teacher.set_loop_start_index(idx)
                    ls, le, _ = midi_teacher.get_loop_range()
                    print(f"Set loop start to {ls}")
                elif event.button == 3:
                    # right-click: set the loop end
                    midi_teacher.set_loop_end_index(idx)
                    ls, le, _ = midi_teacher.get_loop_range()
                    print(f"Set loop end to {le}")

                pressed_notes_set.clear()

    render()
    clock.tick(60)

pygame.quit()
synth.delete()
