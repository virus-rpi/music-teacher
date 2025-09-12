import pygame
import mido
import threading

LOWEST_NOTE = 21   # A0
HIGHEST_NOTE = 108 # C8
TOTAL_KEYS = HIGHEST_NOTE - LOWEST_NOTE + 1

WHITE_KEY_COLOR = (230, 230, 230)
BLACK_KEY_COLOR = (40, 40, 40)
BG_COLOR = (20, 20, 20)
HIGHLIGHT_COLOR = (0, 200, 255)
BLACK_KEY_HIGHLIGHT_COLOR = (0, 100, 180)
FADE_DURATION = 1500  # milliseconds

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
KEY_CORNER_RADIUS = 6

pressed_keys = {}
pressed_fade_keys = {}

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

pedals = {"soft": False, "sostenuto": False, "sustain": False}
PEDAL_CC = {"sustain": 64, "sostenuto": 66, "soft": 67}
PEDAL_LABELS = {"soft": "Soft", "sostenuto": "Sostenuto", "sustain": "Sustain"}
PEDAL_COLOR = (180, 180, 180)
PEDAL_ACTIVE_COLOR = (0, 200, 255)
PEDAL_WIDTH = int(WHITE_KEY_WIDTH * 1.2)
PEDAL_HEIGHT = int(WHITE_KEY_HEIGHT * 0.6)
PEDAL_SPACING = int(WHITE_KEY_WIDTH * 0.2)
PEDAL_CORNER_RADIUS = 12
PEDAL_Y = 20 + WHITE_KEY_HEIGHT + 30

def is_black(note_number):
    return note_names[note_number % 12].endswith("#")

def note_name(note_number):
    octave = (note_number // 12) - 1
    return f"{note_names[note_number % 12]}{octave}"

def interpolate_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def draw_piano():
    screen.fill(BG_COLOR)
    now = pygame.time.get_ticks()
    white_key_rects = []
    black_key_rects = []
    white_index = 0
    for midi_note in range(LOWEST_NOTE, HIGHEST_NOTE + 1):
        if not is_black(midi_note):
            x = white_index * WHITE_KEY_WIDTH
            rect = pygame.Rect(x, 20, WHITE_KEY_WIDTH, WHITE_KEY_HEIGHT)
            color = WHITE_KEY_COLOR
            if pressed_keys.get(midi_note, False):
                if midi_note in pressed_fade_keys:
                    elapsed = now - pressed_fade_keys[midi_note]
                    if elapsed < FADE_DURATION:
                        t = elapsed / FADE_DURATION
                        color = interpolate_color(HIGHLIGHT_COLOR, WHITE_KEY_COLOR, t)
                    else:
                        del pressed_fade_keys[midi_note]
            pygame.draw.rect(screen, color, rect, border_radius=KEY_CORNER_RADIUS)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1, border_radius=KEY_CORNER_RADIUS)
            white_key_rects.append((midi_note, rect))
            white_index += 1
    white_index = 0
    for midi_note in range(LOWEST_NOTE, HIGHEST_NOTE):
        if not is_black(midi_note):
            x = white_index * WHITE_KEY_WIDTH
            if note_names[midi_note % 12] not in ['E', 'B']:
                black_x = x + WHITE_KEY_WIDTH * 0.7
                rect = pygame.Rect(black_x, 20,
                                   BLACK_KEY_WIDTH, BLACK_KEY_HEIGHT)
                color = BLACK_KEY_COLOR
                if pressed_keys.get(midi_note + 1, False):
                    if (midi_note + 1) in pressed_fade_keys:
                        elapsed = now - pressed_fade_keys[midi_note + 1]
                        if elapsed < FADE_DURATION:
                            t = elapsed / FADE_DURATION
                            color = interpolate_color(BLACK_KEY_HIGHLIGHT_COLOR, BLACK_KEY_COLOR, t)
                        else:
                            del pressed_fade_keys[midi_note + 1]
                pygame.draw.rect(screen, color, rect, border_radius=KEY_CORNER_RADIUS)
                black_key_rects.append((midi_note + 1, rect))
            white_index += 1
    pedal_names = ["soft", "sostenuto", "sustain"]
    total_width = PEDAL_WIDTH * 3 + PEDAL_SPACING * 2
    start_x = (SCREEN_WIDTH - total_width) // 2
    for i, pedal in enumerate(pedal_names):
        x = start_x + i * (PEDAL_WIDTH + PEDAL_SPACING)
        rect = pygame.Rect(x, PEDAL_Y, PEDAL_WIDTH, PEDAL_HEIGHT)
        color = PEDAL_ACTIVE_COLOR if pedals[pedal] else PEDAL_COLOR
        pygame.draw.rect(screen, color, rect, border_radius=PEDAL_CORNER_RADIUS)
        pygame.draw.rect(screen, (0, 0, 0), rect, 2, border_radius=PEDAL_CORNER_RADIUS)

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
            elif msg.type in ("note_off", "note_on"):
                pressed_keys[msg.note] = False
                pressed_fade_keys.pop(msg.note, None)
            elif msg.type == "control_change":
                for pedal, cc in PEDAL_CC.items():
                    if msg.control == cc:
                        pedals[pedal] = msg.value >= 64

midi_thread = threading.Thread(target=midi_listener, daemon=True)
midi_thread.start()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
    draw_piano()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
