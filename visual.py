import pygame

WHITE_KEY_COLOR = (230, 230, 230)
BLACK_KEY_COLOR = (40, 40, 40)
BG_COLOR = (20, 20, 20)
HIGHLIGHT_COLOR = (0, 200, 255)
BLACK_KEY_HIGHLIGHT_COLOR = (0, 100, 180)
FADE_DURATION = 1500  # milliseconds
KEY_CORNER_RADIUS = 6
PEDAL_COLOR = (180, 180, 180)
PEDAL_ACTIVE_COLOR = (0, 200, 255)
PEDAL_CORNER_RADIUS = 12

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def is_black(note_number):
    return note_names[note_number % 12].endswith("#")

def interpolate_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def draw_piano(screen, pressed_keys, pressed_fade_keys, pedals, dims):
    screen_width, screen_height = dims['SCREEN_WIDTH'], dims['SCREEN_HEIGHT']
    WHITE_KEY_WIDTH = dims['WHITE_KEY_WIDTH']
    WHITE_KEY_HEIGHT = dims['WHITE_KEY_HEIGHT']
    BLACK_KEY_WIDTH = dims['BLACK_KEY_WIDTH']
    BLACK_KEY_HEIGHT = dims['BLACK_KEY_HEIGHT']
    PEDAL_WIDTH = dims['PEDAL_WIDTH']
    PEDAL_HEIGHT = dims['PEDAL_HEIGHT']
    PEDAL_SPACING = dims['PEDAL_SPACING']
    PEDAL_Y = dims['PEDAL_Y']
    LOWEST_NOTE = dims['LOWEST_NOTE']
    HIGHEST_NOTE = dims['HIGHEST_NOTE']

    screen.fill(BG_COLOR)
    now = pygame.time.get_ticks()
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
            white_index += 1
    white_index = 0
    for midi_note in range(LOWEST_NOTE, HIGHEST_NOTE):
        if not is_black(midi_note):
            x = white_index * WHITE_KEY_WIDTH
            if note_names[midi_note % 12] not in ['E', 'B']:
                black_x = x + WHITE_KEY_WIDTH * 0.7
                rect = pygame.Rect(black_x, 20, BLACK_KEY_WIDTH, BLACK_KEY_HEIGHT)
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
            white_index += 1
    pedal_names = ["soft", "sostenuto", "sustain"]
    total_width = PEDAL_WIDTH * 3 + PEDAL_SPACING * 2
    start_x = (screen_width - total_width) // 2
    for i, pedal in enumerate(pedal_names):
        x = start_x + i * (PEDAL_WIDTH + PEDAL_SPACING)
        rect = pygame.Rect(x, PEDAL_Y, PEDAL_WIDTH, PEDAL_HEIGHT)
        color = PEDAL_ACTIVE_COLOR if pedals[pedal] else PEDAL_COLOR
        pygame.draw.rect(screen, color, rect, border_radius=PEDAL_CORNER_RADIUS)
        pygame.draw.rect(screen, (0, 0, 0), rect, 2, border_radius=PEDAL_CORNER_RADIUS)

