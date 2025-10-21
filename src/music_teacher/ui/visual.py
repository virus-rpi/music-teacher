import pygame
import pygame.gfxdraw

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

note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def is_black(note_number):
    return note_names[note_number % 12].endswith("#")


def interpolate_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_hand_marker(screen, x, y, width, height, hand, border_radius):
    indicator_color = (0, 140, 255) if hand == "L" else (255, 60, 60)
    inner_rect = pygame.Rect(int(x) + 2, int(y) + 2, int(width) - 4, int(height) - 4)
    pygame.draw.rect(
        screen, indicator_color, inner_rect, border_radius=int(border_radius)
    )
    font_size = int(height * 0.7)
    font = pygame.font.SysFont("Segoe UI", font_size, bold=True)
    text = font.render(hand, True, (255, 255, 255))
    text_rect = text.get_rect(center=(int(x + width // 2), int(y + height // 2)))
    screen.blit(text, text_rect)


def draw_piano(
    screen, pressed_keys, pressed_fade_keys, pedals, dims, highlighted_notes=None
):
    highlighted_notes = highlighted_notes or {}
    screen_width, screen_height = dims["SCREEN_WIDTH"], dims["SCREEN_HEIGHT"]
    white_key_width = dims["WHITE_KEY_WIDTH"]
    white_key_height = dims["WHITE_KEY_HEIGHT"]
    black_key_width = dims["BLACK_KEY_WIDTH"]
    black_key_height = dims["BLACK_KEY_HEIGHT"]
    pedal_width = dims["PEDAL_WIDTH"]
    pedal_height = dims["PEDAL_HEIGHT"]
    pedal_spacing = dims["PEDAL_SPACING"]
    pedal_y = dims["PEDAL_Y"]
    lowest_note = dims["LOWEST_NOTE"]
    highest_note = dims["HIGHEST_NOTE"]
    piano_y_offset = int(dims.get("PIANO_Y_OFFSET", 0))

    bg_top = max(0, int(piano_y_offset) - 12)
    bg_height = int(white_key_height + pedal_height + 60)
    pygame.draw.rect(screen, BG_COLOR, (0, bg_top, screen_width, bg_height))

    now = pygame.time.get_ticks()
    white_index = 0
    for midi_note in range(lowest_note, highest_note + 1):
        if not is_black(midi_note):
            x = white_index * white_key_width
            rect = pygame.Rect(x, piano_y_offset, white_key_width, white_key_height)
            color = WHITE_KEY_COLOR
            hand = highlighted_notes.get(midi_note)
            if hand:
                color = (0, 140, 255) if hand == "L" else (255, 60, 60)
            elif pressed_keys.get(midi_note, False):
                if midi_note in pressed_fade_keys:
                    elapsed = now - pressed_fade_keys[midi_note]
                    if elapsed < FADE_DURATION:
                        t = elapsed / FADE_DURATION
                        color = interpolate_color(HIGHLIGHT_COLOR, WHITE_KEY_COLOR, t)
                    else:
                        del pressed_fade_keys[midi_note]
            try:
                pygame.draw.rect(screen, color, rect, border_radius=KEY_CORNER_RADIUS)
            except ValueError:
                continue
            pygame.draw.rect(
                screen, (0, 0, 0), rect, 1, border_radius=KEY_CORNER_RADIUS
            )
            white_index += 1
    white_index = 0
    for midi_note in range(lowest_note, highest_note):
        if not is_black(midi_note):
            x = white_index * white_key_width
            if note_names[midi_note % 12] not in ["E", "B"]:
                black_x = x + white_key_width * 0.7
                rect = pygame.Rect(
                    black_x, piano_y_offset, black_key_width, black_key_height
                )
                color = BLACK_KEY_COLOR  # Always start with default
                hand = highlighted_notes.get(midi_note + 1)
                if hand:
                    color = (0, 140, 255) if hand == "L" else (255, 60, 60)
                if pressed_keys.get(midi_note + 1, False):
                    if (midi_note + 1) in pressed_fade_keys:
                        elapsed = now - pressed_fade_keys[midi_note + 1]
                        if elapsed < FADE_DURATION:
                            t = elapsed / FADE_DURATION
                            color = interpolate_color(
                                BLACK_KEY_HIGHLIGHT_COLOR, BLACK_KEY_COLOR, t
                            )
                        else:
                            del pressed_fade_keys[midi_note + 1]
                pygame.draw.rect(screen, color, rect, border_radius=KEY_CORNER_RADIUS)
            white_index += 1
    pedal_names = ["soft", "sostenuto", "sustain"]
    total_width = pedal_width * 3 + pedal_spacing * 2
    start_x = (screen_width - total_width) // 2
    for i, pedal in enumerate(pedal_names):
        x = start_x + i * (pedal_width + pedal_spacing)
        rect = pygame.Rect(x, pedal_y, pedal_width, pedal_height)
        color = PEDAL_ACTIVE_COLOR if pedals[pedal] else PEDAL_COLOR
        pygame.draw.rect(screen, color, rect, border_radius=PEDAL_CORNER_RADIUS)
        pygame.draw.rect(screen, (0, 0, 0), rect, 2, border_radius=PEDAL_CORNER_RADIUS)


def draw_progress_bar(surface, progress, dims):
    screen_width = dims["SCREEN_WIDTH"]
    bar_height = 28
    bar_margin = 24
    bar_width = int(screen_width * 0.7)
    x = int((screen_width - bar_width) / 2)
    y = bar_margin

    mask_surface = pygame.Surface((bar_width, bar_height), pygame.SRCALPHA)
    pygame.draw.rect(
        mask_surface, (255, 255, 255), mask_surface.get_rect(), border_radius=14
    )

    shadow_offset = 6
    shadow_color = (0, 0, 0, 80)
    shadow_surface = pygame.Surface((bar_width, bar_height), pygame.SRCALPHA)
    pygame.draw.rect(
        shadow_surface, shadow_color, shadow_surface.get_rect(), border_radius=14
    )
    surface.blit(shadow_surface, (x + shadow_offset, y + shadow_offset))

    bg_color = (30, 30, 50, 255)
    bg_surface = pygame.Surface((bar_width, bar_height), pygame.SRCALPHA)
    pygame.draw.rect(bg_surface, bg_color, bg_surface.get_rect(), border_radius=14)
    bg_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    surface.blit(bg_surface, (x, y))

    fill_width = int(bar_width * progress)
    if fill_width > 0:
        fill_surface = pygame.Surface((fill_width, bar_height), pygame.SRCALPHA)
        pygame.draw.rect(
            fill_surface,
            (0, 200, 255),
            (0, 0, fill_width, bar_height),
            border_radius=14,
        )
        fill_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        surface.blit(fill_surface, (x, y))

    font = pygame.font.SysFont("Segoe UI", 22, bold=True)
    percent = int(progress * 100)
    text = font.render(
        f"{percent}%", True, (255, 255, 255) if progress < 0.49 else (0, 0, 0)
    )
    text_rect = text.get_rect(center=(screen_width // 2, y + bar_height // 2))
    surface.blit(text, text_rect)


def draw_ui_overlay(
    screen,
    midi_teacher,
    dims,
    guided_teacher,
    font_small=None,
    font_medium=None,
    alpha=1.0,
):
    """Draw the top UI overlay: progress bar, loop markers, status and instructions.
    Draw everything onto an overlay surface, apply the requested alpha, then blit to the screen.
    """
    # create fonts if not supplied
    if font_small is None:
        font_small = pygame.font.SysFont("Segoe UI", 16)
    if font_medium is None:
        font_medium = pygame.font.SysFont("Segoe UI", 20, bold=True)

    screen_width = dims["SCREEN_WIDTH"]
    screen_height = dims["SCREEN_HEIGHT"]
    bar_height = 28
    bar_margin = 24
    bar_width = int(screen_width * 0.7)
    x = int((screen_width - bar_width) / 2)
    y = bar_margin

    total = midi_teacher.get_total_chords()
    cur = midi_teacher.get_current_index()
    progress = midi_teacher.get_progress() if total > 0 else 0.0

    overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    draw_progress_bar(overlay, progress, dims)

    loop_start, loop_end, loop_enabled = midi_teacher.get_loop_range()
    if total > 0 and loop_enabled:
        start_px = x + int((loop_start / max(1, total - 1)) * bar_width)
        end_px = x + int((loop_end / max(1, total - 1)) * bar_width)
        color = (0, 200, 255) if loop_enabled else (120, 120, 120)
        pygame.draw.line(overlay, color, (start_px, y), (start_px, y + bar_height), 3)
        pygame.draw.line(overlay, color, (end_px, y), (end_px, y + bar_height), 3)
        shade = pygame.Surface((max(1, end_px - start_px), bar_height), pygame.SRCALPHA)
        shade.fill((*color, 40))
        overlay.blit(shade, (start_px, y))

    margin = 16

    status = (
        f"Chord {cur}/{total}   Loop: {'ON' if loop_enabled else 'OFF'}"
        if not guided_teacher.is_active
        else f"Chord {cur}/{total}   Auto-Advance: {'ON' if guided_teacher.auto_advance else 'OFF'}"
    )
    txt = font_medium.render(status, True, (220, 220, 220))
    txt_w, txt_h = txt.get_size()
    status_y = max(margin, screen_height - margin - txt_h * 4)
    overlay.blit(txt, (margin, status_y))

    if guided_teacher.is_active:
        instr_lines = [
            "Press R to replay the current measure.",
            "Press SPACE to play the section notes.",
            "Press ENTER to advance to the next task.",
            "Press SHIFT+ENTER to go back to the previous task.",
            "Press A to toggle auto-advance.",
        ]
    else:
        instr_lines = [
            "Click progress bar to seek.",
            "<- / -> : step by 1 chord. Shift for 10, Ctrl for 5.",
            ", : set loop start at current. . : set loop end at current. L : toggle loop.",
        ]
    instr_lines += [
        "T : toggle teaching mode. G: toggle guidance mode. D : debug advance. S : toggle synth."
    ]

    line_h = font_small.get_height()
    padding_between = 4
    total_h = (len(instr_lines) + 1) * (line_h + padding_between)
    instr_top = screen_height - margin - total_h - txt_h * 2
    for i, line in enumerate(instr_lines):
        it = font_small.render(line, True, (180, 180, 180))
        iw, ih = it.get_size()
        px = screen_width - margin - iw
        py = instr_top + i * (line_h + padding_between)
        overlay.blit(it, (px, py))
    aa = max(0.0, min(1.0, float(alpha)))
    overlay.set_alpha(int(aa * 255))
    screen.blit(overlay, (0, 0))


def draw_guided_mode_overlay(screen, guided_teacher, sheet_music_renderer, dims):
    if not guided_teacher.is_active:
        return
    score = guided_teacher.get_last_score()
    if score is not None:
        font = pygame.font.SysFont("Segoe UI", 24, bold=True)
        score_text = f"Score: {score * 100:.0f}%"
        text = font.render(score_text, True, (255, 255, 255))
        text_rect = text.get_rect(
            center=(
                dims["SCREEN_WIDTH"] // 2,
                dims["SHEET_Y"] + sheet_music_renderer.strip_height + 30,
            )
        )
        screen.blit(text, text_rect)
    guide_text = guided_teacher.get_guide_text()
    if guide_text:
        font = pygame.font.SysFont("Segoe UI", 16)
        text = font.render(guide_text, True, (255, 255, 255))
        text_rect = text.get_rect(
            center=(
                dims["SCREEN_WIDTH"] // 2,
                dims["SHEET_Y"] + sheet_music_renderer.strip_height + 60,
            )
        )
        screen.blit(text, text_rect)
