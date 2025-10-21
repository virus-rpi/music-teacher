from typing import Optional
from mido import MidiTrack
from mt_types import Note, PedalEvent, pedal_type


def extract_notes_and_pedal(track: MidiTrack, mark: Optional[str] = None) -> tuple[list[Note], list[PedalEvent]]:
    notes = []
    pedals = []
    current_time = 0
    note_on = {}

    pedal_types: dict[int, pedal_type] = {64: "sustain", 66: "sostenuto", 67: "soft"}

    for msg in track:
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            note_on[msg.note] = (current_time, msg.velocity)
        elif msg.type in ("note_off", "note_on") and msg.note in note_on:
            start_time, velocity = note_on.pop(msg.note)
            duration = current_time - start_time
            notes.append(Note(
                pitch=msg.note,
                onset_ms=start_time,
                duration_ms=duration,
                velocity=velocity,
                mark=mark
            ))
        elif msg.type == "control_change" and msg.control in pedal_types:
            pedals.append(PedalEvent(
                time_ms=current_time,
                value=msg.value,
                pedal_type=pedal_types[msg.control]
            ))

    # Handle any notes that didn't get a note_off
    while note_on:
        pitch, (start_time, velocity) = note_on.popitem()
        duration = current_time - start_time
        notes.append(Note(
            pitch=pitch,
            onset_ms=start_time,
            duration_ms=duration,
            velocity=velocity,
            mark=mark
        ))

    return notes, pedals

