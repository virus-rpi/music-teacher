from typing import Optional
from mido import MidiTrack
from mt_types import Note, PedalEvent, pedal_type


def extract_notes_and_pedal(track: MidiTrack, mark: Optional[str] = None) -> tuple[list[Note], list[PedalEvent]]:
    """
    Extract note events and pedal control changes from a MIDI track, returning notes with absolute onset and duration and pedal events with absolute times relative to the track start.
    
    Parameters:
        track (MidiTrack): MIDI track to parse.
        mark (Optional[str]): Optional label to attach to each returned Note.
    
    Returns:
        tuple[list[Note], list[PedalEvent]]: A tuple where the first element is a list of Note objects (pitch, onset_ms, duration_ms, velocity, mark) and the second element is a list of PedalEvent objects (time_ms, value, pedal_type). Notes and pedal times are measured from the start of the track in the same time units provided by the track messages.
    
    Notes:
        - If the same pitch receives a new note-on while it is already active, the previously active note is closed at that time.
        - Any notes still active at the end of the track are closed at the track's final time and included in the returned notes.
    """
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
