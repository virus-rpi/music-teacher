import mido

# Helper function: MIDI note number -> note name
def note_name(note_number):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note_number // 12) - 1
    note = notes[note_number % 12]
    return f"{note}{octave}"

print("Available MIDI input ports:")
for port in mido.get_input_names():
    print(port)

with mido.open_input("ARIUS:ARIUS MIDI 1 24:0") as inport:
    print(f"Input port: {inport.name}")
    print("Listening for MIDI input... Press Ctrl+C to quit.")
    for msg in inport:
        if msg.type == 'note_on' and msg.velocity > 0:  # key pressed
            print(f"Key pressed: {note_name(msg.note)} (MIDI {msg.note})")
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            print(f"Key released: {note_name(msg.note)} (MIDI {msg.note})")
