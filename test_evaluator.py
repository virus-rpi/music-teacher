import zipfile
import os
from mido import MidiFile
from evaluator import Evaluator

SAVE_FILE = 'save.mtsf'
MIDI_NAME = 'song.mid'
EXTRACT_DIR = 'extracted_test_midi'

# Ensure extract dir exists
os.makedirs(EXTRACT_DIR, exist_ok=True)

# Extract song.mid from save.mtsf
with zipfile.ZipFile(SAVE_FILE, 'r') as zf:
    if MIDI_NAME in zf.namelist():
        zf.extract(MIDI_NAME, EXTRACT_DIR)
        midi_path = os.path.join(EXTRACT_DIR, MIDI_NAME)
    else:
        raise FileNotFoundError(f"{MIDI_NAME} not found in {SAVE_FILE}")

# Load MIDI file
midi = MidiFile(midi_path)

# For test, use the same track for both reference and recording
# If there are multiple tracks, use the first two for reference
tracks = midi.tracks
if len(tracks) < 1:
    raise ValueError("No tracks found in MIDI file.")

recording = midi.merged_track
reference = (tracks[0], tracks[1])

# Run evaluation
evaluator = Evaluator(recording, reference)
eval_result = evaluator.full_evaluation

print("Evaluation Result:")
print(eval_result)

