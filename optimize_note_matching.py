import os
import zipfile
import numpy as np
from mido import MidiFile
from evaluator import _extract_notes_and_pedal, Note
import random
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend (change to 'Qt5Agg' or 'GTK3Agg' if needed)
import matplotlib.pyplot as plt
plt.ion()

# ============================================
# =============== INIT PART ==================
# ============================================

SAVE_FILE = 'save.mtsf'
MIDI_NAME = 'song.mid'
EXTRACT_DIR = 'extracted_test_midi'
os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(SAVE_FILE, 'r') as zf:
    if MIDI_NAME in zf.namelist():
        zf.extract(MIDI_NAME, EXTRACT_DIR)
        midi_path = os.path.join(EXTRACT_DIR, MIDI_NAME)
    else:
        raise FileNotFoundError(f"{MIDI_NAME} not found in {SAVE_FILE}")

midi = MidiFile(midi_path)
tracks = midi.tracks
reference = (tracks[0], tracks[1])

ref_rh, _ = _extract_notes_and_pedal(reference[0], mark="rh")
ref_lh, _ = _extract_notes_and_pedal(reference[1], mark="lh")
ref_notes = sorted(ref_rh + ref_lh, key=lambda n: n.onset_ms)


# ============================================
# ======== SIMULATE REALISTIC RECORDING ======
# ============================================

def simulate_recording_from_reference(ref_notes):
    """Create a simulated human-like recording from the reference."""
    rec_notes = []
    for n in ref_notes:
        # Random small timing jitter (+/- 50 ms typical)
        onset_jitter = random.gauss(0, 30)
        duration_scale = 1.0 + random.gauss(0, 0.1)
        velocity_jitter = random.randint(-10, 10)
        # Create noisy duplicate
        rec_notes.append(
            Note(
                pitch=n.pitch,
                onset_ms=int(n.onset_ms + onset_jitter),
                duration_ms=max(20, int(n.duration_ms * duration_scale)),
                velocity=max(1, min(127, n.velocity + velocity_jitter)),
                mark=n.mark
            )
        )
    # Randomly drop and add some notes
    if len(rec_notes) > 10:
        if random.random() < 0.2:
            rec_notes.pop(random.randrange(len(rec_notes)))
        if random.random() < 0.2:
            rec_notes.append(random.choice(ref_notes))
    random.shuffle(rec_notes)
    return rec_notes


rec_notes = simulate_recording_from_reference(ref_notes)

# ============================================
# ============= NORMALIZATION PART ============
# ============================================

ref_onsets = np.array([n.onset_ms for n in ref_notes])
rec_onsets = np.array([n.onset_ms for n in rec_notes])

if len(ref_onsets) == 0 or len(rec_onsets) == 0:
    print("Warning: One of the note lists is empty, skipping normalization.")
    rec_notes_norm = rec_notes
else:
    ref_min, ref_max = np.min(ref_onsets), np.max(ref_onsets)
    rec_min, rec_max = np.min(rec_onsets), np.max(rec_onsets)
    ref_span, rec_span = ref_max - ref_min, rec_max - rec_min
    scale = (ref_span / rec_span) if rec_span > 0 else 1.0
    print(f"Normalization scale: {scale:.4f}")
    rec_notes_norm = [
        Note(
            pitch=n.pitch,
            onset_ms=int(ref_min + (n.onset_ms - rec_min) * scale),
            duration_ms=int(n.duration_ms * scale),
            velocity=n.velocity,
            mark=n.mark,
        )
        for n in rec_notes
    ]

print(f"Reference notes: {len(ref_notes)}, Recording (simulated): {len(rec_notes)}")


# ============================================
# ============ MATCHING FUNCTION ==============
# ============================================

def match_lists(A, B, distance_fn, null_cost):
    """Hungarian matching with null penalties."""
    m, n = len(A), len(B)
    size = max(m, n)
    cost_matrix = np.full((size, size), null_cost, dtype=float)
    for i in range(m):
        for j in range(n):
            cost_matrix[i, j] = distance_fn(A[i], B[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < m and c < n and cost_matrix[r, c] < null_cost:
            matches.append((A[r], B[c]))
        elif r < m:
            matches.append((A[r], None))
        elif c < n:
            matches.append((None, B[c]))
    return matches


# ============================================
# ============ EVALUATION FUNCTION ============
# ============================================

def evaluate_matches(matches):
    """Compute a weighted total matching score."""
    correct_notes = sum(1 for r, p in matches if r and p and r.pitch == p.pitch)
    total_notes = sum(1 for r, _ in matches if r)
    accuracy_score = correct_notes / max(1, total_notes)

    timing_devs = [abs(p.onset_ms - r.onset_ms) for r, p in matches if r and p]
    duration_devs = [abs(p.duration_ms - r.duration_ms) for r, p in matches if r and p]

    avg_timing_dev = np.mean(timing_devs) if timing_devs else 0
    avg_duration_dev = np.mean(duration_devs) if duration_devs else 0

    timing_score = max(0, 1 - avg_timing_dev / 200)
    duration_score = max(0, 1 - avg_duration_dev / 200)

    total = 0.5 * accuracy_score + 0.3 * timing_score + 0.2 * duration_score
    return total, (accuracy_score, timing_score, duration_score)


# ============================================
# ========== DEBUG VISUALIZATION ==============
# ============================================

_plot_notes_3d_fig = None
_plot_notes_3d_ax = None
_plot_notes_3d_initialized = False

def plot_notes_3d(ref_notes, rec_notes, matches=None):
    global _plot_notes_3d_fig, _plot_notes_3d_ax, _plot_notes_3d_initialized
    if not ref_notes or not rec_notes:
        print("No notes to plot.")
        return
    if _plot_notes_3d_fig is None or _plot_notes_3d_ax is None:
        _plot_notes_3d_fig = plt.figure()
        _plot_notes_3d_ax = _plot_notes_3d_fig.add_subplot(111, projection='3d')
        _plot_notes_3d_initialized = False
    else:
        _plot_notes_3d_ax.cla()
    ax = _plot_notes_3d_ax
    ax.set_xlabel("Pitch")
    ax.set_ylabel("Onset (ms)")
    ax.set_zlabel("End (ms)")
    ax.set_title("Reference vs Recording (3D) - Interactive (Zoom/Rotate)")
    ref_xyz = np.array([[n.pitch, n.onset_ms, n.onset_ms + n.duration_ms] for n in ref_notes])
    rec_xyz = np.array([[n.pitch, n.onset_ms, n.onset_ms + n.duration_ms] for n in rec_notes])
    ax.scatter(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2], c='b', label="Reference")
    ax.scatter(rec_xyz[:, 0], rec_xyz[:, 1], rec_xyz[:, 2], c='r', label="Recording")
    if matches:
        for r, p in matches:
            if r and p:
                ax.plot(
                    [r.pitch, p.pitch],
                    [r.onset_ms, p.onset_ms],
                    [r.onset_ms + r.duration_ms, p.onset_ms + p.duration_ms],
                    c='g', alpha=0.6
                )
    ax.legend()
    plt.tight_layout()
    _plot_notes_3d_fig.canvas.draw()
    _plot_notes_3d_fig.canvas.flush_events()
    if not _plot_notes_3d_initialized:
        plt.show(block=False)
        _plot_notes_3d_initialized = True
    plt.pause(0.001)


# ============================================
# ========== WEIGHT TRAINING (GA) ============
# ============================================

generations = 30
population_size = 12
initial_mutation_scale = 100
final_mutation_scale = 0.1

def get_mutation_scale(gen):
    return initial_mutation_scale * ((final_mutation_scale / initial_mutation_scale) ** (gen / generations))

# Initial parameters
base_params = [0.01, 3.0, 0.5, 15.0]  # onset_w, pitch_w, dur_w, null_cost
population = [base_params[:]]
best_score = -1
best_params = None

print(f"\nTraining {generations} generations...")

for gen in range(generations + 1):
    mutation_scale = get_mutation_scale(gen)
    results = []
    for idx, params in enumerate(population):
        onset_w, pitch_w, dur_w, k = params
        def weighted_distance(a, b):
            if a is None or b is None:
                return k
            return (
                pitch_w * abs(a.pitch - b.pitch)
                + onset_w * abs(a.onset_ms - b.onset_ms)
                + dur_w * abs((a.onset_ms + a.duration_ms) - (b.onset_ms + b.duration_ms))
            )

        matches = match_lists(ref_notes, rec_notes_norm, weighted_distance, k)
        score, _ = evaluate_matches(matches)
        results.append((score, params, matches))
        if score > best_score:
            best_score, best_params = score, params[:]
        # Interactive plot update after each individual's evaluation
        plot_notes_3d(ref_notes, rec_notes_norm, matches)
        plt.pause(0.01)
    results.sort(reverse=True, key=lambda x: x[0])
    top = results[: population_size // 2]

    print(f"Gen {gen:02d} | Best={best_score:.4f} Params={best_params}")

    # Mutation for next generation
    new_pop = [best_params[:]]
    for _, p, _ in top:
        for _ in range(2):
            mutated = [
                max(1e-4, p[0] + random.gauss(0, mutation_scale * 0.1)),
                max(0.01, p[1] + random.gauss(0, mutation_scale * 0.01)),
                max(1e-4, p[2] + random.gauss(0, mutation_scale * 0.1)),
                max(0.1, p[3] + random.gauss(0, mutation_scale * 0.1))
            ]
            new_pop.append(mutated)
    population = new_pop[:population_size]

print("\n=== Final Results ===")
print(f"Best score: {best_score:.4f}")
print(f"Best params: onset_w={best_params[0]:.3f}, pitch_w={best_params[1]:.3f}, dur_w={best_params[2]:.3f}, k={best_params[3]:.3f}")

# Final visualization
if best_params:
    def best_distance(a, b):
        if a is None or b is None:
            return best_params[3]
        return (
            best_params[1] * abs(a.pitch - b.pitch)
            + best_params[0] * abs(a.onset_ms - b.onset_ms)
            + best_params[2] * abs((a.onset_ms + a.duration_ms) - (b.onset_ms + b.duration_ms))
        )

    final_matches = match_lists(ref_notes, rec_notes_norm, best_distance, best_params[3])
    plot_notes_3d(ref_notes, rec_notes_norm, final_matches)
