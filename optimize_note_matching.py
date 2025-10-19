import os
import zipfile
import numpy as np
from mido import MidiFile
from evaluator import _extract_notes_and_pedal, Note
import random
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


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
recording = midi.merged_track
reference = (tracks[0], tracks[1])

ref_rh, _ = _extract_notes_and_pedal(reference[0], mark="rh")
ref_lh, _ = _extract_notes_and_pedal(reference[1], mark="lh")
rec_notes, _ = _extract_notes_and_pedal(recording)
ref_notes = sorted(ref_rh + ref_lh, key=lambda note: note.onset_ms)

random.shuffle(rec_notes)
rec_notes_work = rec_notes.copy()

ref_onsets = np.array([n.onset_ms for n in ref_notes])
rec_onsets = np.array([n.onset_ms for n in rec_notes_work])

# Handle empty edge cases
if len(ref_onsets) == 0 or len(rec_onsets) == 0:
    print("Warning: One of the note lists is empty, skipping normalization.")
    rec_notes_norm = rec_notes_work
else:
    ref_min, ref_max = np.min(ref_onsets), np.max(ref_onsets)
    rec_min, rec_max = np.min(rec_onsets), np.max(rec_onsets)
    ref_span = ref_max - ref_min
    rec_span = rec_max - rec_min
    scale = (ref_span / rec_span) if rec_span > 0 else 1.0
    print(f"Normalization scale factor: {scale:.4f}")
    print(f"Ref onset range: {ref_min} → {ref_max} (span {ref_span})")
    print(f"Rec onset range: {rec_min} → {rec_max} (span {rec_span})")
    rec_notes_norm = [
        Note(
            pitch=n.pitch,
            onset_ms=int(ref_min + (n.onset_ms - rec_min) * scale),
            duration_ms=int(n.duration_ms * scale),
            velocity=n.velocity,
            mark=n.mark
        )
        for n in rec_notes_work
    ]
print("Initialized.")


# ============================================
# ============ MATCHING PART =================
# ============================================

def match_lists(A, B, distance_fn, null_cost):
    """
    Match two lists (A, B) to minimize total distance with optional null matches.
    Parameters:
    - A: list of elements (reference list)
    - B: list of elements (to match)
    - distance_fn(a, b): function returning a numeric distance/cost between elements
    - null_cost: numeric cost for leaving an element unmatched (matching to null)
    Returns:
    - matches: list of tuples (a, b) where b can be None if unmatched
    """
    m, n = len(A), len(B)
    size = max(m, n)
    cost_matrix = np.full((size, size), fill_value=null_cost, dtype=float)
    for i in range(m):
        for j in range(n):
            cost_matrix[i, j] = distance_fn(A[i], B[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < m and c < n:
            if cost_matrix[r, c] < null_cost:
                matches.append((A[r], B[c]))
            else:
                matches.append((A[r], None))
        elif r < m:
            matches.append((A[r], None))
        elif c < n:
            matches.append((None, B[c]))
    return matches

# ============================================
# ========== EVALUATION PART =================
# ============================================

accuracy_weight = 0.5
timing_weight = 0.3
duration_weight = 0.2

def evaluate_matches(matches):
    """Compute total matching score (accuracy, timing, duration)."""
    correct_notes = sum(1 for r, p in matches if p and r and r.pitch == p.pitch)
    total_notes = sum(1 for r, _ in matches if r)
    accuracy_score = correct_notes / max(1, total_notes)

    timing_devs = [abs(p.onset_ms - r.onset_ms) for r, p in matches if p and r]
    duration_devs = [abs(p.duration_ms - r.duration_ms) for r, p in matches if p and r]

    avg_timing_dev = np.mean(timing_devs) if timing_devs else 0.0
    avg_duration_dev = np.mean(duration_devs) if duration_devs else 0.0

    timing_score = max(0.0, 1.0 - avg_timing_dev / 200.0)
    duration_score = max(0.0, 1.0 - avg_duration_dev / 200.0)

    total_score = (
        accuracy_weight * accuracy_score +
        timing_weight * timing_score +
        duration_weight * duration_score
    )

    return total_score, (accuracy_score, timing_score, duration_score)


# ============================================
# ========== DEBUGGING PART ==================
# ============================================

def plot_notes_and_matches_3d(ref_notes, rec_notes, matches=None):
    """
    Plot all reference and recorded notes in 3D (pitch, onset, end time).
    If matches is provided, overlay triangles for matched pairs.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Pitch')
    ax.set_ylabel('Onset (ms)')
    ax.set_zlabel('End Time (ms)')
    ax.set_title('Note Matching Visualization')
    # Plot all reference notes
    ref_xyz = np.array([[n.pitch, n.onset_ms, n.onset_ms + n.duration_ms] for n in ref_notes])
    ax.scatter(ref_xyz[:,0], ref_xyz[:,1], ref_xyz[:,2], c='b', marker='o', label='Reference')
    # Plot all recorded notes
    rec_xyz = np.array([[n.pitch, n.onset_ms, n.onset_ms + n.duration_ms] for n in rec_notes])
    ax.scatter(rec_xyz[:,0], rec_xyz[:,1], rec_xyz[:,2], c='r', marker='^', label='Recorded')
    # Overlay matches if provided
    if matches:
        for idx, (ref, rec) in enumerate(matches):
            if ref is None or rec is None:
                continue
            ref_pt = [ref.pitch, ref.onset_ms, ref.onset_ms + ref.duration_ms]
            rec_pt = [rec.pitch, rec.onset_ms, rec.onset_ms + rec.duration_ms]
            # Draw triangle (line) between matched notes
            ax.plot([ref_pt[0], rec_pt[0]], [ref_pt[1], rec_pt[1]], [ref_pt[2], rec_pt[2]], c='g', alpha=0.7)
    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()

def report_problematic_matches(matches, timing_tol=5, duration_tol=5):
    """
    Print matches responsible for not having a 100% match.
    Shows unmatched notes and matched notes with pitch/timing/duration errors.
    timing_tol and duration_tol are ms tolerances for considering timing/duration as perfect.
    """
    problems = []
    for idx, (ref, rec) in enumerate(matches):
        if ref is None and rec is not None:
            problems.append(f"Unmatched recorded note: pitch={rec.pitch}, onset={rec.onset_ms}, duration={rec.duration_ms}, mark={getattr(rec, 'mark', None)}")
        elif rec is None and ref is not None:
            problems.append(f"Unmatched reference note: pitch={ref.pitch}, onset={ref.onset_ms}, duration={ref.duration_ms}, mark={getattr(ref, 'mark', None)}")
        elif ref is not None and rec is not None:
            pitch_ok = ref.pitch == rec.pitch
            timing_ok = abs(ref.onset_ms - rec.onset_ms) <= timing_tol
            duration_ok = abs(ref.duration_ms - rec.duration_ms) <= duration_tol
            if not (pitch_ok and timing_ok and duration_ok):
                msg = f"Match error: ref(pitch={ref.pitch}, onset={ref.onset_ms}, dur={ref.duration_ms}) <-> rec(pitch={rec.pitch}, onset={rec.onset_ms}, dur={rec.duration_ms})"
                if not pitch_ok:
                    msg += f" | pitch mismatch ({ref.pitch} vs {rec.pitch})"
                if not timing_ok:
                    msg += f" | onset mismatch ({ref.onset_ms} vs {rec.onset_ms})"
                if not duration_ok:
                    msg += f" | duration mismatch ({ref.duration_ms} vs {rec.duration_ms})"
                problems.append(msg)
    if problems:
        print(f"Problematic matches (not 100%): {len(problems)} found.")
        for p in problems:
            print("  -", p)
    else:
        print("All matches are perfect (100% match).")


# Debug visualization: show all notes before matching
DEBUG_VIS = True
if DEBUG_VIS:
    print("Showing 3D notes before matching...")
    plot_notes_and_matches_3d(ref_notes, rec_notes_norm)

# ============================================
# ========== WEIGHT TRAINING PART ============
# ============================================

generations = 40
population_size = 16
initial_mutation_scale = 100
final_mutation_scale = 0.1

def get_mutation_scale(gen, generations):
    return initial_mutation_scale * ((final_mutation_scale / initial_mutation_scale) ** (gen / generations))

# Initialize population
real_values = [0.01, 3.0, 0.5, 15.0]  # onset_w (ms), pitch_w (halftone), dur_w (note end ms), k (null_cost)
population = [real_values[:]]
best_score = -1
best_params = None
results = []

print(f"Training for {generations} generations, population size {population_size}")
print(f"Initial mutation scale: {initial_mutation_scale}")

# === Evolution loop ===
for gen in range(0, generations + 1):
    mutation_scale = get_mutation_scale(gen, generations)
    gen_results = []
    print(f"\nGeneration {gen}/{generations} (mutation scale: {mutation_scale:.2f})")
    for i, (onset_w, pitch_w, dur_w, k) in enumerate(population):
        def weighted_distance(a, b):
            if a is None or b is None:
                return k
            p_diff = abs(a.pitch - b.pitch)
            t1_diff = abs(a.onset_ms - b.onset_ms)
            t2_diff = abs((a.onset_ms+a.duration_ms) - (b.onset_ms+b.duration_ms))
            return pitch_w * p_diff + onset_w * t1_diff + dur_w * t2_diff
        matches = match_lists(ref_notes, rec_notes_norm, weighted_distance, k)
        score, (acc_score, timing_score, dur_score) = evaluate_matches(matches)
        results.append((score, onset_w, pitch_w, dur_w, k))
        gen_results.append((score, [onset_w, pitch_w, dur_w, k], matches))
        if score > best_score:
            best_score = score
            best_params = [onset_w, pitch_w, dur_w, k]
            print(f"  ** New best score! **")
        print("\r"+"="*i+"-"*(len(population)-i-1), end="")
    gen_results.sort(reverse=True, key=lambda x: x[0])
    top = gen_results[:population_size // 2]
    if not top:
        print("[WARNING] No top candidates found, using best_params for mutation.")
        top = [(best_score, best_params[:], None)]
    print("\nTop 3 candidates this generation:")
    for i in range(min(3, len(top))):
        s, p, _ = top[i]
        print(f"  Score: {s:.4f} Params: onset_w={p[0]:.2f}, pitch_w={p[1]:.2f}, dur_w={p[2]:.2f}, k={p[3]:.2f}")

    # Analyze problems for best candidate
    def analyze_problems(matches):
        pitch_err = onset_err = dur_err = 0
        for ref, rec in matches:
            if ref is None or rec is None:
                continue
            if ref.pitch != rec.pitch:
                pitch_err += 1
            if abs(ref.onset_ms - rec.onset_ms) > 5:
                onset_err += 1
            if abs(ref.duration_ms - rec.duration_ms) > 5:
                dur_err += 1
        total = max(1, len(matches))
        return {
            'pitch': pitch_err / total,
            'onset': onset_err / total,
            'duration': dur_err / total
        }

    new_population = []
    new_population.append(best_params[:])
    temperature = initial_mutation_scale * ((final_mutation_scale / initial_mutation_scale) ** (gen / generations))
    # Use problem analysis to guide some mutations
    problem_stats = analyze_problems(top[0][2]) if top[0][2] is not None else {'pitch':0,'onset':0,'duration':0}
    for idx, (score, params, matches) in enumerate(top):
        for j in range(2):
            mutated = params[:]
            # For some candidates, adapt weights based on problem types
            if j == 1:
                # Increase weights for most problematic type
                max_type = max(problem_stats, key=problem_stats.get)
                if max_type == 'pitch':
                    mutated[1] = max(0.01, mutated[1] * 1.2)
                elif max_type == 'onset':
                    mutated[0] = max(0.0001, mutated[0] * 1.2)
                elif max_type == 'duration':
                    mutated[2] = max(0.0001, mutated[2] * 1.2)
            # Add random noise and enforce minimums
            mutated = [
                max(0.0001, abs(mutated[0] + random.gauss(0, temperature * 0.5))),  # onset_w
                max(0.01, abs(mutated[1] + random.gauss(0, temperature * 0.05))),   # pitch_w
                max(0.0001, abs(mutated[2] + random.gauss(0, temperature * 0.5))),  # dur_w
                max(0.1, abs(mutated[3] + random.gauss(0, temperature * 0.1)))      # k
            ]
            new_population.append(mutated)
    # Add random candidates for diversity
    while len(new_population) < population_size:
        if random.random() < 0.2:  # 20% chance for a random candidate
            new_population.append([
                max(0.0001, random.uniform(0.0001, 0.1)),   # onset_w
                max(0.01, random.uniform(0.01, 5.0)),        # pitch_w
                max(0.0001, random.uniform(0.0001, 0.1)),    # dur_w
                max(0.1, random.uniform(1.0, 50.0))          # k
            ])
        else:
            # Mutate a random parent
            parent = random.choice(top)[1]
            mutated = [
                max(0.0001, abs(parent[0] + random.gauss(0, temperature * 0.5))),
                max(0.01, abs(parent[1] + random.gauss(0, temperature * 0.05))),
                max(0.0001, abs(parent[2] + random.gauss(0, temperature * 0.5))),
                max(0.1, abs(parent[3] + random.gauss(0, temperature * 0.1)))
            ]
            new_population.append(mutated)
    population = new_population[:population_size]
    print(f"Generation {gen}/{generations} complete. Best score: {best_score:.4f}")
    if best_params is not None:
        def best_weighted_distance(a, b):
            if a is None or b is None:
                return best_params[3]
            p_diff = abs(a.pitch - b.pitch)
            t1_diff = abs(a.onset_ms - b.onset_ms)
            t2_diff = abs((a.onset_ms+a.duration_ms) - (b.onset_ms+b.duration_ms))
            return best_params[1] * p_diff + best_params[0] * t1_diff + best_params[2] * t2_diff
        best_matches = match_lists(ref_notes, rec_notes_norm, best_weighted_distance, best_params[3])
        report_problematic_matches(best_matches)

print(f"\n=== Final Results ===")
print(f"Best score: {best_score}")
print(f"Best params: onset_w={best_params[0]}, pitch_w={best_params[1]}, dur_w={best_params[2]}, k={best_params[3]}")

# Show final 3D match visualization for best candidate
if best_params is not None:
    print("Showing 3D match visualization for best candidate after all generations.")
    def best_weighted_distance(a, b):
        if a is None or b is None:
            return best_params[3]
        p_diff = abs(a.pitch - b.pitch)
        t1_diff = abs(a.onset_ms - b.onset_ms)
        t2_diff = abs((a.onset_ms+a.duration_ms) - (b.onset_ms+b.duration_ms))
        return best_params[1] * p_diff + best_params[0] * t1_diff + best_params[2] * t2_diff
    best_matches = match_lists(ref_notes, rec_notes_norm, best_weighted_distance, best_params[3])
    plot_notes_and_matches_3d(ref_notes, rec_notes_norm, best_matches)