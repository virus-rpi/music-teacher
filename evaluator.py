from dataclasses import dataclass
import mido

from midi_teach import MidiTeacher


@dataclass
class Score:
    accuracy: float
    relative_timing: float
    absolute_timing: float
    overall: float

def evaluate(section_task, recording: mido.MidiTrack, teacher: MidiTeacher) -> Score:
    events = []
    abs_ms = 0
    for msg in recording:
        abs_ms += getattr(msg, 'time', 0)
        if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0:
            events.append((abs_ms, msg.note))
    threshold_ms = 50
    recorded_onsets = []
    for t, note in events:
        if not recorded_onsets or t - recorded_onsets[-1][0] > threshold_ms:
            recorded_onsets.append([t, {note}])
        else:
            recorded_onsets[-1][1].add(note)
    recorded_onsets = [(o[0], o[1]) for o in recorded_onsets]

    if not section_task.section.chords:
        return Score(1.0, 1.0, 1.0, 1.0)

    chord_indices = list(range(section_task.start_idx, section_task.end_idx + 1))
    chord_ticks = []
    for idx in chord_indices:
        try:
            chord_ticks.append(teacher.chord_times[idx])
        except Exception:
            chord_ticks.append(None)

    def tick_to_seconds(tick):
        if tick is None:
            return 0.0
        mid = mido.MidiFile(teacher.midi_path)
        ticks_per_beat = mid.ticks_per_beat
        tempo_changes = []
        for track in mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == 'set_tempo':
                    tempo_changes.append((abs_tick, msg.tempo))
        tempo_changes.sort()
        cur_tempo = 500000
        last_tick = 0
        seconds = 0.0
        for tc_tick, tc_tempo in tempo_changes:
            if tc_tick >= tick:
                break
            delta = tc_tick - last_tick
            seconds += (delta * cur_tempo) / (ticks_per_beat * 1e6)
            cur_tempo = tc_tempo
            last_tick = tc_tick
        delta = tick - last_tick
        seconds += (delta * cur_tempo) / (ticks_per_beat * 1e6)
        return seconds

    ref_abs_secs = [tick_to_seconds(t) for t in chord_ticks]
    if any(v is None for v in chord_ticks):
        measure_start_tick = teacher.chord_times[section_task.start_idx] if section_task.start_idx < len(teacher.chord_times) else 0
        ref_abs_secs = [tick_to_seconds(measure_start_tick + t) for t in section_task.section.times]

    ref_start = ref_abs_secs[0]
    ref_onsets_ms = [int((s - ref_start) * 1000.0) for s in ref_abs_secs]
    ref_chords = [set(n for n, _hand in chord) for chord in section_task.section.chords]

    if not recorded_onsets:
        return Score(0.0, 0.0, 0.0, 0.0)

    n_ref = len(ref_onsets_ms)
    n_rec = len(recorded_onsets)
    m = min(n_ref, n_rec)

    acc_scores = []
    for i in range(m):
        ref_set = ref_chords[i]
        rec_set = recorded_onsets[i][1]
        if not ref_set and not rec_set:
            acc_scores.append(1.0)
            continue
        inter = len(ref_set & rec_set)
        denom = (len(ref_set) + len(rec_set))
        if denom == 0:
            acc_scores.append(1.0)
        else:
            acc_scores.append((2.0 * inter) / denom)
    if n_rec < n_ref:
        acc_scores.extend([0.0] * (n_ref - n_rec))
    accuracy_score = sum(acc_scores) / max(1, n_ref)

    def intervals(xs):
        return [xs[i + 1] - xs[i] for i in range(len(xs) - 1)] if len(xs) > 1 else []

    ref_intervals = intervals(ref_onsets_ms[:m])
    rec_intervals = intervals([t for t, _ in recorded_onsets[:m]])
    if not ref_intervals or not rec_intervals:
        relative_score = 1.0
    else:
        mean_ref = sum(ref_intervals) / len(ref_intervals)
        mean_rec = sum(rec_intervals) / len(rec_intervals)
        if mean_ref == 0 or mean_rec == 0:
            relative_score = 0.0
        else:
            nref = [ri / mean_ref for ri in ref_intervals]
            nrec = [ri / mean_rec for ri in rec_intervals]
            L = min(len(nref), len(nrec))
            diffs = [abs(nref[i] - nrec[i]) for i in range(L)]
            avg_diff = sum(diffs) / L
            relative_score = max(0.0, 1.0 - avg_diff)

    ref_total = (ref_onsets_ms[-1] - ref_onsets_ms[0]) if len(ref_onsets_ms) > 1 else 0
    rec_total = (recorded_onsets[m - 1][0] - recorded_onsets[0][0]) if m > 1 else 0
    if ref_total == 0:
        absolute_score = 1.0 if rec_total == 0 else 0.0
    else:
        rel_error = abs(rec_total - ref_total) / float(ref_total)
        absolute_score = max(0.0, 1.0 - min(rel_error, 1.0))

    score = (0.7 * accuracy_score) + (0.2 * relative_score) + (0.1 * absolute_score)
    score = max(0.0, min(1.0, score))

    return Score(accuracy_score, relative_score, absolute_score, score)