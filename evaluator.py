import difflib
from collections import defaultdict
from pprint import pprint
from typing import Optional
import numpy as np
from mido import MidiTrack
from scipy.optimize import linear_sum_assignment

from mt_types import (
    Note, PedalEvent, NoteEvaluation, Issue, HandIssueSummary,
    PerformanceEvaluation, articulation_type
)
from midi_utils import extract_notes_and_pedal

weights = {
    "accuracy": 0.42,
    "timing": 0.20,
    "dynamics": 0.03,
    "articulation": 0.15,
    "pedal": 0.10,
    "tempo": 0.10,
}
assert sum(weights.values()) == 1.0, "Weights must sum to 1.0"

def _match_notes(ref_notes: list[Note], rec_notes: list[Note]) -> tuple[list[tuple[Note, Optional[Note]]], list[Note], float]:
    """
    Match reference notes to recorded notes and return matched pairs, unmatched recorded notes, and a temporal scale factor.
    
    Returns:
        matches (list[tuple[Note, Optional[Note]]]): Tuples pairing each reference Note with the matched recorded Note, or `None` when a reference note was not played.
        extras (list[Note]): Recorded notes that were not matched to any reference note.
        scale (float): Temporal scale factor applied to recorded onsets to align with the reference (reference span divided by recording span; 1.0 when alignment is undefined, 0.0 when there are no recordings).
    """
    if not ref_notes:
        return [], rec_notes.copy(), 1.0
    if not rec_notes:
        return [(r, None) for r in ref_notes], [], 0.0

    ref_onsets = np.array([n.onset_ms for n in ref_notes])
    rec_onsets = np.array([n.onset_ms for n in rec_notes])
    scale = 1.0
    if len(ref_onsets) == 0 or len(rec_onsets) == 0:
        rec_notes_norm = rec_notes
    else:
        ref_min, ref_max = int(np.min(ref_onsets)), int(np.max(ref_onsets))
        rec_min, rec_max = int(np.min(rec_onsets)), int(np.max(rec_onsets))
        ref_span, rec_span = ref_max - ref_min, rec_max - rec_min
        scale = (ref_span / rec_span) if rec_span > 0 else 1.0
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

    onset_w, pitch_w, dur_w, k = 0.001, 3.0, 0.05, 15.0
    def weighted_distance(a, b):
        """
        Compute a weighted distance between two notes based on pitch, onset, and offset times.
        
        Parameters:
            a: The first note object or None. Expected to have `pitch`, `onset_ms`, and `duration_ms` attributes.
            b: The second note object or None. Expected to have `pitch`, `onset_ms`, and `duration_ms` attributes.
        
        Returns:
            A numeric score representing dissimilarity; a larger value means more different. If either `a` or `b` is `None`, returns a large constant indicating no valid match.
        """
        if a is None or b is None:
            return k
        return (
            pitch_w * abs(a.pitch - b.pitch)
            + onset_w * abs(a.onset_ms - b.onset_ms)
            + dur_w * abs((a.onset_ms + a.duration_ms) - (b.onset_ms + b.duration_ms))
        )

    m, n = len(ref_notes), len(rec_notes_norm)
    size = max(m, n)
    cost_matrix = np.full((size, size), k, dtype=float)
    for i in range(m):
        for j in range(n):
            cost_matrix[i, j] = weighted_distance(ref_notes[i], rec_notes_norm[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    extras = []
    for r, c in zip(row_ind, col_ind):
        if r < m and c < n and cost_matrix[r, c] < k:
            matches.append((ref_notes[r], rec_notes[c]))
        elif r < m:
            matches.append((ref_notes[r], None))
        elif c < n:
            extras.append(rec_notes[c])
    return matches, extras, scale


def _detect_articulation(duration: float, reference_duration: int) -> articulation_type:
    """
    Classifies a note's articulation compared to a reference duration.
    
    Parameters:
        duration (float): Measured duration of the played note (same time units as reference_duration).
        reference_duration (int): Expected/reference duration for the note.
    
    Returns:
        articulation_type: `'staccato'` if duration is less than 50% of reference, `'legato'` if duration is greater than 120% of reference, `'normal'` otherwise.
    """
    ratio = duration / (reference_duration + 1e-6)
    if ratio < 0.5:
        return "staccato"
    elif ratio > 1.2:
        return "legato"
    return "normal"

def _evaluate_pedal_use(ref_pedals: list[PedalEvent], rec_pedals: list[PedalEvent], evaluation: PerformanceEvaluation):
    """
    Compare reference and recorded pedal event streams, append pedal-related Issue entries to the evaluation, and set evaluation.pedal_score.
    
    Groups events by pedal type, quantizes times to 50 ms buckets, and aligns reference and recorded sequences using a sequence matcher. For mismatches, missing events, and extra events the function appends Issue objects with a time_ms, computed severity (based on timing and value differences or fixed severities), and a descriptive message. Finally, sets evaluation.pedal_score to 1.0 - (issues / total) clamped at a minimum of 0.0; if there are no pedal events, sets pedal_score to 1.0.
    
    Parameters:
        ref_pedals (list[PedalEvent]): Reference pedal events to compare.
        rec_pedals (list[PedalEvent]): Recorded pedal events to compare.
        evaluation (PerformanceEvaluation): Mutable evaluation object to which pedal issues are appended and whose pedal_score is set.
    """
    ref_by_type = defaultdict(list)
    rec_by_type = defaultdict(list)
    for p in ref_pedals:
        ref_by_type[p.pedal_type].append(p)
    for p in rec_pedals:
        rec_by_type[p.pedal_type].append(p)

    total = 0
    issues = 0
    correct = 0

    for pt in set(ref_by_type.keys()).union(rec_by_type.keys()):
        ref_events = ref_by_type.get(pt, [])
        rec_events = rec_by_type.get(pt, [])

        ref_seq = [(round(e.time_ms / 50), e.value) for e in ref_events]
        rec_seq = [(round(e.time_ms / 50), e.value) for e in rec_events]

        matcher = difflib.SequenceMatcher(a=ref_seq, b=rec_seq, autojunk=False)
        opcodes = matcher.get_opcodes()
        total += max(len(ref_seq), len(rec_seq))

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                for k in range(i2 - i1):
                    correct += 1
            elif tag == "replace":
                for k in range(max(i2 - i1, j2 - j1)):
                    ref_idx = i1 + k if i1 + k < i2 else None
                    rec_idx = j1 + k if j1 + k < j2 else None
                    ref_event = ref_events[ref_idx] if ref_idx is not None else None
                    rec_event = rec_events[rec_idx] if rec_idx is not None else None
                    if ref_event and rec_event:
                        time_diff = abs(rec_event.time_ms - ref_event.time_ms)
                        value_diff = abs(rec_event.value - ref_event.value)
                        evaluation.issues.append(Issue(
                            time_ms=rec_event.time_ms,
                            severity=min((time_diff / 300 + value_diff / 127), 1.0),
                            category="pedal",
                            description=f"{pt} pedal mismatch: time {time_diff} ms, value {value_diff}"
                        ))
                    elif ref_event:
                        evaluation.issues.append(Issue(
                            time_ms=ref_event.time_ms,
                            severity=1.0,
                            category="pedal",
                            description=f"Missing {pt} pedal event"
                        ))
                    elif rec_event:
                        evaluation.issues.append(Issue(
                            time_ms=rec_event.time_ms,
                            severity=0.7,
                            category="pedal",
                            description=f"Extra {pt} pedal event"
                        ))
                    issues += 1
            elif tag == "delete":
                for k in range(i1, i2):
                    ref_event = ref_events[k]
                    evaluation.issues.append(Issue(
                        time_ms=ref_event.time_ms,
                        severity=1.0,
                        category="pedal",
                        description=f"Missing {pt} pedal event"
                    ))
                    issues += 1
            elif tag == "insert":
                for k in range(j1, j2):
                    rec_event = rec_events[k]
                    evaluation.issues.append(Issue(
                        time_ms=rec_event.time_ms,
                        severity=0.7,
                        category="pedal",
                        description=f"Extra {pt} pedal event"
                    ))
                    issues += 1

    if total > 0:
        evaluation.pedal_score = max(0.0, 1.0 - issues / total)
    else:
        evaluation.pedal_score = 1.0

def _pitch_to_name(pitch: int) -> str:
    """
    Convert a MIDI pitch number to a musical pitch name with octave.
    
    Parameters:
        pitch (int): MIDI note number (typically 0–127); 60 corresponds to C4.
    
    Returns:
        str: Pitch name with octave (e.g., "C4", "F#3").
    """
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (pitch // 12) - 1
    name = names[pitch % 12]
    return f"{name}{octave}"


def _generate_tips(ev: PerformanceEvaluation) -> tuple[list[str], str]:
    """
    Generate prioritized coaching tips and an encouragement message from a performance evaluation.
    
    Parameters:
        ev (PerformanceEvaluation): Evaluation containing per-note metrics, aggregate scores, detected issues, and hand summaries.
    
    Returns:
        tuple[list[str], str]: A pair where the first element is a list of user-facing tip strings sorted by priority (highest first), and the second element is a short encouragement message based on the overall score.
    """
    tips = []

    if ev.tempo_deviation_ratio < 0.95:
        percent = (1.0 - ev.tempo_deviation_ratio) * 100
        tips.append((f"You are playing too slow ({percent:.1f}% slower than reference). Try increasing your tempo a bit!", (1.0 - ev.tempo_deviation_ratio) * weights.get("tempo", 1.0) * 10))
    elif ev.tempo_deviation_ratio > 1.05:
        percent = (ev.tempo_deviation_ratio - 1.0) * 100
        tips.append((f"You are playing too fast ({percent:.1f}% faster than reference). Slow down slightly to match the reference.", (ev.tempo_deviation_ratio - 1.0) * weights.get("tempo", 1.0) * 10))
    elif abs(ev.tempo_deviation_ratio - 1.0) > 0.01:
        percent = (ev.tempo_deviation_ratio - 1.0) * 100
        tips.append((f"Adjust your tempo slightly to match the reference (off by {percent:.1f}%).", abs(ev.tempo_deviation_ratio - 1.0) * weights.get("tempo", 1.0) * 10))

    if ev.accuracy_score < 0.85:
        wrong_notes = [i for i in ev.issues if i.category == "accuracy"]
        locations = [f"index {idx}" for idx, i in enumerate(wrong_notes[:3])]
        if locations:
            loc_str = " at " + ", ".join(locations)
        else:
            loc_str = ""
        msg = f"You missed quite a few notes{loc_str} ({ev.wrong_notes + ev.missing_notes} total). Focus on accuracy first."
        tips.append((msg, (1.0 - ev.accuracy_score) * weights.get("accuracy", 1.0)))
    elif ev.accuracy_score < 1.0:
        tips.append(("To reach perfection, double-check each note for accuracy.", 0.5 * (1.0 - ev.accuracy_score) * weights.get("accuracy", 1.0)))

    if ev.extra_notes > 0:
        extra_notes = [i for i in ev.issues if i.category == "accuracy" and i.severity < 1.0]
        locations = [f"index {idx}" for idx, i in enumerate(extra_notes[:3])]
        if locations:
            loc_str = " at " + ", ".join(locations)
        else:
            loc_str = ""
        msg = f"You are adding {ev.extra_notes} unnecessary notes{loc_str}. Be careful not to press extra keys."
        tips.append((msg, 0.5 * weights.get("accuracy", 1.0)))

    if ev.missing_notes > 0:
        missing_notes = [i for i in ev.issues if i.category == "accuracy" and i.severity == 1.0]
        locations = [f"index {idx}" for idx, i in enumerate(missing_notes[:3])]
        if locations:
            loc_str = " at " + ", ".join(locations)
        else:
            loc_str = ""
        msg = f"You are missing {ev.missing_notes} notes{loc_str}. Be careful not to miss any keys."
        tips.append((msg, 0.5 * weights.get("accuracy", 1.0)))

    if ev.timing_score < 0.8:
        timing_issues = [i for i in ev.issues if i.category == "timing"]
        locations = [f"index {idx}" for idx, i in enumerate(timing_issues[:3])]
        if locations:
            loc_str = " at " + ", ".join(locations)
        else:
            loc_str = ""
        msg = f"Your timing is off{loc_str} (average deviation {ev.avg_timing_deviation_ms/1000:.2f} s). Practice with a metronome."
        tips.append((msg, (1.0 - ev.timing_score) * weights.get("timing", 1.0)))
    elif ev.rhythmic_stability > 50:
        tips.append((f"Your rhythm fluctuates (stability {ev.rhythmic_stability:.1f}). Try keeping a steadier beat.", min(ev.rhythmic_stability / 200, 1.0) * weights.get("timing", 1.0)))
    elif ev.timing_score < 1.0:
        tips.append(("To reach perfection, refine your microtiming for each note.", 0.5 * (1.0 - ev.timing_score) * weights.get("timing", 1.0)))

    if ev.dynamics_score < 0.85:
        dynamic_issues = [n for n in ev.notes if abs(n.velocity_deviation) > 10]
        locations = [f"index {idx}" for idx, n in enumerate(dynamic_issues[:3])]
        if locations:
            loc_str = " at " + ", ".join(locations)
        else:
            loc_str = ""
        msg = f"Your dynamics are uneven{loc_str}. Try to control volume more consistently."
        tips.append((msg, (1.0 - ev.dynamics_score) * weights.get("dynamics", 1.0)))
    elif ev.dynamics_score < 1.0:
        tips.append(("To reach perfection, make your dynamics perfectly balanced.", 0.5 * (1.0 - ev.dynamics_score) * weights.get("dynamics", 1.0)))

    num_staccato = sum(1 for n in ev.notes if n.articulation == "staccato")
    num_legato = sum(1 for n in ev.notes if n.articulation == "legato")
    articulation_issues = [n for n in ev.notes if n.articulation and n.articulation != "normal"]

    if num_staccato > len(ev.notes) * 0.3:
        tips.append(("You are playing too staccato. Hold the notes longer for smoother phrasing.",
                     (num_staccato / len(ev.notes)) * weights.get("articulation", 1.0)))
    elif num_legato > len(ev.notes) * 0.3:
        tips.append(("You are playing too legato. Try separating the notes a bit more.",
                     (num_legato / len(ev.notes)) * weights.get("articulation", 1.0)))
    elif num_staccato + num_legato > 0:
        tips.append(("To reach perfection, refine your articulation to match the reference.", 0.5 * ((num_staccato + num_legato) / len(ev.notes)) * weights.get("articulation", 1.0)))

    if articulation_issues:
        locations = [f"index {idx}" for idx, n in enumerate(articulation_issues[:3])]
        if locations:
            loc_str = " at " + ", ".join(locations)
        else:
            loc_str = ""
        msg = f"Refine articulation to match the reference{loc_str}."
        tips.append((msg, (sum(1 for _ in articulation_issues)/len(ev.notes)) * weights.get("articulation", 1.0)))

    if ev.pedal_score < 0.8:
        pedal_issues = [i for i in ev.issues if i.category == "pedal"]
        locations = [f"index {idx}" for idx, i in enumerate(pedal_issues[:3])]
        if locations:
            loc_str = " at " + ", ".join(locations)
        else:
            loc_str = ""
        msg = f"Your pedal usage needs improvement{loc_str}. Listen carefully to the pedal changes in the reference."
        tips.append((msg, (1.0 - ev.pedal_score) * weights.get("pedal", 1.0)))
    elif ev.pedal_score < 1.0:
        tips.append(("To reach perfection, perfect your pedal timing and depth.", 0.5 * (1.0 - ev.pedal_score) * weights.get("pedal", 1.0)))

    rh_issues = ev.hand_summary.get("rh", HandIssueSummary())
    lh_issues = ev.hand_summary.get("lh", HandIssueSummary())
    if rh_issues.total_issues > 0 and lh_issues.total_issues > 0:
        if rh_issues.total_issues > lh_issues.total_issues * 1.5:
            tips.append(("Your right hand seems to have more mistakes. Focus on right-hand passages.", 0))
        elif lh_issues.total_issues > rh_issues.total_issues * 1.5:
            tips.append(("Your left hand seems to struggle more. Slow down left-hand parts for clarity.", 0))

    if ev.overall_score == 1.0:
        encouragement = "Perfect performance! You don't need this anymore. Just practice on your own and bring in your emotions."
    elif ev.overall_score > 0.97:
        encouragement = "Outstanding performance! Don't play too robotically — bring in your emotions. "
    elif ev.overall_score > 0.9:
        encouragement = "Excellent performance! "
    elif ev.overall_score > 0.75:
        encouragement = "Good work! "
    else:
        encouragement = "Keep practicing! "

    tips_sorted = [t for t, _ in sorted(tips, key=lambda x: x[1], reverse=True)]
    return tips_sorted, encouragement


class Evaluator:
    def __init__(self, recording: MidiTrack, reference: tuple[list[Note], list[PedalEvent]]):
        """
        Initialize the Evaluator with a recorded MIDI track and its reference data.
        
        Parameters:
            recording (MidiTrack): The recorded MIDI track to be evaluated.
            reference (tuple[list[Note], list[PedalEvent]]): A tuple containing the reference note list (score notes in order) and the reference pedal event list.
        
        Notes:
            Caches an evaluation result in `_evaluation`, initialized to `None` for lazy computation.
        """
        self.recording: MidiTrack = recording
        self.reference: tuple[list[Note], list[PedalEvent]] = reference
        self._evaluation: Optional[PerformanceEvaluation] = None

    @property
    def full_evaluation(self) -> PerformanceEvaluation:
        """
        Provide the complete PerformanceEvaluation for the recording, computing and caching it on first access.
        
        Returns:
            PerformanceEvaluation: The computed evaluation containing per-note metrics, aggregated scores, issues, pedal analysis, and hand summaries.
        """
        if not self._evaluation:
            self._evaluate()
        return self._evaluation

    @property
    def score(self) -> float:
        """
        Overall weighted performance score for the evaluated recording.
        
        Returns:
            overall_score (float): Aggregate score between 0.0 and 1.0 representing the performance quality computed from accuracy, timing, dynamics, articulation, pedal use, and tempo.
        """
        return self.full_evaluation.overall_score

    @property
    def tip(self) -> str:
        """
        Return a short coaching message combining an encouragement and the highest-priority tip for the current evaluation.
        
        If the evaluation has no cached comments, this method generates and stores tip messages (and an encouragement) before returning the result. The returned string is the encouragement followed by the first tip; if no tips are available, returns "Perfect performance!".
        Returns:
        	A single string containing the encouragement and the top tip, or "Perfect performance!" if no tips exist.
        """
        evaluation = self.full_evaluation
        encouragement = ""
        if not evaluation.comments:
            evaluation.comments, encouragement = _generate_tips(evaluation)
        return encouragement + str(evaluation.comments[0]) if evaluation.comments else encouragement.strip() or "Perfect performance!"

    def _evaluate(self):
        """
        Run a full performance evaluation of the current recording against the reference and store the resulting PerformanceEvaluation on the instance.
        
        Performs note-to-note alignment between reference and recorded notes, compares pitch, timing, duration, velocity, and articulation for each matched note, detects missing and extra notes, evaluates pedal usage, and aggregates per-note metrics into summary scores (accuracy, timing, dynamics, articulation, pedal, tempo) and hand-specific issue summaries. The completed PerformanceEvaluation is assigned to self._evaluation and also printed for inspection.
        """
        rec_notes, rec_pedals = extract_notes_and_pedal(self.recording)
        ref_all, ref_pedals = self.reference

        matches, extras, tempo_ratio = _match_notes(ref_all, rec_notes)
        evaluation = PerformanceEvaluation(total_notes=len(ref_all))
        evaluation.tempo_deviation_ratio = tempo_ratio

        tempo_dev = abs(1.0 - tempo_ratio)
        evaluation.tempo_accuracy_score = max(0.0, 1.0 - tempo_dev)

        hand_stats = {"rh": HandIssueSummary(), "lh": HandIssueSummary(), "unknown": HandIssueSummary()}

        for r, played in matches:
            hand = r.mark or "unknown"

            if played is None:
                evaluation.missing_notes += 1
                evaluation.notes.append(NoteEvaluation(
                    pitch_correct=False,
                    time_ms=r.onset_ms,
                    comments="Note missing"
                ))
                issue = Issue(
                    time_ms=r.onset_ms, severity=1.0,
                    category="accuracy", description=f"Missing note {_pitch_to_name(r.pitch)}"
                )
                evaluation.issues.append(issue)
                hand_stats[hand].total_issues += 1
                hand_stats[hand].by_category["accuracy"] += 1
                continue

            pitch_correct = r.pitch == played.pitch
            pitch_error = None if pitch_correct else (played.pitch - r.pitch)
            onset_dev = played.onset_ms - r.onset_ms
            dur_dev = (played.duration_ms / (tempo_ratio + 1e-6)) - r.duration_ms
            vel_dev = played.velocity - r.velocity
            articulation = _detect_articulation(played.duration_ms / tempo_ratio, r.duration_ms)

            note_eval = NoteEvaluation(
                pitch_correct=pitch_correct,
                pitch_error=pitch_error,
                onset_deviation_ms=onset_dev,
                duration_deviation_ms=dur_dev,
                velocity_deviation=vel_dev,
                articulation=articulation,
                time_ms=played.onset_ms,
            )
            evaluation.notes.append(note_eval)

            if not pitch_correct:
                issue = Issue(
                    time_ms=played.onset_ms, note=played.pitch,
                    severity=min(abs(pitch_error)/12, 1.0),
                    category="accuracy", description=f"Wrong pitch {_pitch_to_name(played.pitch)} vs {_pitch_to_name(r.pitch)}"
                )
                evaluation.issues.append(issue)
                evaluation.wrong_notes += 1
                hand_stats[hand].total_issues += 1
                hand_stats[hand].by_category["accuracy"] += 1
            else:
                evaluation.correct_notes += 1

            if abs(onset_dev) > 30:
                issue = Issue(
                    time_ms=played.onset_ms, note=played.pitch,
                    severity=min(abs(onset_dev)/200, 1.0),
                    category="timing", description=f"Timing off by {onset_dev:.1f} ms"
                )
                evaluation.issues.append(issue)
                hand_stats[hand].total_issues += 1
                hand_stats[hand].by_category["timing"] += 1

            if articulation != "normal":
                issue = Issue(
                    time_ms=played.onset_ms, note=played.pitch,
                    severity=0.5, category="articulation",
                    description=f"Played {articulation}"
                )
                evaluation.issues.append(issue)
                hand_stats[hand].total_issues += 1
                hand_stats[hand].by_category["articulation"] += 1

        evaluation.extra_notes = len(extras)
        for n in extras:
            evaluation.issues.append(Issue(
                time_ms=n.onset_ms, note=n.pitch,
                severity=0.5, category="accuracy",
                description=f"Extra note {n.pitch}"
            ))

        if evaluation.notes:
            evaluation.avg_timing_deviation_ms = float(np.mean([abs(n.onset_deviation_ms) for n in evaluation.notes]))
            evaluation.avg_duration_deviation_ms = float(np.mean([abs(n.duration_deviation_ms) for n in evaluation.notes]))
            evaluation.avg_velocity_deviation = float(np.mean([abs(n.velocity_deviation) for n in evaluation.notes]))

        deviations = [n.onset_deviation_ms for n in evaluation.notes]
        evaluation.rhythmic_stability = float(np.std(deviations)) if len(deviations) > 1 else 0.0
        evaluation.tempo_consistency = 1.0 / (1.0 + evaluation.rhythmic_stability)

        _evaluate_pedal_use(ref_pedals, rec_pedals, evaluation)

        evaluation.accuracy_score = evaluation.correct_notes / max(1, evaluation.total_notes)
        evaluation.timing_score = max(0.0, 1.0 - evaluation.avg_timing_deviation_ms / 200.0)
        evaluation.dynamics_score = max(0.0, 1.0 - evaluation.avg_velocity_deviation / 40.0)
        evaluation.articulation_score = 1.0 - sum(1 for i in evaluation.issues if i.category == "articulation") / max(1, evaluation.total_notes)

        evaluation.overall_score = (
            weights["accuracy"] * evaluation.accuracy_score +
            weights["timing"] * evaluation.timing_score +
            weights["dynamics"] * evaluation.dynamics_score +
            weights["articulation"] * evaluation.articulation_score +
            weights["pedal"] * evaluation.pedal_score +
            weights["tempo"] * evaluation.tempo_accuracy_score
        )

        for hand, summary in hand_stats.items():
            if summary.total_issues:
                severities = [i.severity for i in evaluation.issues if (hand == "unknown" or any(
                    (r.mark == hand and (r.pitch == i.note or abs(r.onset_ms - i.time_ms) < 50))
                    for r, _ in matches))]
                summary.avg_severity = float(np.mean(severities)) if severities else 0.0

        evaluation.hand_summary = hand_stats

        self._evaluation = evaluation
        pprint(evaluation)