import difflib
from collections import defaultdict
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional, Literal
import numpy as np
from mido import MidiTrack
from scipy.optimize import linear_sum_assignment

articulation_type = Literal["staccato", "legato", "normal"]
issue_category = Literal["accuracy", "timing", "dynamics", "pedal", "articulation"]
pedal_type = Literal["sustain", "sostenuto", "soft"]

@dataclass
class NoteEvaluation:
    pitch_correct: bool
    pitch_error: Optional[int] = None
    onset_deviation_ms: float = 0.0
    duration_deviation_ms: float = 0.0
    velocity_deviation: float = 0.0
    articulation: Optional[articulation_type] = None
    pedal_used: Optional[bool] = None
    time_ms: Optional[int] = None
    comments: Optional[str] = None


@dataclass
class Issue:
    time_ms: int
    note: Optional[int] = None
    severity: float = 0.0    # 0–1 scaled severity
    category: issue_category = ""
    description: str = ""

@dataclass
class HandIssueSummary:
    total_issues: int = 0
    by_category: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_severity: float = 0.0


@dataclass
class PerformanceEvaluation:
    notes: list[NoteEvaluation] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    total_notes: int = 0
    correct_notes: int = 0
    wrong_notes: int = 0
    missing_notes: int = 0
    extra_notes: int = 0

    avg_timing_deviation_ms: float = 0.0
    avg_duration_deviation_ms: float = 0.0
    avg_velocity_deviation: float = 0.0

    rhythmic_stability: float = 0.0
    tempo_consistency: float = 0.0
    dynamic_balance: float = 0.0

    phrasing_similarity: float = 0.0
    hand_independence: float = 0.0
    chord_accuracy: float = 0.0

    tempo_deviation_ratio: float = 1.0
    tempo_accuracy_score: float = 1.0

    hand_summary: dict[str, HandIssueSummary] = field(default_factory=dict)

    overall_score: float = 0.0
    accuracy_score: float = 0.0
    timing_score: float = 0.0
    dynamics_score: float = 0.0
    articulation_score: float = 0.0
    pedal_score: float = 0.0
    comments: Optional[list[str]] = None

@dataclass(frozen=True)
class Note:
    pitch: int
    onset_ms: int
    duration_ms: int
    velocity: int
    mark: Optional[str] = None

@dataclass
class PedalEvent:
    time_ms: int
    value: int
    pedal_type: pedal_type


def _extract_notes_and_pedal(track: MidiTrack, mark: Optional[str]=None) -> tuple[list[Note], list[PedalEvent]]:
    """
    Extract notes and pedal events (CC64).
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

    return notes, pedals


def _match_notes(ref_notes: list[Note], rec_notes: list[Note]) -> tuple[list[tuple[Note, Optional[Note]]], list[Note], float]:
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
    Classify note articulation relative to reference.
    """
    ratio = duration / (reference_duration + 1e-6)
    if ratio < 0.5:
        return "staccato"
    elif ratio > 1.2:
        return "legato"
    return "normal"

def _evaluate_pedal_use(ref_pedals: list[PedalEvent], rec_pedals: list[PedalEvent], evaluation: PerformanceEvaluation):
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
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (pitch // 12) - 1
    name = names[pitch % 12]
    return f"{name}{octave}"


def _generate_tips(ev: PerformanceEvaluation) -> list[str]:
    tips = []

    if ev.tempo_deviation_ratio < 0.9:
        tips.append(("You are playing too slow. Try increasing your tempo a bit!", 1.0 - ev.tempo_deviation_ratio))
    elif ev.tempo_deviation_ratio > 1.1:
        tips.append(("You are playing too fast. Slow down slightly to match the reference.",
                     ev.tempo_deviation_ratio - 1.0))
    elif abs(ev.tempo_deviation_ratio - 1.0) > 0.01:
        tips.append(("To reach perfection, adjust your tempo slightly to match the reference.", ev.tempo_deviation_ratio - 1.0))

    if ev.accuracy_score < 0.85:
        tips.append((
            f"You missed quite a few notes ({ev.wrong_notes + ev.missing_notes} total). Focus on accuracy first.",
            1.0 - ev.accuracy_score
        ))
    elif ev.accuracy_score < 1.0:
        tips.append(("To reach perfection, double-check each note for accuracy.",  1.0 - ev.accuracy_score))

    if ev.extra_notes > 0:
        tips.append((
            f"You are adding {ev.extra_notes} unnecessary notes. Be careful not to press extra keys.",
            0.5
        ))

    if ev.missing_notes > 0:
        tips.append((
            f"You are missing {ev.missing_notes} notes. Be careful not to miss any keys.",
            0.5
        ))

    if ev.timing_score < 0.8:
        tips.append((
            f"Your timing is off (average deviation {ev.avg_timing_deviation_ms:.1f} ms). Practice with a metronome.",
            1.0 - ev.timing_score
        ))
    elif ev.rhythmic_stability > 50:
        tips.append((
            f"Your rhythm fluctuates (stability {ev.rhythmic_stability:.1f}). Try keeping a steadier beat.",
            min(ev.rhythmic_stability / 200, 1.0)
        ))
    elif ev.timing_score < 1.0:
        tips.append(("To reach perfection, refine your microtiming for each note.", 1.0 - ev.timing_score))

    if ev.dynamics_score < 0.85:
        tips.append((
            "Your dynamics are uneven. Try to control volume more consistently.",
            1.0 - ev.dynamics_score
        ))
    elif ev.dynamics_score < 1.0:
        tips.append(("To reach perfection, make your dynamics perfectly balanced.", 1.0 - ev.dynamics_score))

    num_staccato = sum(1 for n in ev.notes if n.articulation == "staccato")
    num_legato = sum(1 for n in ev.notes if n.articulation == "legato")

    if num_staccato > len(ev.notes) * 0.3:
        tips.append(("You are playing too staccato. Hold the notes longer for smoother phrasing.",
                     num_staccato / len(ev.notes)))
    elif num_legato > len(ev.notes) * 0.3:
        tips.append(("You are playing too legato. Try separating the notes a bit more.",
                     num_legato / len(ev.notes)))
    elif num_staccato + num_legato > 0:
        tips.append(("To reach perfection, refine your articulation to match the reference.", (num_staccato + num_legato) / len(ev.notes)))

    if ev.pedal_score < 0.8:
        tips.append(("Your pedal usage needs improvement. Listen carefully to the pedal changes in the reference.",
                     1.0 - ev.pedal_score))
    elif ev.pedal_score < 1.0:
        tips.append(("To reach perfection, perfect your pedal timing and depth.", 1.0 - ev.pedal_score))

    rh_issues = ev.hand_summary.get("rh", HandIssueSummary())
    lh_issues = ev.hand_summary.get("lh", HandIssueSummary())
    if rh_issues.total_issues > lh_issues.total_issues * 1.5:
        tips.append(("Your right hand seems to have more mistakes. Focus on right-hand passages.", 0.6))
    elif lh_issues.total_issues > rh_issues.total_issues * 1.5:
        tips.append(("Your left hand seems to struggle more. Slow down left-hand parts for clarity.", 0.6))

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

    if encouragement and tips_sorted:
        return [encouragement + tip for tip in tips_sorted]
    elif encouragement and not tips_sorted:
        return [encouragement.strip()]
    return tips_sorted


class Evaluator:
    def __init__(self, recording: MidiTrack, reference: tuple[MidiTrack, MidiTrack]):
        self.recording: MidiTrack = recording
        self.reference: tuple[MidiTrack, MidiTrack] = reference

        self.weights: dict[issue_category | str, float] = {
            "accuracy": 0.30,
            "timing": 0.20,
            "dynamics": 0.15,
            "articulation": 0.15,
            "pedal": 0.10,
            "tempo": 0.10,
        }

        self._evaluation: Optional[PerformanceEvaluation] = None

    @property
    def full_evaluation(self) -> PerformanceEvaluation:
        if not self._evaluation:
            self._evaluate()
        return self._evaluation

    @property
    def score(self) -> float:
        return self.full_evaluation.overall_score

    @property
    def tip(self) -> str:
        evaluation = self.full_evaluation
        if not evaluation.comments:
            evaluation.comments = _generate_tips(evaluation)
        return evaluation.comments[0] if evaluation.comments else "No major issues detected. Great job!"

    def _evaluate(self):
        ref_rh, pedals_rh = _extract_notes_and_pedal(self.reference[0], mark="rh")
        ref_lh, pedals_lh = _extract_notes_and_pedal(self.reference[1], mark="lh")
        rec_notes, rec_pedals = _extract_notes_and_pedal(self.recording)

        ref_all = sorted(ref_rh + ref_lh, key=lambda note: note.onset_ms)
        ref_pedals = sorted(pedals_rh + pedals_lh, key=lambda pedal: pedal.time_ms)

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
            self.weights["accuracy"] * evaluation.accuracy_score +
            self.weights["timing"] * evaluation.timing_score +
            self.weights["dynamics"] * evaluation.dynamics_score +
            self.weights["articulation"] * evaluation.articulation_score +
            self.weights["pedal"] * evaluation.pedal_score +
            self.weights["tempo"] * evaluation.tempo_accuracy_score
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
