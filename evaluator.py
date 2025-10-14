import difflib
from collections import defaultdict
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional, Literal
import numpy as np
from mido import MidiTrack

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

    overall_score: float = 0.0
    accuracy_score: float = 0.0
    timing_score: float = 0.0
    dynamics_score: float = 0.0
    articulation_score: float = 0.0
    pedal_score: float = 0.0
    comments: Optional[str] = None

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


def _match_notes(ref_notes: list[Note], rec_notes: list[Note]) -> tuple[list[tuple[Note, Optional[Note]]], list[Note]]:
    """
    Alternative algorithm:
    1. Strip the furthest notes (by onset/pitch/duration distance) from rec_notes until len(rec_notes) == len(ref_notes), collecting extras.
    2. Normalize rec_notes tempo to match ref_notes (scale onset/duration).
    3. Match each rec_note to the closest ref_note (using a distance metric combining onset, pitch, and duration).
    4. Return the matches and extras.
    """
    if not ref_notes:
        return [], rec_notes.copy()
    if not rec_notes:
        return [(r, None) for r in ref_notes], []
    rec_notes_work = rec_notes.copy()
    extras = []
    def note_distance(n1: Note, n2: Note) -> float:
        return (
            abs(n1.onset_ms - n2.onset_ms) / 100.0 +
            abs(n1.pitch - n2.pitch) +
            abs(n1.duration_ms - n2.duration_ms) / 100.0
        )
    while len(rec_notes_work) > len(ref_notes):
        distances = [min(note_distance(rn, ref) for ref in ref_notes) for rn in rec_notes_work]
        idx = int(np.argmax(distances))
        extras.append(rec_notes_work.pop(idx))

    ref_onsets = np.array([n.onset_ms for n in ref_notes])
    rec_onsets = np.array([n.onset_ms for n in rec_notes_work])
    if len(ref_onsets) > 1 and len(rec_onsets) > 1:
        ref_span = ref_onsets[-1] - ref_onsets[0]
        rec_span = rec_onsets[-1] - rec_onsets[0]
        scale = (ref_span / rec_span) if rec_span > 0 else 1.0
    else:
        scale = 1.0
    rec_notes_norm = []
    for n in rec_notes_work:
        onset = int(ref_onsets[0] + (n.onset_ms - rec_onsets[0]) * scale)
        duration = int(n.duration_ms * scale)
        rec_notes_norm.append(Note(
            pitch=n.pitch,
            onset_ms=onset,
            duration_ms=duration,
            velocity=n.velocity,
            mark=n.mark
        ))
    ref_used = set()
    matches = []
    for rec in rec_notes_norm:
        best_idx = None
        best_dist = float('inf')
        for i, ref in enumerate(ref_notes):
            if i in ref_used:
                continue
            dist = note_distance(rec, ref)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None:
            matches.append((ref_notes[best_idx], rec_notes_work[rec_notes_norm.index(rec)]))
            ref_used.add(best_idx)
    for i, ref in enumerate(ref_notes):
        if i not in ref_used:
            matches.append((ref, None))
    return matches, extras


def _detect_articulation(duration: int, reference_duration: int) -> articulation_type:
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


class Evaluator:
    def __init__(self, recording: MidiTrack, reference: tuple[MidiTrack, MidiTrack]):
        self.recording: MidiTrack = recording
        self.reference: tuple[MidiTrack, MidiTrack] = reference

        self.weights: dict[issue_category, float] = {
            "accuracy": 0.35,
            "timing": 0.25,
            "dynamics": 0.15,
            "articulation": 0.15,
            "pedal": 0.10,
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
        return ""

    def _evaluate(self):
        ref_rh, pedals_rh = _extract_notes_and_pedal(self.reference[0])
        ref_lh, pedals_lh = _extract_notes_and_pedal(self.reference[1])
        rec_notes, rec_pedals = _extract_notes_and_pedal(self.recording)

        ref_all = sorted(ref_rh + ref_lh, key=lambda note: note.onset_ms)
        ref_pedals = sorted(pedals_rh + pedals_lh, key=lambda pedal: pedal.time_ms)

        matches, extras = _match_notes(ref_all, rec_notes)
        evaluation = PerformanceEvaluation(total_notes=len(ref_all))

        for r, played in matches:
            if played is None:
                evaluation.missing_notes += 1
                evaluation.notes.append(NoteEvaluation(
                    pitch_correct=False,
                    time_ms=r.onset_ms,
                    comments="Note missing"
                ))
                evaluation.issues.append(Issue(
                    time_ms=r.onset_ms, severity=1.0,
                    category="accuracy", description=f"Missing note {_pitch_to_name(r.pitch)}"
                ))
                continue

            pitch_correct = r.pitch == played.pitch
            pitch_error = None if pitch_correct else (played.pitch - r.pitch)
            onset_dev = played.onset_ms - r.onset_ms
            dur_dev = played.duration_ms - r.duration_ms
            vel_dev = played.velocity - r.velocity

            articulation = _detect_articulation(played.duration_ms, r.duration_ms)

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
                evaluation.wrong_notes += 1
                evaluation.issues.append(Issue(
                    time_ms=played.onset_ms, note=played.pitch,
                    severity=min(abs(pitch_error)/12, 1.0),
                    category="accuracy", description=f"Wrong pitch {_pitch_to_name(played.pitch)} vs {_pitch_to_name(r.pitch)}"
                ))
            else:
                evaluation.correct_notes += 1

            if abs(onset_dev) > 30:
                evaluation.issues.append(Issue(
                    time_ms=played.onset_ms, note=played.pitch,
                    severity=min(abs(onset_dev)/200, 1.0),
                    category="timing", description=f"Timing off by {onset_dev:.1f} ms"
                ))

            if articulation != "normal":
                evaluation.issues.append(Issue(
                    time_ms=played.onset_ms, note=played.pitch,
                    severity=0.5,
                    category="articulation", description=f"Played {articulation}"
                ))

        evaluation.extra_notes = len(extras)
        for n in extras:
            evaluation.issues.append(Issue(
                time_ms=n.onset_ms, note=n.pitch,
                severity=0.5,
                category="accuracy", description=f"Extra note {n.pitch}"
            ))

        if evaluation.notes:
            evaluation.avg_timing_deviation_ms = float(np.mean([abs(n.onset_deviation_ms) for n in evaluation.notes]))
            evaluation.avg_duration_deviation_ms = float(np.mean([abs(n.duration_deviation_ms) for n in evaluation.notes]))
            evaluation.avg_velocity_deviation = float(np.mean([abs(n.velocity_deviation) for n in evaluation.notes]))

        deviations = [n.onset_deviation_ms for n in evaluation.notes]
        if len(deviations) > 1:
            evaluation.rhythmic_stability = float(np.std(deviations))
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
            self.weights["pedal"] * evaluation.pedal_score
        )

        self._evaluation = evaluation
        pprint(evaluation)
