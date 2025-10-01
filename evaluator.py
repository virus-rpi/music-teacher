from dataclasses import dataclass, field
from pprint import pprint
from typing import List, Optional
import numpy as np
from mido import MidiTrack
import difflib


@dataclass
class NoteEvaluation:
    pitch_correct: bool
    pitch_error: Optional[int] = None
    onset_deviation_ms: float = 0.0
    duration_deviation_ms: float = 0.0
    velocity_deviation: float = 0.0
    articulation: Optional[str] = None   # "staccato", "legato", "normal"
    pedal_used: Optional[bool] = None
    time_ms: Optional[int] = None        # onset time for later highlighting
    comments: Optional[str] = None


@dataclass
class Issue:
    time_ms: int
    note: Optional[int] = None
    severity: float = 0.0    # 0–1 scaled severity
    category: str = ""       # e.g. "pitch", "timing", "dynamics", "pedal", "articulation"
    description: str = ""


@dataclass
class PerformanceEvaluation:
    notes: List[NoteEvaluation] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)

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
    pedal_accuracy: float = 0.0

    phrasing_similarity: float = 0.0
    hand_independence: float = 0.0
    chord_accuracy: float = 0.0

    overall_score: float = 0.0
    comments: Optional[str] = None


def _extract_notes_and_pedal(track: MidiTrack):
    """
    Extract notes and pedal events (CC64).
    """
    notes = []
    pedals = []
    current_time = 0
    note_on = {}

    for msg in track:
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            note_on[msg.note] = (current_time, msg.velocity)
        elif msg.type in ("note_off", "note_on") and msg.note in note_on:
            start_time, velocity = note_on.pop(msg.note)
            duration = current_time - start_time
            notes.append({
                "pitch": msg.note,
                "onset_ms": start_time,
                "duration_ms": duration,
                "velocity": velocity
            })
        elif msg.type == "control_change" and msg.control == 64:
            pedals.append({"time_ms": current_time, "value": msg.value})

    return notes, pedals


def _match_notes(ref_notes, rec_notes):
    ref_pitches = [n["pitch"] for n in ref_notes]
    rec_pitches = [n["pitch"] for n in rec_notes]

    matcher = difflib.SequenceMatcher(a=ref_pitches, b=rec_pitches, autojunk=False)
    matches = []
    extras = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                matches.append((ref_notes[i1 + k], rec_notes[j1 + k]))
        elif tag == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                matches.append((ref_notes[i1 + k], rec_notes[j1 + k]))
            for k in range(i1 + (j2 - j1), i2):
                matches.append((ref_notes[k], None))  # missing
            for k in range(j1 + (i2 - i1), j2):
                extras.append(rec_notes[k])  # extra
        elif tag == "delete":
            for k in range(i1, i2):
                matches.append((ref_notes[k], None))
        elif tag == "insert":
            for k in range(j1, j2):
                extras.append(rec_notes[k])

    return matches, extras


def _detect_articulation(duration, reference_duration):
    """
    Classify note articulation relative to reference.
    """
    ratio = duration / (reference_duration + 1e-6)
    if ratio < 0.5:
        return "staccato"
    elif ratio > 1.2:
        return "legato"
    return "normal"


class Evaluator:
    def __init__(self, recording: MidiTrack, reference: tuple[MidiTrack, MidiTrack]):
        self.recording: MidiTrack = recording
        self.reference: tuple[MidiTrack, MidiTrack] = reference

        self.weights: dict[str, float] = {
            "pitch": 0.35,
            "timing": 0.25,
            "dynamics": 0.15,
            "articulation": 0.15,
            "pedal": 0.10,
        }

        self._evaluation: Optional[PerformanceEvaluation] = None


    @property
    def score(self) -> float:
        if not self._evaluation:
            self._evaluate()
        return self._evaluation.overall_score

    @property
    def tip(self) -> str:
        return ""

    def _evaluate(self):
        ref_rh, pedals_rh = _extract_notes_and_pedal(self.reference[0])
        ref_lh, pedals_lh = _extract_notes_and_pedal(self.reference[1])
        rec_notes, rec_pedals = _extract_notes_and_pedal(self.recording)

        ref_all = sorted(ref_rh + ref_lh, key=lambda n: n["onset_ms"])
        ref_pedals = sorted(pedals_rh + pedals_lh, key=lambda p: p["time_ms"])

        matches, extras = _match_notes(ref_all, rec_notes)
        evaluation = PerformanceEvaluation(total_notes=len(ref_all))

        for r, played in matches:
            if played is None:
                evaluation.missing_notes += 1
                evaluation.notes.append(NoteEvaluation(
                    pitch_correct=False,
                    time_ms=r["onset_ms"],
                    comments="Note missing"
                ))
                evaluation.issues.append(Issue(
                    time_ms=r["onset_ms"], severity=1.0,
                    category="pitch", description=f"Missing note {r['pitch']}"
                ))
                continue

            pitch_correct = r["pitch"] == played["pitch"]
            pitch_error = None if pitch_correct else (played["pitch"] - r["pitch"])
            onset_dev = played["onset_ms"] - r["onset_ms"]
            dur_dev = played["duration_ms"] - r["duration_ms"]
            vel_dev = played["velocity"] - r["velocity"]

            articulation = _detect_articulation(played["duration_ms"], r["duration_ms"])

            note_eval = NoteEvaluation(
                pitch_correct=pitch_correct,
                pitch_error=pitch_error,
                onset_deviation_ms=onset_dev,
                duration_deviation_ms=dur_dev,
                velocity_deviation=vel_dev,
                articulation=articulation,
                time_ms=played["onset_ms"],
            )
            evaluation.notes.append(note_eval)

            if not pitch_correct:
                evaluation.wrong_notes += 1
                evaluation.issues.append(Issue(
                    time_ms=played["onset_ms"], note=played["pitch"],
                    severity=min(abs(pitch_error)/12, 1.0),
                    category="pitch", description=f"Wrong pitch {played['pitch']} vs {r['pitch']}"
                ))
            else:
                evaluation.correct_notes += 1

            if abs(onset_dev) > 30:
                evaluation.issues.append(Issue(
                    time_ms=played["onset_ms"], note=played["pitch"],
                    severity=min(abs(onset_dev)/200, 1.0),
                    category="timing", description=f"Timing off by {onset_dev:.1f} ms"
                ))

            if articulation != "normal":
                evaluation.issues.append(Issue(
                    time_ms=played["onset_ms"], note=played["pitch"],
                    severity=0.5,
                    category="articulation", description=f"Played {articulation}"
                ))

        evaluation.extra_notes = len(extras)
        for n in extras:
            evaluation.issues.append(Issue(
                time_ms=n["onset_ms"], note=n["pitch"],
                severity=0.5,
                category="pitch", description=f"Extra note {n['pitch']}"
            ))

        if evaluation.notes:
            evaluation.avg_timing_deviation_ms = float(np.mean([abs(n.onset_deviation_ms) for n in evaluation.notes]))
            evaluation.avg_duration_deviation_ms = float(np.mean([abs(n.duration_deviation_ms) for n in evaluation.notes]))
            evaluation.avg_velocity_deviation = float(np.mean([abs(n.velocity_deviation) for n in evaluation.notes]))

        deviations = [n.onset_deviation_ms for n in evaluation.notes]
        if len(deviations) > 1:
            evaluation.rhythmic_stability = float(np.std(deviations))
        evaluation.tempo_consistency = 1.0 / (1.0 + evaluation.rhythmic_stability)

        ref_pedal_on = sum(1 for p in ref_pedals if p["value"] >= 64)
        rec_pedal_on = sum(1 for p in rec_pedals if p["value"] >= 64)
        evaluation.pedal_accuracy = 1.0 - abs(ref_pedal_on - rec_pedal_on) / max(1, len(ref_pedals))

        if abs(ref_pedal_on - rec_pedal_on) > 0:
            evaluation.issues.append(Issue(
                time_ms=0,
                severity=0.5,
                category="pedal",
                description=f"Pedal usage differs (ref {ref_pedal_on}, rec {rec_pedal_on})"
            ))

        pitch_score = evaluation.correct_notes / max(1, evaluation.total_notes)
        timing_score = max(0.0, 1.0 - evaluation.avg_timing_deviation_ms / 200.0)
        dynamics_score = max(0.0, 1.0 - evaluation.avg_velocity_deviation / 40.0)
        articulation_score = 1.0 - sum(1 for i in evaluation.issues if i.category == "articulation") / max(1, evaluation.total_notes)
        pedal_score = evaluation.pedal_accuracy

        evaluation.overall_score = (
            self.weights["pitch"] * pitch_score +
            self.weights["timing"] * timing_score +
            self.weights["dynamics"] * dynamics_score +
            self.weights["articulation"] * articulation_score +
            self.weights["pedal"] * pedal_score
        )

        self._evaluation = evaluation
        pprint(evaluation)
