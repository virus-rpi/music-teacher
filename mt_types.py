"""Common type definitions for the music teacher application."""

from dataclasses import dataclass, field
from typing import Optional, Literal
from collections import defaultdict

articulation_type = Literal["staccato", "legato", "normal"]
issue_category = Literal["accuracy", "timing", "dynamics", "pedal", "articulation"]
pedal_type = Literal["sustain", "sostenuto", "soft"]


@dataclass(frozen=True)
class Note:
    """Represents a single MIDI note with timing and velocity information."""
    pitch: int
    onset_ms: int
    duration_ms: int
    velocity: int
    mark: Optional[str] = None

    def copy(self, **kwargs):
        """
        Return a copy of the Note with any provided fields replaced.
        
        Parameters:
            **kwargs: Optional fields to override on the copied Note. Accepted keys are:
                pitch (int): MIDI pitch value.
                onset_ms (int): Onset time in milliseconds.
                duration_ms (int): Duration in milliseconds.
                velocity (int): Note velocity.
                mark (Optional[str]): Optional mark or annotation.
        
        Returns:
            Note: A new Note instance with fields taken from `kwargs` when provided, otherwise copied from the original.
        """
        return Note(
            pitch=kwargs.get('pitch', self.pitch),
            onset_ms=kwargs.get('onset_ms', self.onset_ms),
            duration_ms=kwargs.get('duration_ms', self.duration_ms),
            velocity=kwargs.get('velocity', self.velocity),
            mark=kwargs.get('mark', self.mark),
        )


@dataclass
class PedalEvent:
    """Represents a pedal control change event."""
    time_ms: int
    value: int
    pedal_type: pedal_type

    def copy(self, **kwargs):
        """
        Create a new PedalEvent with one or more fields replaced by the supplied keyword arguments.
        
        Parameters:
            **kwargs: Optional overrides for fields of the new PedalEvent. Supported keys are:
                time_ms (int): Event time in milliseconds.
                value (int): Control value (typically 0–127).
                pedal_type (pedal_type): Type of pedal, e.g., "sustain", "sostenuto", or "soft".
        
        Returns:
            PedalEvent: A new PedalEvent instance with fields taken from `kwargs` when provided, otherwise copied from the original.
        """
        return PedalEvent(
            time_ms=kwargs.get('time_ms', self.time_ms),
            value=kwargs.get('value', self.value),
            pedal_type=kwargs.get('pedal_type', self.pedal_type),
        )


@dataclass
class NoteEvaluation:
    """Evaluation results for a single note."""
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
    """Represents a specific issue found during evaluation."""
    time_ms: int
    note: Optional[int] = None
    severity: float = 0.0    # 0–1 scaled severity
    category: issue_category = ""
    description: str = ""


@dataclass
class HandIssueSummary:
    """Summary of issues for a specific hand (left/right)."""
    total_issues: int = 0
    by_category: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_severity: float = 0.0


@dataclass
class PerformanceEvaluation:
    """Complete evaluation of a performance."""
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
