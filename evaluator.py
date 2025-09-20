from dataclasses import dataclass
import mido
from typing import Optional
from midi_teach import MidiTeacher

WEIGHTS = {
    'accuracy': 0.6,
    'relative': 0.25,
    'absolute': 0.15,
}

assert WEIGHTS['accuracy'] + WEIGHTS['relative'] + WEIGHTS['absolute'] == 1.0, "Weights must sum to 1.0"


@dataclass
class Score:
    accuracy: float
    relative_timing: float
    absolute_timing: float
    overall: float

class Evaluator:
    def __init__(self, section_task):
        self.section = section_task.section
        self.start_idx = section_task.start_idx
        self.end_idx = section_task.end_idx
        self.recording = section_task.recording
        self.teacher: MidiTeacher = section_task.teacher.midi_teacher

        self._score = None
        self.analytics: Optional[dict] = None

    @property
    def score(self):
        if not self._score:
            self._score = self._evaluate()
        return self._score

    def _evaluate(self) -> Score:
        events = []
        abs_ms = 0
        for msg in self.recording:
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

        if not self.section.chords:
            self.analytics = {'reason': 'no_chords'}
            return Score(1.0, 1.0, 1.0, 1.0)

        chord_indices = list(range(self.start_idx, self.end_idx + 1))
        chord_ticks = []
        for idx in chord_indices:
            try:
                chord_ticks.append(self.teacher.chord_times[idx])
            except Exception:
                chord_ticks.append(None)

        def tick_to_seconds(tick):
            if tick is None:
                return 0.0
            mid = mido.MidiFile(self.teacher.midi_path)
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
            measure_start_tick = self.teacher.chord_times[self.start_idx] if self.start_idx < len(self.teacher.chord_times) else 0
            ref_abs_secs = [tick_to_seconds(measure_start_tick + t) for t in self.section.times]

        ref_start = ref_abs_secs[0]
        ref_onsets_ms = [int((s - ref_start) * 1000.0) for s in ref_abs_secs]
        ref_chords = [set(n for n, _hand in chord) for chord in self.section.chords]

        if not recorded_onsets:
            self.analytics = {'reason': 'no_recording'}
            return Score(0.0, 0.0, 0.0, 0.0)

        n_ref = len(ref_onsets_ms)
        n_rec = len(recorded_onsets)
        m = min(n_ref, n_rec)

        acc_scores = []
        per_chord_issues = []  # list of dicts with missed/extra/score
        for i in range(m):
            ref_set = ref_chords[i]
            rec_set = recorded_onsets[i][1]
            if not ref_set and not rec_set:
                acc_scores.append(1.0)
                per_chord_issues.append({'missed': set(), 'extra': set(), 'score': 1.0})
                continue
            inter = len(ref_set & rec_set)
            denom = (len(ref_set) + len(rec_set))
            if denom == 0:
                s = 1.0
            else:
                s = (2.0 * inter) / denom
            acc_scores.append(s)
            per_chord_issues.append({'missed': (ref_set - rec_set), 'extra': (rec_set - ref_set), 'score': s})
        if n_rec < n_ref:
            acc_scores.extend([0.0] * (n_ref - n_rec))
            for i in range(n_rec, n_ref):
                per_chord_issues.append({'missed': ref_chords[i], 'extra': set(), 'score': 0.0})
        accuracy_score = sum(acc_scores) / max(1, n_ref)

        def intervals(xs):
            return [xs[i + 1] - xs[i] for i in range(len(xs) - 1)] if len(xs) > 1 else []

        ref_intervals = intervals(ref_onsets_ms[:m])
        rec_intervals = intervals([t for t, _ in recorded_onsets[:m]])
        interval_diffs = []
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
                interval_diffs = [abs(nref[i] - nrec[i]) for i in range(L)]

        ref_total = (ref_onsets_ms[-1] - ref_onsets_ms[0]) if len(ref_onsets_ms) > 1 else 0
        rec_total = (recorded_onsets[m - 1][0] - recorded_onsets[0][0]) if m > 1 else 0
        if ref_total == 0:
            absolute_score = 1.0 if rec_total == 0 else 0.0
        else:
            rel_error = abs(rec_total - ref_total) / float(ref_total)
            absolute_score = max(0.0, 1.0 - min(rel_error, 1.0))

        score = (WEIGHTS['accuracy'] * accuracy_score) + (WEIGHTS['relative'] * relative_score) + (WEIGHTS['absolute'] * absolute_score)
        score = max(0.0, min(1.0, score))

        # collect analytics to help generate guidance
        # find worst chord by accuracy
        worst_idx = None
        worst_score = 1.0
        for i, info in enumerate(per_chord_issues):
            if info['score'] < worst_score:
                worst_score = info['score']
                worst_idx = i
        worst_interval_idx = None
        worst_interval_diff = 0.0
        for i, d in enumerate(interval_diffs):
            if d > worst_interval_diff:
                worst_interval_diff = d
                worst_interval_idx = i

        tempo_bias = None
        if ref_total > 0 and m > 1:
            tempo_bias = float(rec_total - ref_total) / float(ref_total)

        analytics = {
            'accuracy': accuracy_score,
            'per_chord': per_chord_issues,
            'worst_chord_idx': worst_idx,
            'worst_chord_score': worst_score,
            'relative': relative_score,
            'interval_diffs': interval_diffs,
            'worst_interval_idx': worst_interval_idx,
            'worst_interval_diff': worst_interval_diff,
            'absolute': absolute_score,
            'tempo_bias': tempo_bias,
            'n_ref': n_ref,
            'n_rec': n_rec,
            'ref_onsets_ms': ref_onsets_ms,
            'rec_onsets_ms': [t for t, _ in recorded_onsets[:m]],
        }

        self.analytics = analytics

        return Score(accuracy_score, relative_score, absolute_score, score)

    def _ordinal(self, n: int) -> str:
        if 10 <= (n % 100) <= 20:
            suf = 'th'
        else:
            suf = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suf}"

    def generate_guidance(self, score: Score) -> str:
        """Generate a short human-friendly guidance message based on the last evaluation analytics.
        The method prefers the single largest-impact problem (accuracy, relative timing, absolute timing).
        """
        if not self.analytics:
            return "No guidance available."

        # compute component impacts (how much they reduced the overall score)
        acc_impact = WEIGHTS['accuracy'] * (1.0 - self.analytics.get('accuracy', 1.0))
        rel_impact = WEIGHTS['relative'] * (1.0 - self.analytics.get('relative', 1.0))
        abs_impact = WEIGHTS['absolute'] * (1.0 - self.analytics.get('absolute', 1.0))

        impacts = [('accuracy', acc_impact), ('relative', rel_impact), ('absolute', abs_impact)]
        impacts.sort(key=lambda x: x[1], reverse=True)
        primary = impacts[0][0]

        worst_chord_idx = self.analytics.get('worst_chord_idx')
        per_chord = self.analytics.get('per_chord', [])
        worst_interval_idx = self.analytics.get('worst_interval_idx')
        tempo_bias = self.analytics.get('tempo_bias')

        if score.overall == 1.0:
            return "Machine Perfect! Keep it up! But don't forget to play with feeling."

        if score.overall >= 0.98:
            if tempo_bias is not None and abs(tempo_bias) > 0.02:
                if tempo_bias > 0:
                    return "Almost perfect! Now play a little faster overall."
                else:
                    return "Almost perfect! Now play a little slower overall."
            return "Excellent — almost perfect! Keep it up."

        if self.analytics.get('n_rec', 0) == 0:
            return "I didn't detect any notes. Try playing the section more clearly or check your MIDI input."

        if primary == 'accuracy':
            if worst_chord_idx is None:
                return "Focus on pressing the right keys."
            info = per_chord[worst_chord_idx]
            missed = info.get('missed', set())
            extra = info.get('extra', set())
            human_idx = worst_chord_idx + 1
            if missed and not extra:
                notes = ','.join(str(n) for n in sorted(missed))
                return f"Focus on pressing the right keys at the {self._ordinal(human_idx)} chord (you missed: {notes})."
            if extra and not missed:
                notes = ','.join(str(n) for n in sorted(extra))
                return f"Avoid extra keys at the {self._ordinal(human_idx)} chord (extra: {notes})."
            if missed and extra:
                return f"At the {self._ordinal(human_idx)} chord you both missed some notes and played extras — practice that chord slowly."
            return "Focus on pressing the right keys consistently. Try playing slower and cleanly."

        if primary == 'relative':
            if worst_interval_idx is None:
                return "Work on keeping consistent spacing between notes. Practice slowly with a metronome."
            a = worst_interval_idx + 1
            b = worst_interval_idx + 2
            ref_onsets = self.analytics.get('ref_onsets_ms', [])
            rec_onsets = self.analytics.get('rec_onsets_ms', [])
            if len(ref_onsets) > worst_interval_idx + 1 and len(rec_onsets) > worst_interval_idx + 1:
                ref_interval = ref_onsets[worst_interval_idx + 1] - ref_onsets[worst_interval_idx]
                rec_interval = rec_onsets[worst_interval_idx + 1] - rec_onsets[worst_interval_idx]
                if rec_interval > ref_interval:
                    return f"Leave more time between the {self._ordinal(a)} and the {self._ordinal(b)} onset — you're stretching that gap."
                else:
                    return f"Bring the {self._ordinal(a)} and the {self._ordinal(b)} closer together — you're making that gap too short."
            return "Work on the timing between consecutive onsets — one gap is noticeably off."

        if primary == 'absolute':
            if tempo_bias is None:
                return "Try matching the overall tempo of the section. Use a metronome."
            pct = abs(tempo_bias) * 100
            if tempo_bias > 0:
                if pct < 5:
                    return "Almost perfect! Now just play it a little faster overall."
                return f"Play the whole section faster by about {int(pct)}% to match the target tempo."
            else:
                if pct < 5:
                    return "Almost perfect! Now just play it a little slower overall."
                return f"Play the whole section slower by about {int(pct)}% to match the target tempo."

        return "Keep practicing — try slowing down and focusing on accuracy and steady timing."
