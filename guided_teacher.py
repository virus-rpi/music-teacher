from collections import deque

NOTES_PER_SECTION = 4

class GuidedTeacher:
    def __init__(self, midi_teacher, synth):
        self._section_progress = None
        self.midi_teacher = midi_teacher
        self.synth = synth
        self.is_active = False
        self.tasks = deque()
        self.history = []
        self.current_task = None
        self.measure_sections = []
        self.current_measure_index = 0
        self.current_section_visual_info = None
        self.last_score = 0.0
        self._feedback_given = False

    def get_last_score(self):
        return self.last_score

    def start(self):
        self.is_active = True
        self.current_measure_index = 0
        self._feedback_given = False
        self._generate_tasks_for_measure(self.current_measure_index)

    def stop(self):
        self.is_active = False
        self.tasks.clear()
        self.history.clear()
        self.current_task = None

    def update(self, pressed_notes, pressed_note_events):
        if not self.is_active or not self.current_task:
            return

        if self.current_task['type'] == 'practice_section':
            self._handle_practice_section(pressed_notes, pressed_note_events)
        elif self.current_task['type'] == 'practice_measure':
            self._handle_practice_measure(pressed_notes, pressed_note_events)
        elif self.current_task['type'] == 'practice_transition':
            self._handle_practice_transition(pressed_notes, pressed_note_events)

    def _generate_tasks_for_measure(self, measure_index):
        self.tasks.clear()
        self.measure_sections = self._split_measure_into_sections(measure_index)
        
        # Play the whole measure once
        self.synth.play_measure(measure_index, self.midi_teacher)

        for section in self.measure_sections:
            self.tasks.append({'type': 'practice_section', 'section': section, 'measure': measure_index})
        
        self.tasks.append({'type': 'practice_measure', 'measure': measure_index})
        self.tasks.append({'type': 'practice_transition', 'from_measure': measure_index, 'to_measure': measure_index + 1})
        
        self._next_task()

    def _split_measure_into_sections(self, measure_index):
        measure_chords, measure_times, measure_xs, _ = self.midi_teacher.get_notes_for_measure(measure_index)
        if not measure_chords:
            return []

        sections = []
        notes_per_section = min(NOTES_PER_SECTION, len(measure_chords))

        for i in range(len(measure_chords) - notes_per_section + 1):
            section_chords = measure_chords[i:i + notes_per_section]
            section_times = measure_times[i:i + notes_per_section]
            section_xs = measure_xs[i:i + notes_per_section]
            sections.append((section_chords, section_times, section_xs))

        print(f"[Debug] Split measure {measure_index} into {len(sections)} sections.")

        return sections

    def _handle_practice_section(self, pressed_notes, pressed_note_events):
        if not pressed_note_events:
            return

        section_chords, section_times, _ = self.current_task['section']

        # Initialize progress tracking if not present or if section changed
        if not hasattr(self, '_section_progress') or self._section_progress.get('section_id') != id(self.current_task['section']):
            self._section_progress = {
                'section_id': id(self.current_task['section']),
                'chord_idx': 0,
                'matched_events': []
            }

        chord_idx = self._section_progress['chord_idx']
        matched_events = self._section_progress['matched_events']

        # Try to match the next chord in order, allowing extra notes and handling single-note events
        while chord_idx < len(section_chords) and pressed_note_events:
            expected_chord = set(note for note, _ in section_chords[chord_idx])
            for i, (notes, timestamp) in enumerate(pressed_note_events):
                # Ensure notes is always a set
                if isinstance(notes, int):
                    played_set = {notes}
                else:
                    played_set = set(notes)
                # Only require all expected notes to be present (ignore extra notes)
                if expected_chord.issubset(played_set):
                    matched_events.append((notes, timestamp))
                    chord_idx += 1
                    pressed_note_events = pressed_note_events[i+1:]
                    break
            else:
                break

        self._section_progress['chord_idx'] = chord_idx
        self._section_progress['matched_events'] = matched_events

        # If all chords matched in order, evaluate
        if chord_idx == len(section_chords):
            # Build pressed_notes and pressed_note_events for evaluation (record all notes, including wrong ones)
            eval_pressed_notes = set(n for notes, _ in matched_events for n in (notes if isinstance(notes, (list, tuple, set)) else [notes]))
            eval_pressed_note_events = [(notes, ts) for notes, ts in matched_events]
            self.last_score = self._evaluate_performance(section_chords, section_times, eval_pressed_notes, eval_pressed_note_events)

            if self.last_score >= 0.8:
                self.history.append(self.current_task)
                self._section_progress = {}  # Reset for next section
                self._next_task()
            else:
                if not self._feedback_given:
                    self.synth.play_notes(section_chords, section_times)
                    self._feedback_given = True
                self._section_progress = {}  # Reset for retry

    def _handle_practice_measure(self, pressed_notes, pressed_note_events):
        measure_index = self.current_task['measure']
        measure_chords, measure_times, _, _ = self.midi_teacher.get_notes_for_measure(measure_index)
        self.last_score = self._evaluate_performance(measure_chords, measure_times, pressed_notes, pressed_note_events)

        if self.last_score >= 0.8:
            self.history.append(self.current_task)
            self._next_task()
        else:
            if not self._feedback_given:
                self.synth.play_measure(measure_index, self.midi_teacher)
                self._feedback_given = True

    def _handle_practice_transition(self, pressed_notes, pressed_note_events):
        from_measure = self.current_task['from_measure']
        to_measure = self.current_task['to_measure']
        
        transition_chords, transition_times, _ = self._get_transition_notes(from_measure, to_measure)
        self.last_score = self._evaluate_performance(transition_chords, transition_times, pressed_notes, pressed_note_events)

        if self.last_score >= 0.8:
            self.history.append(self.current_task)
            self.current_measure_index += 1
            self._generate_tasks_for_measure(self.current_measure_index)
        else:
            if not self._feedback_given:
                self.synth.play_notes(transition_chords, transition_times)
                self._feedback_given = True

    def _get_transition_notes(self, from_measure, to_measure):
        from_chords, from_times, from_xs = self.midi_teacher.get_notes_for_measure(from_measure)
        to_chords, to_times, to_xs = self.midi_teacher.get_notes_for_measure(to_measure)
        if not from_chords or not to_chords:
            return [], [], []
        return from_chords[-2:] + to_chords[:2], from_times[-2:] + to_times[:2], from_xs[-2:] + to_xs[:2]

    def _evaluate_performance(self, expected_chords, expected_times, pressed_notes, pressed_note_events):
        if not expected_chords:
            return 1.0
        if not pressed_note_events:
            return 0.0

        expected_note_set = {note for chord in expected_chords for note, hand in chord}
        # Accuracy
        accuracy = len(expected_note_set.intersection(pressed_notes)) / len(expected_note_set)

        # Relative Timing
        expected_intervals = [expected_times[i+1] - expected_times[i] for i in range(len(expected_times)-1)]
        pressed_intervals = [pressed_note_events[i+1][1] - pressed_note_events[i][1] for i in range(len(pressed_note_events)-1)]
        relative_timing_score = 0.0
        if expected_intervals and pressed_intervals:
            sum_expected = sum(expected_intervals)
            sum_pressed = sum(pressed_intervals)
            if sum_expected > 0 and sum_pressed > 0:
                expected_intervals_norm = [i / sum_expected for i in expected_intervals]
                pressed_intervals_norm = [i / sum_pressed for i in pressed_intervals]
                min_len = min(len(expected_intervals_norm), len(pressed_intervals_norm))
                correlation = sum(expected_intervals_norm[i] * pressed_intervals_norm[i] for i in range(min_len))
                relative_timing_score = correlation

        # Absolute Timing
        absolute_timing_score = 0.0
        if expected_times and pressed_note_events:
            time_diff = pressed_note_events[0][1] - expected_times[0]
            total_diff = 0
            for i in range(min(len(expected_times), len(pressed_note_events))):
                total_diff += abs((pressed_note_events[i][1] - time_diff) - expected_times[i])
            avg_diff = total_diff / min(len(expected_times), len(pressed_note_events))
            absolute_timing_score = max(0.0, 1 - (avg_diff / 1000.0)) # 1 second diff = 0 score

        final_score = (accuracy * 1.0) + (relative_timing_score * 0.6) + (absolute_timing_score * 0.2)
        return final_score / (1.0 + 0.6 + 0.2)

    def _next_task(self):
        if self.tasks:
            self.current_task = self.tasks.popleft()
            if self.current_task['type'] == 'practice_section':
                self.current_section_visual_info = self.current_task['section'][2]
            else:
                self.current_section_visual_info = None
            self._feedback_given = False
        else:
            self.current_task = None
            self.is_active = False # Or move to the next set of measures
            self.current_section_visual_info = None
            self._feedback_given = False

    def repeat_measure(self):
        if self.current_task:
            measure_index = self.current_task.get('measure')
            if measure_index is not None:
                self.synth.play_measure(measure_index, self.midi_teacher)

    def get_current_task_info(self):
        return self.current_task
