"""Autonomous sequence evaluation for learned BioLogic grammar transitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from biologic.encodings import random_bipolar, random_input_codebook, random_state_codebook, sign
from biologic.grammars import GrammarTask, transition_coverage
from biologic.learning import (
    EligibilityTraceTransitionLearner,
    FeatureMode,
    LearnedAssociativeTransition,
    OutputMode,
    StateInputPairEncoder,
    UnwrittenMode,
    UpdateRule,
)
from biologic.register import HopfieldRegister, NearestAttractorRegister


@dataclass(frozen=True)
class MaskedNearestRegister:
    """Nearest cleanup using only visible written coordinates."""

    codebook: np.ndarray

    def cleanup_masked(self, proposal: np.ndarray, mask: np.ndarray) -> tuple[int, np.ndarray]:
        """Return nearest state using only coordinates where ``mask`` is true."""
        visible = np.asarray(mask, dtype=bool)
        if visible.shape != (self.codebook.shape[1],):
            raise ValueError("mask has wrong shape")
        if not np.any(visible):
            return 0, self.codebook[0].copy()
        x = np.asarray(proposal, dtype=int)
        scores = self.codebook[:, visible] @ x[visible]
        state_idx = int(np.argmax(scores))
        return state_idx, self.codebook[state_idx].copy()


@dataclass
class BioLogicSequenceLearner:
    """Learn and evaluate grammar transitions from DFA traces."""

    task: GrammarTask
    state_dim: int
    input_dim: int
    rng: np.random.Generator
    feature_mode: FeatureMode = "exact_pair"
    hidden_dim: int | None = None
    register_type: str = "nearest"
    output_mode: OutputMode = "dense"
    write_fraction: float = 1.0
    unwritten_mode: UnwrittenMode = "random_noise"
    update_rule: UpdateRule = "delta"
    learning_rate: float = 1.0

    def __post_init__(self) -> None:
        self.state_to_idx = {
            state: idx for idx, state in enumerate(self.task.dfa.states)
        }
        self.idx_to_state = {
            idx: state for state, idx in self.state_to_idx.items()
        }
        self.symbol_to_idx = {
            symbol: idx for idx, symbol in enumerate(self.task.alphabet)
        }
        self.state_codebook = random_state_codebook(
            self.task.dfa.num_states,
            self.state_dim,
            self.rng,
        )
        self.input_codebook = random_input_codebook(
            len(self.task.alphabet),
            self.input_dim,
            self.rng,
        )
        self.encoder = StateInputPairEncoder(
            num_states=self.task.dfa.num_states,
            num_inputs=len(self.task.alphabet),
            mode=self.feature_mode,
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
            input_dim=self.input_dim,
            rng=self.rng,
        )
        self.writer = LearnedAssociativeTransition(
            state_dim=self.state_dim,
            feature_dim=self.encoder.feature_dim,
            rng=self.rng,
            learning_rate=self.learning_rate,
            update_rule=self.update_rule,
            output_mode=self.output_mode,
            write_fraction=self.write_fraction,
            unwritten_mode=self.unwritten_mode,
        )
        self.learner = EligibilityTraceTransitionLearner(self.writer)
        if self.register_type == "nearest":
            self.register = NearestAttractorRegister(self.state_codebook)
        elif self.register_type == "hopfield":
            self.register = HopfieldRegister.from_codebook(self.state_codebook)
        elif self.register_type == "masked_nearest":
            self.register = NearestAttractorRegister(self.state_codebook)
        else:
            raise ValueError(f"unknown register_type: {self.register_type}")

    def fit_from_dfa_traces(
        self,
        train_strings: Sequence[list[str]],
        epochs: int,
        shuffle: bool = True,
    ) -> None:
        """Train from oracle DFA traces through local transition updates."""
        examples = self._trace_examples(train_strings)
        for _ in range(epochs):
            self.learner.train_epoch(examples, rng=self.rng, shuffle=shuffle)

    def predict_trace(self, symbols: list[str]) -> list[int]:
        """Run the learned transition model autonomously and return state indices."""
        current_idx = self.state_to_idx[self.task.dfa.start_state]
        trace = [current_idx]
        current_vec = self.state_codebook[current_idx]
        for symbol in symbols:
            input_idx = self.symbol_to_idx[symbol]
            input_vec = self.input_codebook[input_idx]
            proposal = self.writer.propose(
                current_vec,
                input_vec,
                self.encoder,
                current_idx,
                input_idx,
            )
            current_idx, current_vec = self.register.cleanup(proposal)
            trace.append(current_idx)
        return trace

    def predict_accept(self, symbols: list[str]) -> bool:
        """Predict accept/reject by autonomous learned-state rollout."""
        final_idx = self.predict_trace(symbols)[-1]
        return self.idx_to_state[final_idx] in self.task.dfa.accept_states

    def evaluate(
        self,
        strings: Sequence[list[str]],
    ) -> dict[str, float]:
        """Evaluate string labels, state tracking, and transition accuracy."""
        if not strings:
            return {
                "string_accuracy": math.nan,
                "accept_accuracy": math.nan,
                "reject_accuracy": math.nan,
                "state_tracking_accuracy": math.nan,
                "final_state_accuracy": math.nan,
                "transition_accuracy_teacher_forced": math.nan,
                "transition_accuracy_autonomous": math.nan,
                "valid_next_symbol_accuracy": math.nan,
                "extracted_transition_table_accuracy": math.nan,
            }

        string_correct = 0
        accept_correct = 0
        accept_total = 0
        reject_correct = 0
        reject_total = 0
        tracked_correct = 0
        tracked_total = 0
        final_correct = 0
        teacher_correct = 0
        teacher_total = 0
        autonomous_correct = 0
        autonomous_total = 0
        valid_next_scores: list[float] = []

        for symbols in strings:
            true_trace_names = self.task.dfa.trace(symbols)
            true_trace = [self.state_to_idx[state] for state in true_trace_names]
            pred_trace = self.predict_trace(symbols)
            predicted_accept = self.idx_to_state[pred_trace[-1]] in self.task.dfa.accept_states
            true_accept = true_trace_names[-1] in self.task.dfa.accept_states
            string_correct += int(predicted_accept == true_accept)
            if true_accept:
                accept_total += 1
                accept_correct += int(predicted_accept == true_accept)
            else:
                reject_total += 1
                reject_correct += int(predicted_accept == true_accept)
            final_correct += int(pred_trace[-1] == true_trace[-1])
            for pred_idx, true_idx in zip(pred_trace[1:], true_trace[1:]):
                tracked_correct += int(pred_idx == true_idx)
                tracked_total += 1
            for step_idx, symbol in enumerate(symbols):
                input_idx = self.symbol_to_idx[symbol]
                true_current_idx = true_trace[step_idx]
                true_next_idx = true_trace[step_idx + 1]
                teacher_pred = self._transition_step(true_current_idx, input_idx)
                teacher_correct += int(teacher_pred == true_next_idx)
                teacher_total += 1
                autonomous_correct += int(pred_trace[step_idx + 1] == true_next_idx)
                autonomous_total += 1
                valid_next_scores.append(
                    self._valid_next_symbol_score(true_current_idx, true_next_idx)
                )

        extracted = self.extracted_transition_table_accuracy()
        return {
            "string_accuracy": string_correct / len(strings),
            "accept_accuracy": (
                accept_correct / accept_total if accept_total else math.nan
            ),
            "reject_accuracy": (
                reject_correct / reject_total if reject_total else math.nan
            ),
            "state_tracking_accuracy": (
                tracked_correct / tracked_total if tracked_total else math.nan
            ),
            "final_state_accuracy": final_correct / len(strings),
            "transition_accuracy_teacher_forced": (
                teacher_correct / teacher_total if teacher_total else math.nan
            ),
            "transition_accuracy_autonomous": (
                autonomous_correct / autonomous_total if autonomous_total else math.nan
            ),
            "valid_next_symbol_accuracy": (
                float(np.mean(valid_next_scores)) if valid_next_scores else math.nan
            ),
            "extracted_transition_table_accuracy": extracted,
        }

    def accuracy_by_length(
        self,
        strings: Sequence[list[str]],
    ) -> dict[int, float]:
        """Return accept/reject accuracy for each string length."""
        groups: dict[int, list[float]] = {}
        for symbols in strings:
            length = len(symbols)
            groups.setdefault(length, [])
            groups[length].append(
                float(self.predict_accept(symbols) == self.task.dfa.accepts(symbols))
            )
        return {length: float(np.mean(values)) for length, values in groups.items()}

    def topk_masked_transition_accuracy(self, k_values: Sequence[int]) -> dict[int, float]:
        """Evaluate one-step transition recovery using top-k visible coordinates."""
        masked = MaskedNearestRegister(self.state_codebook)
        results: dict[int, float] = {}
        for k in k_values:
            correct = 0
            total = 0
            for state, symbol, target in self.task.dfa.transition_items():
                state_idx = self.state_to_idx[state]
                input_idx = self.symbol_to_idx[symbol]
                target_idx = self.state_to_idx[target]
                phi = self.encoder.encode(
                    self.state_codebook[state_idx],
                    self.input_codebook[input_idx],
                    state_idx,
                    input_idx,
                )
                raw = self.writer.raw(phi)
                proposal = sign(raw)
                mask = np.zeros(self.state_dim, dtype=bool)
                n_write = min(max(0, k), self.state_dim)
                if n_write > 0:
                    mask[np.argsort(np.abs(raw))[-n_write:]] = True
                pred_idx, _ = masked.cleanup_masked(proposal, mask)
                correct += int(pred_idx == target_idx)
                total += 1
            results[int(k)] = correct / total if total else math.nan
        return results

    def extracted_transition_table_accuracy(self) -> float:
        """Evaluate all DFA transitions under teacher-forced current state."""
        correct = 0
        total = 0
        for state, symbol, target in self.task.dfa.transition_items():
            state_idx = self.state_to_idx[state]
            input_idx = self.symbol_to_idx[symbol]
            target_idx = self.state_to_idx[target]
            correct += int(self._transition_step(state_idx, input_idx) == target_idx)
            total += 1
        return correct / total if total else math.nan

    def _trace_examples(
        self,
        strings: Sequence[list[str]],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        examples: list[tuple[np.ndarray, np.ndarray]] = []
        for symbols in strings:
            state = self.task.dfa.start_state
            for symbol in symbols:
                next_state = self.task.dfa.step(state, symbol)
                state_idx = self.state_to_idx[state]
                input_idx = self.symbol_to_idx[symbol]
                target_idx = self.state_to_idx[next_state]
                phi = self.encoder.encode(
                    self.state_codebook[state_idx],
                    self.input_codebook[input_idx],
                    state_idx,
                    input_idx,
                )
                examples.append((phi, self.state_codebook[target_idx]))
                state = next_state
        return examples

    def _transition_step(self, state_idx: int, input_idx: int) -> int:
        proposal = self.writer.propose(
            self.state_codebook[state_idx],
            self.input_codebook[input_idx],
            self.encoder,
            state_idx,
            input_idx,
        )
        pred_idx, _ = self.register.cleanup(proposal)
        return pred_idx

    def _valid_next_symbol_score(self, current_state_idx: int, true_next_idx: int) -> float:
        valid_symbols = {
            symbol
            for symbol in self.task.alphabet
            if self.state_to_idx[
                self.task.dfa.step(self.idx_to_state[current_state_idx], symbol)
            ]
            == true_next_idx
        }
        predicted_symbols = set()
        for symbol in self.task.alphabet:
            input_idx = self.symbol_to_idx[symbol]
            if self._transition_step(current_state_idx, input_idx) == true_next_idx:
                predicted_symbols.add(symbol)
        if not valid_symbols and not predicted_symbols:
            return 1.0
        union = valid_symbols | predicted_symbols
        return len(valid_symbols & predicted_symbols) / len(union) if union else 0.0


def theoretical_topk_bound(num_states: int, delta: float) -> int:
    """Return ceil(log2((m - 1) / delta)) for sparse basin targeting."""
    if num_states <= 1:
        return 1
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    return int(math.ceil(math.log2((num_states - 1) / delta)))


def first_k_reaching(values: dict[int, float], threshold: float) -> int | None:
    """Return the first k whose accuracy reaches threshold."""
    for k in sorted(values):
        if values[k] >= threshold:
            return k
    return None


def make_length_splits(
    task: GrammarTask,
    train_max_len: int,
    test_max_len: int,
    train_num_strings: int,
    test_num_strings: int,
    rng: np.random.Generator,
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Create train, in-distribution, and longer test string sets."""
    train = [
        symbols
        for symbols, _label in task.sample_balanced_dataset(
            train_num_strings,
            (0, train_max_len),
            rng,
        )
    ]
    in_dist = [
        symbols
        for symbols, _label in task.sample_balanced_dataset(
            test_num_strings,
            (0, train_max_len),
            rng,
        )
    ]
    long = [
        symbols
        for symbols, _label in task.sample_balanced_dataset(
            test_num_strings,
            (train_max_len + 1, test_max_len),
            rng,
        )
    ]
    return train, in_dist, long


def transition_coverage_fraction(task: GrammarTask, strings: Sequence[list[str]]) -> float:
    """Return the fraction of reachable DFA transitions observed in strings."""
    return transition_coverage(task.dfa, strings)[1]


def random_untrained_state_baseline(
    task: GrammarTask,
    strings: Sequence[list[str]],
    rng: np.random.Generator,
) -> float:
    """Simple random final-state baseline for accept/reject classification."""
    if not strings:
        return math.nan
    accept_indices = {
        idx for idx, state in enumerate(task.dfa.states) if state in task.dfa.accept_states
    }
    correct = 0
    for symbols in strings:
        pred_idx = int(rng.integers(0, task.dfa.num_states))
        pred_accept = pred_idx in accept_indices
        correct += int(pred_accept == task.dfa.accepts(symbols))
    return correct / len(strings)


def majority_label_baseline(
    train_strings: Sequence[list[str]],
    test_strings: Sequence[list[str]],
    task: GrammarTask,
) -> float:
    """Majority-label accept/reject baseline."""
    if not train_strings or not test_strings:
        return math.nan
    train_accept_rate = float(np.mean([task.dfa.accepts(s) for s in train_strings]))
    predict_accept = train_accept_rate >= 0.5
    return float(
        np.mean([predict_accept == task.dfa.accepts(symbols) for symbols in test_strings])
    )
