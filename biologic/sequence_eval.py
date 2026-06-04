"""Autonomous sequence evaluation for learned BioLogic grammar transitions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    _transition_table_cache: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _oracle_transition_table_cache: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _encoded_string_cache: dict[int, dict[int, np.ndarray]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

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
        deduplicate_transitions: bool = False,
    ) -> None:
        """Train from oracle DFA traces through local transition updates."""
        examples = self._trace_examples(
            train_strings,
            deduplicate_transitions=deduplicate_transitions,
        )
        for _ in range(epochs):
            self.learner.train_epoch(examples, rng=self.rng, shuffle=shuffle)
        self._transition_table_cache = None

    def predict_trace(self, symbols: list[str]) -> list[int]:
        """Run the learned transition model autonomously and return state indices."""
        transition_table = self._transition_table()
        current_idx = self.state_to_idx[self.task.dfa.start_state]
        trace = [current_idx]
        for symbol in symbols:
            input_idx = self.symbol_to_idx[symbol]
            current_idx = int(transition_table[current_idx, input_idx])
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

        learned_table = self._transition_table()
        oracle_table = self._oracle_transition_table()
        start_idx = self.state_to_idx[self.task.dfa.start_state]
        accept_mask = self._accept_mask()
        valid_next_score_table = self._valid_next_score_table(learned_table)

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
        valid_next_score_sum = 0.0
        valid_next_total = 0

        for encoded in self._encoded_groups(strings).values():
            count, length = encoded.shape
            true_idx = np.full(count, start_idx, dtype=int)
            pred_idx = np.full(count, start_idx, dtype=int)
            for column in range(length):
                input_idx = encoded[:, column]
                true_next_idx = oracle_table[true_idx, input_idx]
                pred_next_idx = learned_table[pred_idx, input_idx]
                teacher_pred = learned_table[true_idx, input_idx]

                tracked_correct += int(np.count_nonzero(pred_next_idx == true_next_idx))
                teacher_correct += int(np.count_nonzero(teacher_pred == true_next_idx))
                autonomous_correct += int(np.count_nonzero(pred_next_idx == true_next_idx))
                tracked_total += count
                teacher_total += count
                autonomous_total += count
                valid_next_score_sum += float(
                    np.sum(valid_next_score_table[true_idx, true_next_idx])
                )
                valid_next_total += count

                true_idx = true_next_idx
                pred_idx = pred_next_idx

            predicted_accept = accept_mask[pred_idx]
            true_accept = accept_mask[true_idx]
            correct = predicted_accept == true_accept
            string_correct += int(np.count_nonzero(correct))
            accept_selector = true_accept
            reject_selector = ~true_accept
            accept_total += int(np.count_nonzero(accept_selector))
            reject_total += int(np.count_nonzero(reject_selector))
            accept_correct += int(np.count_nonzero(correct & accept_selector))
            reject_correct += int(np.count_nonzero(correct & reject_selector))
            final_correct += int(np.count_nonzero(pred_idx == true_idx))

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
                valid_next_score_sum / valid_next_total
                if valid_next_total
                else math.nan
            ),
            "extracted_transition_table_accuracy": extracted,
        }

    def accuracy_by_length(
        self,
        strings: Sequence[list[str]],
    ) -> dict[int, float]:
        """Return accept/reject accuracy for each string length."""
        start_idx = self.state_to_idx[self.task.dfa.start_state]
        learned_table = self._transition_table()
        oracle_table = self._oracle_transition_table()
        accept_mask = self._accept_mask()
        results: dict[int, float] = {}
        for length, encoded in self._encoded_groups(strings).items():
            count = encoded.shape[0]
            pred_idx = np.full(count, start_idx, dtype=int)
            true_idx = np.full(count, start_idx, dtype=int)
            for column in range(length):
                input_idx = encoded[:, column]
                pred_idx = learned_table[pred_idx, input_idx]
                true_idx = oracle_table[true_idx, input_idx]
            results[length] = float(np.mean(accept_mask[pred_idx] == accept_mask[true_idx]))
        return results

    def topk_masked_transition_accuracy(self, k_values: Sequence[int]) -> dict[int, float]:
        """Evaluate one-step transition recovery using top-k visible coordinates."""
        sorted_k = sorted({int(k) for k in k_values})
        correct_by_k = {k: 0 for k in sorted_k}
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
            order = np.argsort(np.abs(raw))[::-1]
            scores = np.zeros(self.task.dfa.num_states, dtype=float)
            visible_count = 0
            for k in sorted_k:
                n_write = min(max(0, k), self.state_dim)
                if n_write > visible_count:
                    indices = order[visible_count:n_write]
                    scores += self.state_codebook[:, indices] @ proposal[indices]
                    visible_count = n_write
                correct_by_k[k] += int(int(np.argmax(scores)) == target_idx)
            total += 1
        results: dict[int, float] = {}
        for k in sorted_k:
            results[k] = correct_by_k[k] / total if total else math.nan
        return results

    def extracted_transition_table_accuracy(self) -> float:
        """Evaluate all DFA transitions under teacher-forced current state."""
        learned_table = self._transition_table()
        oracle_table = self._oracle_transition_table()
        if learned_table.size == 0:
            return math.nan
        return float(np.mean(learned_table == oracle_table))

    def _trace_examples(
        self,
        strings: Sequence[list[str]],
        deduplicate_transitions: bool = True,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        examples: list[tuple[np.ndarray, np.ndarray]] = []
        seen_pairs: set[tuple[int, int]] = set()
        for symbols in strings:
            state = self.task.dfa.start_state
            for symbol in symbols:
                next_state = self.task.dfa.step(state, symbol)
                state_idx = self.state_to_idx[state]
                input_idx = self.symbol_to_idx[symbol]
                if deduplicate_transitions:
                    pair = (state_idx, input_idx)
                    if pair in seen_pairs:
                        state = next_state
                        continue
                    seen_pairs.add(pair)
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
        return int(self._transition_table()[state_idx, input_idx])

    def _transition_step_uncached(self, state_idx: int, input_idx: int) -> int:
        proposal = self.writer.propose(
            self.state_codebook[state_idx],
            self.input_codebook[input_idx],
            self.encoder,
            state_idx,
            input_idx,
        )
        pred_idx, _ = self.register.cleanup(proposal)
        return pred_idx

    def _transition_table(self) -> np.ndarray:
        if self._transition_table_cache is None:
            table = np.empty(
                (self.task.dfa.num_states, len(self.task.alphabet)),
                dtype=int,
            )
            for state_idx in range(self.task.dfa.num_states):
                for input_idx in range(len(self.task.alphabet)):
                    table[state_idx, input_idx] = self._transition_step_uncached(
                        state_idx,
                        input_idx,
                    )
            self._transition_table_cache = table
        return self._transition_table_cache

    def _oracle_transition_table(self) -> np.ndarray:
        if self._oracle_transition_table_cache is None:
            table = np.empty(
                (self.task.dfa.num_states, len(self.task.alphabet)),
                dtype=int,
            )
            for state_idx, state in self.idx_to_state.items():
                for symbol, input_idx in self.symbol_to_idx.items():
                    next_state = self.task.dfa.step(state, symbol)
                    table[state_idx, input_idx] = self.state_to_idx[next_state]
            self._oracle_transition_table_cache = table
        return self._oracle_transition_table_cache

    def _accept_indices(self) -> set[int]:
        return {
            idx
            for idx, state in self.idx_to_state.items()
            if state in self.task.dfa.accept_states
        }

    def _accept_mask(self) -> np.ndarray:
        mask = np.zeros(self.task.dfa.num_states, dtype=bool)
        for idx in self._accept_indices():
            mask[idx] = True
        return mask

    def _encoded_groups(
        self,
        strings: Sequence[list[str]],
    ) -> dict[int, np.ndarray]:
        cache_key = id(strings)
        cached = self._encoded_string_cache.get(cache_key)
        if cached is not None:
            return cached
        grouped: dict[int, list[list[int]]] = {}
        for symbols in strings:
            length = len(symbols)
            grouped.setdefault(length, []).append(
                [self.symbol_to_idx[symbol] for symbol in symbols]
            )
        encoded = {
            length: np.asarray(values, dtype=int)
            for length, values in grouped.items()
        }
        self._encoded_string_cache[cache_key] = encoded
        return encoded

    def _valid_next_score_table(self, learned_table: np.ndarray) -> np.ndarray:
        oracle_table = self._oracle_transition_table()
        num_states = self.task.dfa.num_states
        scores = np.zeros((num_states, num_states), dtype=float)
        for state_idx in range(num_states):
            for next_idx in range(num_states):
                valid = oracle_table[state_idx] == next_idx
                predicted = learned_table[state_idx] == next_idx
                union = valid | predicted
                if not np.any(union):
                    scores[state_idx, next_idx] = 1.0
                else:
                    scores[state_idx, next_idx] = (
                        np.count_nonzero(valid & predicted) / np.count_nonzero(union)
                    )
        return scores

    def _rollout_final_idx(
        self,
        transition_table: np.ndarray,
        start_idx: int,
        symbols: Sequence[str],
    ) -> int:
        current_idx = start_idx
        for symbol in symbols:
            current_idx = int(transition_table[current_idx, self.symbol_to_idx[symbol]])
        return current_idx

    def _valid_next_symbol_score_from_table(
        self,
        transition_table: np.ndarray,
        current_state_idx: int,
        true_next_idx: int,
    ) -> float:
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
            if int(transition_table[current_state_idx, input_idx]) == true_next_idx:
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
