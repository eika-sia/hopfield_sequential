"""Reward-modulated transition learning for BioLogic state machines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from biologic.encodings import random_bipolar, random_input_codebook, random_state_codebook, sign
from biologic.grammars import GrammarTask, transition_coverage
from biologic.learning import FeatureMode, StateInputPairEncoder, UnwrittenMode
from biologic.register import HopfieldRegister, NearestAttractorRegister

RLCondition = Literal[
    "supervised_baseline",
    "state_shaped_rl",
    "terminal_rl",
    "sparse_terminal_rl",
]
CleanupType = Literal["nearest", "hopfield"]
OutputMode = Literal["dense", "sparse_topk"]


@dataclass(frozen=True)
class RLEpisodeStats:
    """Diagnostics collected from one training episode."""

    episode_return: float
    mean_td_error: float
    mean_trace_norm: float


@dataclass(frozen=True)
class RLEvaluation:
    """Evaluation metrics for an RL transition learner."""

    string_accuracy: float
    state_tracking_accuracy: float
    transition_accuracy: float
    seen_transition_accuracy: float
    unseen_transition_accuracy: float
    accept_accuracy: float
    reject_accuracy: float


@dataclass(frozen=True)
class TransitionEvalItem:
    """One hidden-DFA transition evaluation row."""

    state_id: int
    symbol: str
    seen_in_training: bool
    true_next_state: int
    pred_next_state: int
    correct: bool


def update_transition_writer_rl(
    weights: np.ndarray,
    executed_state_vec: np.ndarray,
    trace: np.ndarray,
    delta: float,
    eta: float,
    mask: np.ndarray | None = None,
) -> None:
    """Reward-modulated transition update.

    RL condition: reinforce the executed next-state basin.
    Do NOT use the true DFA next-state vector here.
    """
    if mask is None:
        weights += eta * delta * np.outer(executed_state_vec, trace)
        return
    visible = np.asarray(mask, dtype=bool)
    weights[visible, :] += (
        eta * delta * executed_state_vec[visible, np.newaxis] * trace[np.newaxis, :]
    )


def update_value(
    value_weights: np.ndarray,
    state_vec: np.ndarray,
    delta: float,
    alpha: float,
) -> None:
    """Linear TD value-function update."""
    value_weights += alpha * delta * state_vec / state_vec.size


@dataclass
class RLTransitionAgent:
    """BioLogic transition writer trained by scalar reward."""

    task: GrammarTask
    state_dim: int
    input_dim: int
    rng: np.random.Generator
    feature_type: FeatureMode = "exact_pair"
    hidden_dim: int | None = None
    cleanup: CleanupType = "nearest"
    output_mode: OutputMode = "dense"
    write_fraction: float = 1.0
    unwritten_mode: UnwrittenMode = "random_noise"
    eta: float = 0.005
    alpha: float = 0.01
    gamma: float = 0.95
    lambda_trace: float = 0.8
    clf_lr: float = 0.05

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
        resolved_hidden_dim = (
            None if self.feature_type == "exact_pair" else self.hidden_dim
        )
        self.encoder = StateInputPairEncoder(
            num_states=self.task.dfa.num_states,
            num_inputs=len(self.task.alphabet),
            mode=self.feature_type,
            hidden_dim=resolved_hidden_dim,
            state_dim=self.state_dim,
            input_dim=self.input_dim,
            rng=self.rng,
        )
        self.W = np.zeros((self.state_dim, self.encoder.feature_dim), dtype=float)
        self.value_weights = np.zeros(self.state_dim, dtype=float)
        self.beta = np.zeros(self.state_dim, dtype=float)
        if self.cleanup == "nearest":
            self.register = NearestAttractorRegister(self.state_codebook)
        elif self.cleanup == "hopfield":
            self.register = HopfieldRegister.from_codebook(self.state_codebook)
        else:
            raise ValueError(f"unsupported cleanup: {self.cleanup}")

    @property
    def feature_dim(self) -> int:
        """Return context-feature dimension."""
        return self.encoder.feature_dim

    def context_feature(self, state_idx: int, input_idx: int) -> np.ndarray:
        """Encode the executed internal state and current symbol."""
        return self.encoder.encode(
            self.state_codebook[state_idx],
            self.input_codebook[input_idx],
            state_idx,
            input_idx,
        )

    def raw(self, phi: np.ndarray) -> np.ndarray:
        """Return unthresholded transition proposal scores."""
        active = np.flatnonzero(phi)
        if 0 < active.size <= 16:
            return self.W[:, active] @ phi[active]
        return self.W @ phi

    def propose(
        self,
        phi: np.ndarray,
        current_state_idx: int,
        noise_std: float,
    ) -> tuple[int, np.ndarray, np.ndarray | None]:
        """Propose and clean up the next internal state."""
        scores = self.raw(phi)
        if noise_std > 0.0:
            scores = scores + noise_std * self.rng.normal(size=self.state_dim)
        proposal = sign(scores)
        mask: np.ndarray | None = None
        if self.output_mode == "sparse_topk":
            n_write = max(1, int(round(self.write_fraction * self.state_dim)))
            n_write = min(n_write, self.state_dim)
            mask = np.zeros(self.state_dim, dtype=bool)
            mask[np.argsort(np.abs(scores))[-n_write:]] = True
            if self.unwritten_mode == "keep_current":
                sparse_proposal = self.state_codebook[current_state_idx].copy()
            else:
                sparse_proposal = random_bipolar(self.state_dim, self.rng)
            sparse_proposal[mask] = proposal[mask]
            proposal = sparse_proposal
        state_idx, state_vec = self.register.cleanup(proposal)
        return state_idx, state_vec, mask

    def predict_accept_idx(self, state_idx: int) -> bool:
        """Predict accept/reject from the current internal state."""
        state_vec = self.state_codebook[state_idx]
        logit = float(self.beta @ state_vec / self.state_dim)
        return logit >= 0.0

    def train_episode(
        self,
        symbols: Sequence[str],
        condition: RLCondition,
        noise_std: float,
    ) -> RLEpisodeStats:
        """Train one episode without transition targets for RL conditions."""
        if condition == "supervised_baseline":
            return self._train_supervised_episode(symbols)
        current_state = self.task.dfa.start_state
        current_idx = self.state_to_idx[self.task.dfa.start_state]
        current_vec = self.state_codebook[current_idx]
        trace = np.zeros(self.feature_dim, dtype=float)
        td_errors: list[float] = []
        trace_norms: list[float] = []
        total_return = 0.0
        last_mask: np.ndarray | None = None

        for symbol in symbols:
            input_idx = self.symbol_to_idx[symbol]
            phi = self.context_feature(current_idx, input_idx)
            trace = self.lambda_trace * trace + phi
            next_idx, next_vec, mask = self.propose(phi, current_idx, noise_std)
            last_mask = mask
            true_next_state = self.task.dfa.step(current_state, symbol)
            true_next_idx = self.state_to_idx[true_next_state]
            reward = 0.0
            if condition == "state_shaped_rl":
                reward = 1.0 if next_idx == true_next_idx else -1.0
            delta = reward + self.gamma * self.value(next_vec) - self.value(current_vec)
            update_transition_writer_rl(
                self.W,
                next_vec,
                trace,
                delta,
                self.eta,
                mask=mask if condition == "sparse_terminal_rl" else None,
            )
            update_value(self.value_weights, current_vec, delta, self.alpha)
            td_errors.append(abs(delta))
            trace_norms.append(float(np.linalg.norm(trace)))
            total_return += reward
            current_state = true_next_state
            current_idx = next_idx
            current_vec = next_vec

        true_accept = self.task.dfa.accepts(list(symbols))
        pred_accept = self.predict_accept_idx(current_idx)
        terminal_reward = 1.0 if pred_accept == true_accept else -1.0
        delta_terminal = terminal_reward - self.value(current_vec)
        if trace.size:
            update_transition_writer_rl(
                self.W,
                current_vec,
                trace,
                delta_terminal,
                self.eta,
                mask=last_mask if condition == "sparse_terminal_rl" else None,
            )
        update_value(self.value_weights, current_vec, delta_terminal, self.alpha)
        self.update_readout(current_idx, true_accept)
        td_errors.append(abs(delta_terminal))
        trace_norms.append(float(np.linalg.norm(trace)))
        total_return += terminal_reward
        return RLEpisodeStats(
            episode_return=total_return,
            mean_td_error=float(np.mean(td_errors)) if td_errors else 0.0,
            mean_trace_norm=float(np.mean(trace_norms)) if trace_norms else 0.0,
        )

    def value(self, state_vec: np.ndarray) -> float:
        """Return linear value estimate."""
        return float(self.value_weights @ state_vec / self.state_dim)

    def update_readout(self, state_idx: int, true_accept: bool) -> None:
        """Supervised accept/reject readout update from final label."""
        target = 1.0 if true_accept else -1.0
        pred = 1.0 if self.predict_accept_idx(state_idx) else -1.0
        error = target - pred
        self.beta += self.clf_lr * error * self.state_codebook[state_idx] / self.state_dim

    def _train_supervised_episode(self, symbols: Sequence[str]) -> RLEpisodeStats:
        state = self.task.dfa.start_state
        for symbol in symbols:
            input_idx = self.symbol_to_idx[symbol]
            state_idx = self.state_to_idx[state]
            next_state = self.task.dfa.step(state, symbol)
            next_idx = self.state_to_idx[next_state]
            phi = self.context_feature(state_idx, input_idx)
            prediction = sign(self.raw(phi)).astype(float)
            target = self.state_codebook[next_idx].astype(float)
            error = target - prediction
            active = np.flatnonzero(phi)
            if 0 < active.size <= 16:
                self.W[:, active] += self.eta * error[:, np.newaxis] * phi[active]
            else:
                self.W += self.eta * np.outer(error, phi)
            state = next_state
        final_idx = self.rollout_final_idx(symbols)
        true_accept = self.task.dfa.accepts(list(symbols))
        pred_accept = self.predict_accept_idx(final_idx)
        self.update_readout(final_idx, true_accept)
        reward = 1.0 if pred_accept == true_accept else -1.0
        return RLEpisodeStats(reward, 0.0, 0.0)

    def transition_table(self, noise_std: float = 0.0) -> np.ndarray:
        """Return autonomous transition table over internal states."""
        table = np.empty(
            (self.task.dfa.num_states, len(self.task.alphabet)),
            dtype=int,
        )
        for state_idx in range(self.task.dfa.num_states):
            for input_idx in range(len(self.task.alphabet)):
                phi = self.context_feature(state_idx, input_idx)
                next_idx, _next_vec, _mask = self.propose(phi, state_idx, noise_std)
                table[state_idx, input_idx] = next_idx
        return table

    def oracle_transition_table(self) -> np.ndarray:
        """Return the hidden DFA transition table for evaluation only."""
        table = np.empty(
            (self.task.dfa.num_states, len(self.task.alphabet)),
            dtype=int,
        )
        for state_idx, state in self.idx_to_state.items():
            for symbol, input_idx in self.symbol_to_idx.items():
                next_state = self.task.dfa.step(state, symbol)
                table[state_idx, input_idx] = self.state_to_idx[next_state]
        return table

    def rollout_final_idx(self, symbols: Sequence[str]) -> int:
        """Roll out autonomous transitions and return final internal state id."""
        table = self.transition_table(noise_std=0.0)
        current_idx = self.state_to_idx[self.task.dfa.start_state]
        for symbol in symbols:
            current_idx = int(table[current_idx, self.symbol_to_idx[symbol]])
        return current_idx

    def evaluate(
        self,
        strings: Sequence[list[str]],
        train_strings: Sequence[list[str]],
    ) -> RLEvaluation:
        """Evaluate autonomous string behavior and hidden transition agreement."""
        if not strings:
            return RLEvaluation(*(math.nan for _ in range(7)))
        learned = self.transition_table(noise_std=0.0)
        oracle = self.oracle_transition_table()
        start_idx = self.state_to_idx[self.task.dfa.start_state]
        accept_states = {
            self.state_to_idx[state] for state in self.task.dfa.accept_states
        }
        string_correct = 0
        accept_correct = 0
        accept_total = 0
        reject_correct = 0
        reject_total = 0
        tracking_correct = 0
        tracking_total = 0
        for symbols in strings:
            true_idx = start_idx
            pred_idx = start_idx
            for symbol in symbols:
                input_idx = self.symbol_to_idx[symbol]
                true_idx = int(oracle[true_idx, input_idx])
                pred_idx = int(learned[pred_idx, input_idx])
                tracking_correct += int(pred_idx == true_idx)
                tracking_total += 1
            pred_accept = self.predict_accept_idx(pred_idx)
            true_accept = true_idx in accept_states
            correct = pred_accept == true_accept
            string_correct += int(correct)
            if true_accept:
                accept_total += 1
                accept_correct += int(correct)
            else:
                reject_total += 1
                reject_correct += int(correct)
        transition_accuracy = float(np.mean(learned == oracle))
        seen, unseen = self.seen_unseen_transition_accuracy(train_strings)
        return RLEvaluation(
            string_accuracy=string_correct / len(strings),
            state_tracking_accuracy=(
                tracking_correct / tracking_total if tracking_total else math.nan
            ),
            transition_accuracy=transition_accuracy,
            seen_transition_accuracy=seen,
            unseen_transition_accuracy=unseen,
            accept_accuracy=(accept_correct / accept_total if accept_total else math.nan),
            reject_accuracy=(reject_correct / reject_total if reject_total else math.nan),
        )

    def seen_unseen_transition_accuracy(
        self,
        train_strings: Sequence[list[str]],
    ) -> tuple[float, float]:
        """Evaluate hidden transition agreement split by train coverage."""
        covered, _coverage = transition_coverage(self.task.dfa, train_strings)
        learned = self.transition_table(noise_std=0.0)
        seen_correct = 0
        seen_total = 0
        unseen_correct = 0
        unseen_total = 0
        for state, symbol, target in self.task.dfa.transition_items():
            state_idx = self.state_to_idx[state]
            input_idx = self.symbol_to_idx[symbol]
            target_idx = self.state_to_idx[target]
            pred_idx = int(learned[state_idx, input_idx])
            if (state, symbol) in covered:
                seen_total += 1
                seen_correct += int(pred_idx == target_idx)
            else:
                unseen_total += 1
                unseen_correct += int(pred_idx == target_idx)
        return (
            seen_correct / seen_total if seen_total else math.nan,
            unseen_correct / unseen_total if unseen_total else math.nan,
        )

    def transition_eval_items(
        self,
        train_strings: Sequence[list[str]],
    ) -> list[TransitionEvalItem]:
        """Return one row per hidden DFA transition for diagnostics."""
        covered, _coverage = transition_coverage(self.task.dfa, train_strings)
        learned = self.transition_table(noise_std=0.0)
        rows: list[TransitionEvalItem] = []
        for state, symbol, target in self.task.dfa.transition_items():
            state_idx = self.state_to_idx[state]
            input_idx = self.symbol_to_idx[symbol]
            true_idx = self.state_to_idx[target]
            pred_idx = int(learned[state_idx, input_idx])
            rows.append(
                TransitionEvalItem(
                    state_id=state_idx,
                    symbol=symbol,
                    seen_in_training=(state, symbol) in covered,
                    true_next_state=true_idx,
                    pred_next_state=pred_idx,
                    correct=pred_idx == true_idx,
                )
            )
        return rows
