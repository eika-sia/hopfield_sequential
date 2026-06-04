"""Local associative learning for BioLogic transition maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from biologic.encodings import random_bipolar, sign

FeatureMode = Literal[
    "exact_pair",
    "noisy_exact_pair",
    "hashed_pair",
    "random_conjunctive",
]
UpdateRule = Literal["hebbian", "delta", "perceptron", "three_factor"]
OutputMode = Literal["dense", "sparse_topk", "sparse_random_mask"]
UnwrittenMode = Literal["random_noise", "keep_current"]


@dataclass
class StateInputPairEncoder:
    """Encode a current-state and input identity as conjunctive features."""

    num_states: int
    num_inputs: int
    mode: FeatureMode = "exact_pair"
    rng: np.random.Generator | None = None
    hidden_dim: int | None = None
    state_dim: int | None = None
    input_dim: int | None = None
    threshold: float = 0.0
    pair_noise: float = 0.0
    num_active_features: int = 1

    def __post_init__(self) -> None:
        if self.num_states <= 0:
            raise ValueError("num_states must be positive")
        if self.num_inputs <= 0:
            raise ValueError("num_inputs must be positive")
        if self.num_active_features <= 0:
            raise ValueError("num_active_features must be positive")
        if not 0.0 <= self.pair_noise <= 1.0:
            raise ValueError("pair_noise must be in [0, 1]")
        if self.rng is None:
            self.rng = np.random.default_rng(0)

        self._num_pairs = self.num_states * self.num_inputs
        self._feature_dim = self._resolve_feature_dim()
        self._hash_table: np.ndarray | None = None
        self._projection: np.ndarray | None = None

        if self.mode == "hashed_pair":
            self._hash_table = self.rng.integers(
                0,
                self._feature_dim,
                size=(self._num_pairs, self.num_active_features),
            )
        elif self.mode == "random_conjunctive":
            if self.state_dim is None or self.input_dim is None:
                raise ValueError(
                    "state_dim and input_dim are required for random_conjunctive"
                )
            concat_dim = self.state_dim + self.input_dim
            self._projection = self.rng.normal(
                loc=0.0,
                scale=1.0 / np.sqrt(concat_dim),
                size=(self._feature_dim, concat_dim),
            )

    @property
    def feature_dim(self) -> int:
        """Return the number of feature units."""
        return self._feature_dim

    @property
    def num_pairs(self) -> int:
        """Return the number of possible state-input pairs."""
        return self._num_pairs

    def _resolve_feature_dim(self) -> int:
        if self.mode in {"exact_pair", "noisy_exact_pair"}:
            return self.num_states * self.num_inputs
        if self.hidden_dim is None or self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive for this feature mode")
        return self.hidden_dim

    def _pair_id(self, state_idx: int, input_idx: int) -> int:
        if not 0 <= state_idx < self.num_states:
            raise ValueError("state_idx out of range")
        if not 0 <= input_idx < self.num_inputs:
            raise ValueError("input_idx out of range")
        return state_idx * self.num_inputs + input_idx

    def encode_indices(self, state_idx: int, input_idx: int) -> np.ndarray:
        """Encode a state-input identity pair as a binary feature vector."""
        pair_id = self._pair_id(state_idx, input_idx)
        if self.mode in {"exact_pair", "noisy_exact_pair"}:
            return self._encode_exact(pair_id)
        if self.mode == "hashed_pair":
            return self._encode_hashed(pair_id)
        raise ValueError("random_conjunctive requires encode_vectors")

    def encode_vectors(self, state_vec: np.ndarray, input_vec: np.ndarray) -> np.ndarray:
        """Encode state and input vectors using random conjunctive features."""
        if self.mode != "random_conjunctive":
            raise ValueError("encode_vectors is only defined for random_conjunctive")
        if self._projection is None:
            raise ValueError("random projection was not initialized")
        state = np.asarray(state_vec, dtype=float)
        inp = np.asarray(input_vec, dtype=float)
        if self.state_dim is not None and state.shape != (self.state_dim,):
            raise ValueError("state_vec has wrong shape")
        if self.input_dim is not None and inp.shape != (self.input_dim,):
            raise ValueError("input_vec has wrong shape")
        activations = self._projection @ np.concatenate([state, inp])
        phi = (activations >= self.threshold).astype(float)
        if not np.any(phi):
            phi[int(np.argmax(activations))] = 1.0
        return phi

    def encode(
        self,
        state_vec: np.ndarray,
        input_vec: np.ndarray,
        state_idx: int | None = None,
        input_idx: int | None = None,
    ) -> np.ndarray:
        """Encode by indices when available, otherwise by vectors."""
        if self.mode == "random_conjunctive":
            return self.encode_vectors(state_vec, input_vec)
        if state_idx is None or input_idx is None:
            raise ValueError("state_idx and input_idx are required for this mode")
        return self.encode_indices(state_idx, input_idx)

    def active_indices(self, state_idx: int, input_idx: int) -> tuple[int, ...]:
        """Return active feature indices for an identity pair."""
        phi = self.encode_indices(state_idx, input_idx)
        return tuple(int(idx) for idx in np.flatnonzero(phi))

    def collision_count(self) -> int:
        """Return how many state-input pairs collide in feature identity."""
        if self.mode not in {"hashed_pair", "noisy_exact_pair", "exact_pair"}:
            return 0
        seen: set[tuple[int, ...]] = set()
        collisions = 0
        for state_idx in range(self.num_states):
            for input_idx in range(self.num_inputs):
                active = self.active_indices(state_idx, input_idx)
                if active in seen:
                    collisions += 1
                seen.add(active)
        return collisions

    def _encode_exact(self, pair_id: int) -> np.ndarray:
        active_pair = pair_id
        if self.mode == "noisy_exact_pair" and self.rng is not None:
            if self.rng.random() < self.pair_noise:
                candidates = [idx for idx in range(self._num_pairs) if idx != pair_id]
                active_pair = int(self.rng.choice(candidates))
        phi = np.zeros(self._feature_dim, dtype=float)
        phi[active_pair] = 1.0
        if self.mode == "noisy_exact_pair" and self.num_active_features > 1:
            assert self.rng is not None
            extra_count = min(self.num_active_features - 1, self._feature_dim - 1)
            candidates = np.array(
                [idx for idx in range(self._feature_dim) if idx != active_pair],
                dtype=int,
            )
            extras = self.rng.choice(candidates, size=extra_count, replace=False)
            phi[extras] = 1.0
        return phi

    def _encode_hashed(self, pair_id: int) -> np.ndarray:
        if self._hash_table is None:
            raise ValueError("hash table was not initialized")
        phi = np.zeros(self._feature_dim, dtype=float)
        phi[self._hash_table[pair_id]] = 1.0
        return phi


@dataclass
class LearnedAssociativeTransition:
    """Learn pair-feature to state-coordinate transition weights."""

    state_dim: int
    feature_dim: int
    rng: np.random.Generator
    learning_rate: float = 1.0
    weight_decay: float = 0.0
    clip_value: float | None = None
    update_rule: UpdateRule = "delta"
    output_mode: OutputMode = "dense"
    write_fraction: float = 1.0
    unwritten_mode: UnwrittenMode = "random_noise"
    use_bias: bool = True

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not 0.0 <= self.write_fraction <= 1.0:
            raise ValueError("write_fraction must be in [0, 1]")
        if self.update_rule not in {"hebbian", "delta", "perceptron", "three_factor"}:
            raise ValueError("unsupported update_rule")
        if self.output_mode not in {"dense", "sparse_topk", "sparse_random_mask"}:
            raise ValueError("unsupported output_mode")
        if self.unwritten_mode not in {"random_noise", "keep_current"}:
            raise ValueError("unsupported unwritten_mode")
        self.W = np.zeros((self.state_dim, self.feature_dim), dtype=float)
        self.bias = np.zeros(self.state_dim, dtype=float)

    def reset(self) -> None:
        """Reset weights and bias to zero."""
        self.W.fill(0.0)
        self.bias.fill(0.0)

    def copy(self) -> "LearnedAssociativeTransition":
        """Return a deep copy of the learned transition object."""
        copied = LearnedAssociativeTransition(
            state_dim=self.state_dim,
            feature_dim=self.feature_dim,
            rng=self.rng,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            update_rule=self.update_rule,
            output_mode=self.output_mode,
            write_fraction=self.write_fraction,
            unwritten_mode=self.unwritten_mode,
            use_bias=self.use_bias,
        )
        copied.W = self.W.copy()
        copied.bias = self.bias.copy()
        return copied

    def raw(self, phi: np.ndarray) -> np.ndarray:
        """Return the unthresholded coordinate proposal."""
        features = np.asarray(phi, dtype=float)
        if features.shape != (self.feature_dim,):
            raise ValueError("phi has wrong shape")
        active = self._sparse_feature_indices(features)
        raw = self._raw_checked(features, active)
        return raw

    def _sparse_feature_indices(self, features: np.ndarray) -> np.ndarray | None:
        """Return active feature indices when a dense matrix multiply is wasteful."""
        active = np.flatnonzero(features)
        sparse_limit = max(1, min(16, features.size // 2))
        if 0 < active.size <= sparse_limit:
            return active
        return None

    def _raw_checked(
        self,
        features: np.ndarray,
        active: np.ndarray | None,
    ) -> np.ndarray:
        if active is None:
            raw = self.W @ features
        else:
            raw = self.W[:, active] @ features[active]
        if self.use_bias:
            raw = raw + self.bias
        return raw

    def propose_features(
        self,
        phi: np.ndarray,
        current_state_vec: np.ndarray | None = None,
    ) -> np.ndarray:
        """Propose a next-state vector from active pair features."""
        raw = self.raw(phi)
        if self.output_mode == "dense":
            return sign(raw)

        n_write = int(round(self.write_fraction * self.state_dim))
        if self.unwritten_mode == "keep_current" and current_state_vec is not None:
            proposal = np.asarray(current_state_vec, dtype=int).copy()
        else:
            proposal = random_bipolar(self.state_dim, self.rng)
        if n_write <= 0:
            return proposal

        if self.output_mode == "sparse_topk":
            indices = np.argsort(np.abs(raw))[-n_write:]
        else:
            indices = self.rng.choice(self.state_dim, size=n_write, replace=False)
        proposal[indices] = sign(raw[indices])
        return proposal.astype(int)

    def propose(
        self,
        state_vec: np.ndarray,
        input_vec: np.ndarray,
        pair_encoder: StateInputPairEncoder,
        state_idx: int | None = None,
        input_idx: int | None = None,
    ) -> np.ndarray:
        """Encode the interface pair and propose the next state."""
        phi = pair_encoder.encode(state_vec, input_vec, state_idx, input_idx)
        return self.propose_features(phi, current_state_vec=state_vec)

    def update(
        self,
        phi: np.ndarray,
        target_state_vec: np.ndarray,
        predicted_vec: np.ndarray | None = None,
        modulatory_signal: float = 1.0,
    ) -> float:
        """Apply a local pre-post-modulatory update and return mean error."""
        features = np.asarray(phi, dtype=float)
        target = np.asarray(target_state_vec, dtype=float)
        if features.shape != (self.feature_dim,):
            raise ValueError("phi has wrong shape")
        if target.shape != (self.state_dim,):
            raise ValueError("target_state_vec has wrong shape")
        if self.weight_decay:
            self.W *= 1.0 - self.weight_decay
        active = self._sparse_feature_indices(features)

        if self.update_rule in {"hebbian", "three_factor"}:
            update_vec = target
        else:
            prediction = (
                np.asarray(predicted_vec, dtype=float)
                if predicted_vec is not None
                else sign(self._raw_checked(features, active)).astype(float)
            )
            error = target - prediction
            if self.update_rule == "perceptron":
                update_vec = np.where(error != 0.0, target, 0.0)
            else:
                update_vec = error

        scale = self.learning_rate * modulatory_signal
        if active is None:
            self.W += scale * np.outer(update_vec, features)
        else:
            self.W[:, active] += scale * update_vec[:, np.newaxis] * features[active]
        if self.clip_value is not None:
            np.clip(self.W, -self.clip_value, self.clip_value, out=self.W)
        error_after = target - sign(self._raw_checked(features, active)).astype(float)
        return float(np.mean(np.abs(error_after)))


@dataclass
class EligibilityTraceTransitionLearner:
    """Delayed eligibility-trace learner for pair-to-state associations."""

    transition: LearnedAssociativeTransition
    trace_decay: float = 0.0
    normalize_features: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.trace_decay <= 1.0:
            raise ValueError("trace_decay must be in [0, 1]")
        self.eligibility = np.zeros(self.transition.feature_dim, dtype=float)

    def reset_trace(self) -> None:
        """Clear the eligibility trace."""
        self.eligibility.fill(0.0)

    def observe_pair(self, phi: np.ndarray) -> None:
        """Store active pair features in the eligibility trace."""
        features = np.asarray(phi, dtype=float)
        if features.shape != (self.transition.feature_dim,):
            raise ValueError("phi has wrong shape")
        if self.normalize_features:
            total = float(np.sum(np.abs(features)))
            if total > 0.0:
                features = features / total
        self.eligibility = self.trace_decay * self.eligibility + features

    def observe_next_state(
        self,
        target_state_vec: np.ndarray,
        modulatory_signal: float = 1.0,
    ) -> float:
        """Use the current eligibility trace to update toward the next state."""
        return self.transition.update(
            self.eligibility,
            target_state_vec,
            modulatory_signal=modulatory_signal,
        )

    def train_example(
        self,
        phi: np.ndarray,
        target_state_vec: np.ndarray,
        modulatory_signal: float = 1.0,
    ) -> float:
        """Train on one delayed pair-to-next-state demonstration."""
        self.observe_pair(phi)
        return self.observe_next_state(
            target_state_vec,
            modulatory_signal=modulatory_signal,
        )

    def train_epoch(
        self,
        examples: Sequence[tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator | None = None,
        shuffle: bool = True,
    ) -> float:
        """Train over a collection of ``(phi, target_state_vec)`` examples."""
        if not examples:
            return 0.0
        order = np.arange(len(examples))
        if shuffle and rng is not None:
            rng.shuffle(order)
        losses: list[float] = []
        for index in order:
            phi, target = examples[int(index)]
            self.reset_trace()
            losses.append(self.train_example(phi, target))
        return float(np.mean(losses)) if losses else 0.0
