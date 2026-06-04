"""Structured finite-state grammar tasks for sequence experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

import numpy as np

SymbolString = list[str]
LabeledString = tuple[SymbolString, bool]
CachedLabeledString = tuple[tuple[str, ...], bool]
PositiveSampler = Callable[[tuple[int, int], np.random.Generator], SymbolString]
MAX_EXHAUSTIVE_STRINGS: int = 250_000
_ENUMERATION_CACHE: dict[tuple[str, int], tuple[CachedLabeledString, ...]] = {}


def _search_size(alphabet_size: int, max_len: int) -> int:
    return sum(alphabet_size**length for length in range(max_len + 1))


@dataclass(frozen=True)
class DFA:
    """A total deterministic finite automaton."""

    states: list[str]
    alphabet: list[str]
    start_state: str
    accept_states: set[str]
    transition: dict[tuple[str, str], str]

    def __post_init__(self) -> None:
        if self.start_state not in self.states:
            raise ValueError("start_state must be in states")
        unknown_accepts = self.accept_states - set(self.states)
        if unknown_accepts:
            raise ValueError(f"unknown accept states: {sorted(unknown_accepts)}")
        for state in self.states:
            for symbol in self.alphabet:
                target = self.transition.get((state, symbol))
                if target not in self.states:
                    raise ValueError(
                        f"missing or invalid transition for ({state!r}, {symbol!r})"
                    )

    @property
    def num_states(self) -> int:
        """Return the number of DFA states."""
        return len(self.states)

    def step(self, state: str, symbol: str) -> str:
        """Advance one DFA transition."""
        if symbol not in self.alphabet:
            raise ValueError(f"symbol {symbol!r} is not in alphabet")
        return self.transition[(state, symbol)]

    def run(self, symbols: list[str]) -> tuple[str, bool]:
        """Return final state and acceptance for a string."""
        state = self.start_state
        for symbol in symbols:
            state = self.step(state, symbol)
        return state, state in self.accept_states

    def accepts(self, symbols: list[str]) -> bool:
        """Return whether the DFA accepts a string."""
        return self.run(symbols)[1]

    def trace(self, symbols: list[str]) -> list[str]:
        """Return states including start state and all subsequent states."""
        states = [self.start_state]
        state = self.start_state
        for symbol in symbols:
            state = self.step(state, symbol)
            states.append(state)
        return states

    def transition_items(self) -> list[tuple[str, str, str]]:
        """Return all ``(state, symbol, next_state)`` transitions."""
        return [
            (state, symbol, self.transition[(state, symbol)])
            for state in self.states
            for symbol in self.alphabet
        ]

    def reachable_states(self) -> set[str]:
        """Return states reachable from the start state."""
        reached = {self.start_state}
        queue: deque[str] = deque([self.start_state])
        while queue:
            state = queue.popleft()
            for symbol in self.alphabet:
                nxt = self.step(state, symbol)
                if nxt not in reached:
                    reached.add(nxt)
                    queue.append(nxt)
        return reached


@dataclass(frozen=True)
class GrammarTask:
    """A named regular-language task backed by a DFA."""

    name: str
    dfa: DFA
    description: str
    positive_sampler: PositiveSampler | None = None

    @property
    def alphabet(self) -> list[str]:
        """Return the task alphabet."""
        return self.dfa.alphabet

    def sample_positive(
        self,
        length_range: tuple[int, int],
        rng: np.random.Generator,
    ) -> list[str]:
        """Sample one accepted string."""
        if self.positive_sampler is not None:
            return self.positive_sampler(length_range, rng)
        return self._sample_by_label(True, length_range, rng)

    def sample_negative(
        self,
        length_range: tuple[int, int],
        rng: np.random.Generator,
    ) -> list[str]:
        """Sample one rejected string."""
        return self._sample_by_label(False, length_range, rng)

    def sample_balanced_dataset(
        self,
        n: int,
        length_range: tuple[int, int],
        rng: np.random.Generator,
    ) -> list[LabeledString]:
        """Sample approximately balanced accepted/rejected strings."""
        if n <= 0:
            return []
        min_len, max_len = length_range
        search_size = _search_size(len(self.alphabet), max_len)
        if search_size <= MAX_EXHAUSTIVE_STRINGS and self.positive_sampler is None:
            candidates = [
                (list(symbols), label)
                for symbols, label in _cached_enumeration(self, max_len)
                if len(symbols) >= min_len
            ]
            positives = [item for item in candidates if item[1]]
            negatives = [item for item in candidates if not item[1]]
            if positives and negatives:
                examples: list[LabeledString] = []
                for idx in range(n):
                    pool = positives if idx % 2 == 0 else negatives
                    symbols, label = pool[int(rng.integers(0, len(pool)))]
                    examples.append((list(symbols), label))
                rng.shuffle(examples)
                return examples

        examples: list[LabeledString] = []
        for idx in range(n):
            want_positive = idx % 2 == 0
            try:
                symbols = (
                    self.sample_positive(length_range, rng)
                    if want_positive
                    else self.sample_negative(length_range, rng)
                )
            except RuntimeError:
                symbols = self._sample_random(length_range, rng)
            examples.append((symbols, self.dfa.accepts(symbols)))
        rng.shuffle(examples)
        return examples

    def enumerate_strings(self, max_len: int) -> list[LabeledString]:
        """Enumerate all strings up to ``max_len`` with labels."""
        if max_len < 0:
            raise ValueError("max_len must be nonnegative")
        search_size = _search_size(len(self.alphabet), max_len)
        if search_size > MAX_EXHAUSTIVE_STRINGS:
            raise ValueError(
                f"refusing to enumerate {search_size} strings; "
                f"limit is {MAX_EXHAUSTIVE_STRINGS}"
            )
        examples: list[LabeledString] = []
        for length in range(max_len + 1):
            for symbols in product(self.alphabet, repeat=length):
                word = list(symbols)
                examples.append((word, self.dfa.accepts(word)))
        return examples

    def train_test_split(
        self,
        train_max_len: int,
        test_max_len: int,
    ) -> tuple[list[LabeledString], list[LabeledString], list[LabeledString]]:
        """Split exhaustive strings into train, in-length test, and longer test."""
        all_examples = self.enumerate_strings(test_max_len)
        train = [item for item in all_examples if len(item[0]) <= train_max_len]
        in_length = [item for item in all_examples if len(item[0]) <= train_max_len]
        longer = [item for item in all_examples if len(item[0]) > train_max_len]
        return train, in_length, longer

    def _sample_random(
        self,
        length_range: tuple[int, int],
        rng: np.random.Generator,
    ) -> list[str]:
        min_len, max_len = length_range
        if min_len < 0 or max_len < min_len:
            raise ValueError("invalid length_range")
        length = int(rng.integers(min_len, max_len + 1))
        return [str(rng.choice(self.alphabet)) for _ in range(length)]

    def _sample_by_label(
        self,
        accepted: bool,
        length_range: tuple[int, int],
        rng: np.random.Generator,
    ) -> list[str]:
        for _ in range(10_000):
            symbols = self._sample_random(length_range, rng)
            if self.dfa.accepts(symbols) == accepted:
                return symbols
        max_len = length_range[1]
        if _search_size(len(self.alphabet), max_len) > MAX_EXHAUSTIVE_STRINGS:
            raise RuntimeError("label is too rare for random sampling")
        candidates = [
            symbols
            for symbols, label in _cached_enumeration(self, max_len)
            if len(symbols) >= length_range[0] and label == accepted
        ]
        if not candidates:
            raise RuntimeError("no strings with requested label in length range")
        return list(candidates[int(rng.integers(0, len(candidates)))])


def _cached_enumeration(
    task: GrammarTask,
    max_len: int,
) -> tuple[CachedLabeledString, ...]:
    key = (task.name, max_len)
    cached = _ENUMERATION_CACHE.get(key)
    if cached is None:
        cached = tuple(
            (tuple(symbols), label)
            for symbols, label in task.enumerate_strings(max_len)
        )
        _ENUMERATION_CACHE[key] = cached
    return cached


def _complete_with_dead(
    states: list[str],
    alphabet: list[str],
    transitions: dict[tuple[str, str], str],
    dead: str = "dead",
) -> tuple[list[str], dict[tuple[str, str], str]]:
    out_states = list(states)
    if dead not in out_states:
        out_states.append(dead)
    out = dict(transitions)
    for state in out_states:
        for symbol in alphabet:
            out.setdefault((state, symbol), dead)
    return out_states, out


def even_ones_task() -> GrammarTask:
    states = ["even", "odd"]
    alphabet = ["0", "1"]
    transitions = {
        ("even", "0"): "even",
        ("even", "1"): "odd",
        ("odd", "0"): "odd",
        ("odd", "1"): "even",
    }
    return GrammarTask(
        "even_ones",
        DFA(states, alphabet, "even", {"even"}, transitions),
        "accept iff the number of 1 symbols is even",
    )


def no_substring_11_task() -> GrammarTask:
    states = ["start", "last1", "dead"]
    alphabet = ["0", "1"]
    transitions = {
        ("start", "0"): "start",
        ("start", "1"): "last1",
        ("last1", "0"): "start",
        ("last1", "1"): "dead",
        ("dead", "0"): "dead",
        ("dead", "1"): "dead",
    }
    return GrammarTask(
        "no_substring_11",
        DFA(states, alphabet, "start", {"start", "last1"}, transitions),
        'accept iff the string does not contain substring "11"',
    )


def ends_with_01_task() -> GrammarTask:
    states = ["none", "last0", "ends01"]
    alphabet = ["0", "1"]
    transitions = {
        ("none", "0"): "last0",
        ("none", "1"): "none",
        ("last0", "0"): "last0",
        ("last0", "1"): "ends01",
        ("ends01", "0"): "last0",
        ("ends01", "1"): "none",
    }
    return GrammarTask(
        "ends_with_01",
        DFA(states, alphabet, "none", {"ends01"}, transitions),
        'accept iff the string ends with suffix "01"',
    )


def contains_101_task() -> GrammarTask:
    states = ["none", "seen1", "seen10", "found"]
    alphabet = ["0", "1"]
    transitions = {
        ("none", "0"): "none",
        ("none", "1"): "seen1",
        ("seen1", "0"): "seen10",
        ("seen1", "1"): "seen1",
        ("seen10", "0"): "none",
        ("seen10", "1"): "found",
        ("found", "0"): "found",
        ("found", "1"): "found",
    }
    return GrammarTask(
        "contains_101",
        DFA(states, alphabet, "none", {"found"}, transitions),
        'accept iff the string contains substring "101"',
    )


def mod3_ones_task() -> GrammarTask:
    states = ["r0", "r1", "r2"]
    alphabet = ["0", "1"]
    transitions = {
        ("r0", "0"): "r0",
        ("r0", "1"): "r1",
        ("r1", "0"): "r1",
        ("r1", "1"): "r2",
        ("r2", "0"): "r2",
        ("r2", "1"): "r0",
    }
    return GrammarTask(
        "ones_mod3_zero",
        DFA(states, alphabet, "r0", {"r0"}, transitions),
        "accept iff count(1) mod 3 == 0",
    )


def tier_alternating_12_task() -> GrammarTask:
    states = ["start", "last1", "last2", "dead"]
    alphabet = ["0", "1", "2"]
    transitions = {
        ("start", "0"): "start",
        ("start", "1"): "last1",
        ("start", "2"): "last2",
        ("last1", "0"): "last1",
        ("last1", "1"): "dead",
        ("last1", "2"): "last2",
        ("last2", "0"): "last2",
        ("last2", "1"): "last1",
        ("last2", "2"): "dead",
        ("dead", "0"): "dead",
        ("dead", "1"): "dead",
        ("dead", "2"): "dead",
    }
    return GrammarTask(
        "tier_alternating_12",
        DFA(states, alphabet, "start", {"start", "last1", "last2"}, transitions),
        "accept iff the 1/2 tier alternates after deleting 0 symbols",
    )


def tomita_1_task() -> GrammarTask:
    states, transitions = _complete_with_dead(
        ["only1"],
        ["0", "1"],
        {("only1", "1"): "only1"},
    )
    return GrammarTask(
        "tomita_1",
        DFA(states, ["0", "1"], "only1", {"only1"}, transitions),
        "Tomita 1: 1*",
    )


def tomita_2_task() -> GrammarTask:
    states, transitions = _complete_with_dead(
        ["expect1", "expect0"],
        ["0", "1"],
        {
            ("expect1", "1"): "expect0",
            ("expect0", "0"): "expect1",
        },
    )
    return GrammarTask(
        "tomita_2",
        DFA(states, ["0", "1"], "expect1", {"expect1"}, transitions),
        "Tomita 2: (10)*",
    )


def tomita_3_task() -> GrammarTask:
    states = ["safe", "ones_odd", "ones_even", "zeros_odd", "zeros_even", "dead"]
    alphabet = ["0", "1"]
    transitions = {
        ("safe", "0"): "safe",
        ("safe", "1"): "ones_odd",
        ("ones_odd", "0"): "zeros_odd",
        ("ones_odd", "1"): "ones_even",
        ("ones_even", "0"): "safe",
        ("ones_even", "1"): "ones_odd",
        ("zeros_odd", "0"): "zeros_even",
        ("zeros_odd", "1"): "dead",
        ("zeros_even", "0"): "zeros_odd",
        ("zeros_even", "1"): "ones_odd",
        ("dead", "0"): "dead",
        ("dead", "1"): "dead",
    }
    return GrammarTask(
        "tomita_3",
        DFA(
            states,
            alphabet,
            "safe",
            {"safe", "ones_odd", "ones_even", "zeros_even"},
            transitions,
        ),
        "Tomita 3: odd runs of 1s must be followed by even runs of 0s",
    )


def tomita_4_task() -> GrammarTask:
    states = ["z0", "z1", "z2", "dead"]
    alphabet = ["0", "1"]
    transitions = {
        ("z0", "0"): "z1",
        ("z0", "1"): "z0",
        ("z1", "0"): "z2",
        ("z1", "1"): "z0",
        ("z2", "0"): "dead",
        ("z2", "1"): "z0",
        ("dead", "0"): "dead",
        ("dead", "1"): "dead",
    }
    return GrammarTask(
        "tomita_4",
        DFA(states, alphabet, "z0", {"z0", "z1", "z2"}, transitions),
        'Tomita 4: no substring "000"',
    )


def tomita_5_task() -> GrammarTask:
    states = ["ee", "eo", "oe", "oo"]
    alphabet = ["0", "1"]
    transitions = {
        ("ee", "0"): "oe",
        ("ee", "1"): "eo",
        ("eo", "0"): "oo",
        ("eo", "1"): "ee",
        ("oe", "0"): "ee",
        ("oe", "1"): "oo",
        ("oo", "0"): "eo",
        ("oo", "1"): "oe",
    }
    return GrammarTask(
        "tomita_5",
        DFA(states, alphabet, "ee", {"ee"}, transitions),
        "Tomita 5: even number of 0s and even number of 1s",
    )


def tomita_6_task() -> GrammarTask:
    states = ["r0", "r1", "r2"]
    alphabet = ["0", "1"]
    transitions = {
        ("r0", "0"): "r1",
        ("r0", "1"): "r2",
        ("r1", "0"): "r2",
        ("r1", "1"): "r0",
        ("r2", "0"): "r0",
        ("r2", "1"): "r1",
    }
    return GrammarTask(
        "tomita_6",
        DFA(states, alphabet, "r0", {"r0"}, transitions),
        "Tomita 6: count(0)-count(1) is a multiple of 3",
    )


def tomita_7_task() -> GrammarTask:
    states, transitions = _complete_with_dead(
        ["phase0", "phase1", "phase2", "phase3"],
        ["0", "1"],
        {
            ("phase0", "0"): "phase0",
            ("phase0", "1"): "phase1",
            ("phase1", "1"): "phase1",
            ("phase1", "0"): "phase2",
            ("phase2", "0"): "phase2",
            ("phase2", "1"): "phase3",
            ("phase3", "1"): "phase3",
        },
    )
    return GrammarTask(
        "tomita_7",
        DFA(
            states,
            ["0", "1"],
            "phase0",
            {"phase0", "phase1", "phase2", "phase3"},
            transitions,
        ),
        "Tomita 7: 0*1*0*1*",
    )


def reber_task() -> GrammarTask:
    alphabet = ["B", "T", "P", "S", "X", "V", "E"]
    states, transitions = _complete_with_dead(
        ["q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"],
        alphabet,
        {
            ("q0", "B"): "q1",
            ("q1", "T"): "q2",
            ("q1", "P"): "q3",
            ("q2", "S"): "q2",
            ("q2", "X"): "q4",
            ("q3", "T"): "q3",
            ("q3", "V"): "q5",
            ("q4", "X"): "q3",
            ("q4", "S"): "q6",
            ("q5", "P"): "q4",
            ("q5", "V"): "q6",
            ("q6", "E"): "q7",
        },
    )

    def sample_reber(
        length_range: tuple[int, int],
        rng: np.random.Generator,
    ) -> list[str]:
        del length_range
        state = "q0"
        symbols: list[str] = []
        for _ in range(200):
            options = [
                (symbol, target)
                for (src, symbol), target in transitions.items()
                if src == state and target != "dead"
            ]
            if not options:
                break
            symbol, state = options[int(rng.integers(0, len(options)))]
            symbols.append(symbol)
            if state == "q7":
                break
        return symbols

    return GrammarTask(
        "reber",
        DFA(states, alphabet, "q0", {"q7"}, transitions),
        "standard Reber grammar",
        positive_sampler=sample_reber,
    )


def custom_tasks() -> list[GrammarTask]:
    """Return interpretable local structured regular-language tasks."""
    return [
        even_ones_task(),
        no_substring_11_task(),
        ends_with_01_task(),
        contains_101_task(),
        mod3_ones_task(),
        tier_alternating_12_task(),
    ]


def tomita_tasks() -> list[GrammarTask]:
    """Return the seven standard Tomita grammar tasks."""
    return [
        tomita_1_task(),
        tomita_2_task(),
        tomita_3_task(),
        tomita_4_task(),
        tomita_5_task(),
        tomita_6_task(),
        tomita_7_task(),
    ]


def all_tasks() -> list[GrammarTask]:
    """Return all implemented structured grammar tasks."""
    return [*custom_tasks(), *tomita_tasks(), reber_task()]


def task_by_name(name: str) -> GrammarTask:
    """Look up a grammar task by name."""
    for task in all_tasks():
        if task.name == name:
            return task
    raise KeyError(f"unknown grammar task: {name}")


def transition_coverage(
    dfa: DFA,
    strings: Iterable[list[str]],
) -> tuple[set[tuple[str, str]], float]:
    """Return covered state-symbol transitions and coverage fraction."""
    covered: set[tuple[str, str]] = set()
    reachable = dfa.reachable_states()
    possible = {(state, symbol) for state in reachable for symbol in dfa.alphabet}
    for symbols in strings:
        state = dfa.start_state
        for symbol in symbols:
            covered.add((state, symbol))
            state = dfa.step(state, symbol)
    return covered, len(covered & possible) / len(possible) if possible else 0.0


def strings_covering_transitions(
    task: GrammarTask,
    length_range: tuple[int, int],
    rng: np.random.Generator,
    max_strings: int,
) -> list[list[str]]:
    """Sample strings until reachable DFA transitions are covered or budget ends."""
    strings: list[list[str]] = []
    reachable = task.dfa.reachable_states()
    required = {(state, symbol) for state in reachable for symbol in task.alphabet}
    covered: set[tuple[str, str]] = set()
    if _search_size(len(task.alphabet), length_range[1]) <= MAX_EXHAUSTIVE_STRINGS:
        for cached_symbols, _label in _cached_enumeration(task, length_range[1]):
            symbols = list(cached_symbols)
            if len(symbols) < length_range[0]:
                continue
            before = len(covered)
            state = task.dfa.start_state
            for symbol in symbols:
                covered.add((state, symbol))
                state = task.dfa.step(state, symbol)
            if len(covered) > before:
                strings.append(symbols)
            if required <= covered or len(strings) >= max_strings:
                break
    while len(strings) < max_strings and not required <= covered:
        try:
            symbols = task.sample_balanced_dataset(1, length_range, rng)[0][0]
        except RuntimeError:
            symbols = task._sample_random(length_range, rng)
        strings.append(symbols)
        state = task.dfa.start_state
        for symbol in symbols:
            covered.add((state, symbol))
            state = task.dfa.step(state, symbol)
    return strings
