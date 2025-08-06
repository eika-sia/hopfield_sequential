# Moore Machine on Multi-Layer Hopfield Networks

This repository provides a prototype implementation and theoretical model of a **Moore machine** constructed entirely using a **multi-layer Hopfield neural network**. The goal is to demonstrate the viability of implementing finite state machines (FSMs) within biologically plausible or digitally implemented Hopfield architectures—without auxiliary computational logic.

## 🔍 Overview

This project explores a novel way to merge concepts from automata theory with neuroscience-inspired computation. The model encodes Moore machine states as stable attractors in a Hopfield network’s energy landscape. Transitions are achieved through controlled perturbations of network state, simulating input-driven behavior typical of FSMs.

## 📄 Paper

The detailed explanation of the model, theoretical background, performance analysis, and comparisons to digital logic systems can be found in [`Moore_Machine_on_Hopfield.pdf`](./Moore_Machine_on_Hopfield.pdf).

## 🧠 Key Concepts

-   **Moore Machine**: A type of FSM where outputs depend only on the current state.
-   **Hopfield Network**: A recurrent neural network that minimizes an energy function to settle into memory-like attractor states.
-   **Multi-Layer Architecture**: Extends classical Hopfield networks by stacking layers, enabling richer and more complex state transitions.
-   **Biological Plausibility**: The design leverages biologically inspired neuron features (thresholds, oscillations, "write-enable"-like behavior) to achieve logic gate equivalents.
