# Moore Machine on Multi-Layer Hopfield Networks

## Table of Contents

1. [**Moore Machine on Multi-Layer Hopfield Networks**](#moore-machine-on-multi-layer-hopfield-networks)

2. [**🔍 Overview**](#-overview)

3. [**📄 Paper**](#-paper)

4. [**🔄 State Transitions**](#-state-transitions)

5. [**🧠 Key Concepts**](#-key-concepts)

6. [**📁 Project Structure**](#-project-structure)

    - [Core Components](#core-components)

7. [**🚀 Version 2.0 Plans - Neural Graph Computer**](#-version-20-plans---neural-graph-computer)

8. [**🎯 Core Architecture Changes**](#-core-architecture-changes)

    - [1. Distributed Minterm Processing](#1-distributed-minterm-processing)
    - [2. Smart Cross-Neuron Connections](#2-smart-cross-neuron-connections)
    - [3. Recurrent Error Correction](#3-recurrent-error-correction)

9. [**🧠 New Network Layers**](#-new-network-layers)

    - [4. Output Network ("Axon Layer")](#4-output-network-axon-layer)
    - [5. Data Input Layer](#5-data-input-layer)
    - [6. Self-Referential Control](#6-self-referential-control)

10. [**📚 Dynamic Learning Capabilities**](#-dynamic-learning-capabilities)

    - [7. Runtime Transition Learning](#7-runtime-transition-learning)
    - [8. Pattern Imposition Learning](#8-pattern-imposition-learning)
    - [9. Reinforcement-Based Adaptation](#9-reinforcement-based-adaptation)

11. [**🏗️ Implementation Roadmap**](#️-implementation-roadmap)

    - [Phase 1: Core Architecture](#phase-1-core-architecture)
    - [Phase 2: Extended I/O](#phase-2-extended-io)
    - [Phase 3: Learning Systems](#phase-3-learning-systems)
    - [Phase 4: Graph Computer Features](#phase-4-graph-computer-features)

12. [**🎯 Target Applications**](#-target-applications)

    - [Neural Graph Database](#neural-graph-database)
    - [Associative Programming Language](#associative-programming-language)
    - [Biologically-Inspired AI](#biologically-inspired-ai)

13. [**📊 Expected Performance Improvements**](#-expected-performance-improvements)

This repository provides a prototype implementation and theoretical model of a **Moore machine** constructed entirely using a **multi-layer Hopfield neural network**. The goal is to demonstrate the viability of implementing finite state machines (FSMs) within biologically plausible or digitally implemented Hopfield architectures—without auxiliary computational logic.

## 🔍 Overview

This project explores a novel way to merge concepts from automata theory with neuroscience-inspired computation. The model encodes Moore machine states as stable attractors in a Hopfield network’s energy landscape. Transitions are achieved through controlled perturbations of network state, simulating input-driven behavior typical of FSMs.

## 📄 Paper

The detailed explanation of the model, theoretical background, performance analysis, and comparisons to digital logic systems can be found in the draft [`Moore_Machine_on_Hopfield.pdf`](./Moore_Machine_on_Hopfield.pdf) or poster [`Split hopfield.pdf`](./Split%20hopfield.pdf).

## 🔄 State Transitions

![State Transitions](./state_transitions.png)

The diagram above illustrates the character relationship network used in this implementation:

-   **Blue lines**: "LIKES" relationships - showing affection/attraction between characters
-   **Green lines**: "FATHER_OF" relationships - depicting parental connections
-   **Red lines**: "BULLIES" relationships - representing antagonistic interactions

Each relationship type corresponds to a different input signal that triggers specific state transitions in the Moore machine. The network learns these relationships as associative memory patterns and can recall the appropriate target state when given a current state and relationship input.

## 🧠 Key Concepts

-   **Moore Machine**: A type of FSM where outputs depend only on the current state.
-   **Hopfield Network**: A recurrent neural network that minimizes an energy function to settle into memory-like attractor states.
-   **Multi-Layer Architecture**: Extends classical Hopfield networks by stacking layers, enabling richer and more complex state transitions.
-   **Biological Plausibility**: The design leverages biologically inspired neuron features (thresholds, oscillations, "write-enable"-like behavior) to achieve logic gate equivalents.

## 📁 Project Structure

```
hopfield_sequential/
├── README.md                          # This file
├── Moore_Machine_on_Hopfield.pdf      # Research paper
├── state_transitions.png              # Character relationship diagram
├── main.py                            # Main execution script
├── sequential.py                      # SequentialNetwork class implementation
├── src/
│   ├── network.py                     # Base Network class
│   ├── hopfieldNetwork.py             # HopfieldNetwork implementation
│   ├── state_generator.py             # Orthogonal state vector generation
│   └── utils.py                       # Utility functions for minterm generation
└── .gitignore                         # Git ignore rules
```

### Core Components

-   **`sequential.py`**: Main orchestrator class that sets up the multi-layer network architecture
-   **`src/network.py`**: Base network class with inter-layer connectivity and apical state computation
-   **`src/hopfieldNetwork.py`**: Hopfield associative memory implementation with pattern storage and recall
-   **`src/utils.py`**: Utility functions for creating minterm data and generating weight matrices
-   **`state_generator.py`**: Generates orthogonal bipolar vectors for character state representation

## 🚀 Version 2.0 Plans - Neural Graph Computer

Based on architectural analysis and biological plausibility improvements, **hopfield_sequential v2** will transform from a proof-of-concept Moore machine into a **distributed neural graph computer** with learning capabilities.

### 🎯 Core Architecture Changes

#### **1. Distributed Minterm Processing**

**Current (v1):** Single global minterm layer processes all transitions

```
Input_Layer → [Global_Minterm_Layer] → State_Layer
```

**Planned (v2):** Per-neuron distributed minterms with sparse connectivity

```
Input_Layer → [n₁_minterms] → state_neuron₁
            → [n₂_minterms] → state_neuron₂
            → [n₃_minterms] → state_neuron₃
```

**Benefits:**

-   🧠 **Biologically Plausible:** Each neuron processes locally, like real neurons
-   ⚡ **Computationally Efficient:** No global operations required
-   🔧 **Scalable:** Adding neurons doesn't explode complexity
-   🛡️ **Robust:** Damage to individual neurons doesn't break system

#### **2. Smart Cross-Neuron Connections**

Each state neuron will have **double connections** to related neurons:

-   **Minimum:** `ln(stored_states)` connections for basic functionality
-   **Maximum:** `stored_states` connections for full connectivity
-   **Optimal:** `√(stored_states × ln(stored_states))` connections for efficiency

**Connection Selection Strategies:**

-   **Correlation-based:** Connect neurons that frequently co-activate
-   **Pattern-based:** Connect neurons involved in similar transitions
-   **Learned:** Dynamically adjust connections based on performance

#### **3. Recurrent Error Correction**

Handle conflicting neuron decisions through peer consensus:

```python
if error_neurons << total_neurons:
    apply_peer_pressure_correction()  # Let majority fix minority errors
else:
    fallback_to_hopfield_dynamics()   # Too many errors, use energy landscape
```

**Error Tolerance:** ~10-15% error neurons can be corrected through recurrent connections.

### 🧠 New Network Layers

#### **4. Output Network ("Axon Layer")**

```
State_Layer → Output_Network → ASCII_Character
```

-   **Purpose:** Convert internal neural states to human-readable output
-   **Encoding:** 8-bit binary to ASCII character mapping
-   **Modes:**
    -   `always_output`: Continuous state monitoring
    -   `on_demand`: Output only when requested

#### **5. Data Input Layer**

```
Data_Input_Layer → State_Layer (direct pattern imposition)
```

-   **Purpose:** Directly impose new patterns for rapid learning
-   **Learning:** Incremental Hopfield weight updates with each imposition
-   **Use Cases:** Teaching new memories without slow convergence

#### **6. Self-Referential Control**

New input signals for introspection and control:

```python
CONTROL_SIGNALS = {
    "PRINT_STATE": [1, 0, 0, 1],    # Print current memory content
    "PRINT_TYPE": [0, 1, 0, 1],     # Print memory type/label
    "SET_TYPE": [0, 0, 1, 1],       # Change memory classification
    "LEARN_MODE": [1, 1, 0, 0],     # Enter learning mode
}
```

### 📚 Dynamic Learning Capabilities

#### **7. Runtime Transition Learning**

-   **Input Format:** `(from_state, relation_type, to_state)` tuples
-   **Integration:** Dynamically rebuild minterm structures
-   **Persistence:** Store learned transitions for future sessions

#### **8. Pattern Imposition Learning**

-   **Hebbian Updates:** Gradual weight adjustment with each imposed pattern
-   **Interference Management:** Balance old knowledge with new patterns
-   **Capacity Monitoring:** Track approaching Hopfield capacity limits

#### **9. Reinforcement-Based Adaptation**

-   **Reward Signals:** User feedback on output quality
-   **Weight Modulation:** Strengthen successful pathways
-   **Exploration:** Occasional random perturbations for discovery

### 🏗️ Implementation Roadmap

#### **Phase 1: Core Architecture**

-   [ ] Implement distributed minterm processing per neuron
-   [ ] Add smart cross-neuron connection algorithms
-   [ ] Integrate recurrent error correction mechanisms
-   [ ] Maintain backward compatibility with v1 interfaces

#### **Phase 2: Extended I/O**

-   [ ] Build output network with ASCII encoding
-   [ ] Implement data input layer for direct pattern imposition
-   [ ] Add self-referential control signal processing
-   [ ] Create debugging and introspection tools

#### **Phase 3: Learning Systems**

-   [ ] Runtime transition learning from user input
-   [ ] Incremental Hopfield learning for new patterns
-   [ ] Reinforcement learning for pathway optimization
-   [ ] Memory consolidation and forgetting mechanisms

#### **Phase 4: Graph Computer Features**

-   [ ] Multi-network communication protocols
-   [ ] Distributed graph traversal algorithms
-   [ ] Memory type classification and routing
-   [ ] Network composition and decomposition tools

### 🎯 Target Applications

#### **Neural Graph Database**

-   **Nodes:** Individual memory units (SequentialNetworks)
-   **Edges:** Learned relationships between concepts
-   **Queries:** Graph traversal through neural activation patterns

#### **Associative Programming Language**

-   **Variables:** Neural state patterns
-   **Functions:** Transition sequences
-   **Control Flow:** Minterm activation pathways
-   **I/O:** ASCII encoding/decoding through output networks

#### **Biologically-Inspired AI**

-   **Local Processing:** No global coordination required
-   **Fault Tolerance:** Graceful degradation with neuron damage
-   **Continuous Learning:** Adapt without catastrophic forgetting
-   **Introspection:** Self-monitoring and debugging capabilities

### 📊 Expected Performance Improvements

-   **🚀 Speed:** 2-3x faster through distributed processing
-   **📈 Scalability:** Linear growth vs exponential in v1
-   **🧠 Capacity:** Higher pattern storage through smart connections
-   **🛡️ Robustness:** 10-15% error tolerance through peer correction
-   **🎯 Accuracy:** Better convergence through local optimization

---

**Version 2.0** represents a fundamental evolution from a **proof-of-concept Moore machine** to a **practical neural graph computer** with real-world learning and memory capabilities. The modular v1 architecture enables these enhancements without breaking existing functionality.
