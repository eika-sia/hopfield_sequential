import numpy as np
from typing import Dict, Union, List
import logging
from src.network import Network
from numpy.typing import NDArray

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

class HopfieldNetwork(Network):
    def __init__(self, N: int) -> None:
        super().__init__(N)
        
        #? Additional hybrid functionality
        self.states: Dict[str, NDArray[np.integer]] = {}  #* id -> state mapping

    def add_states(self, states: Union[Dict[str, NDArray[np.integer]], List[NDArray[np.integer]]]) -> None:
        """
        Add states to the network. Can accept either:
        - Dictionary: {id: state_vector, ...}
        - List: [state1, state2, ...] (auto-generates ids)
        """
        if isinstance(states, dict):
            self._add_states_from_dict(states)
        else:
            self._add_states_from_list(states)
        
        # Recompute Hopfield weights after adding new states
        self.compute_hopfield_weights()

    def _add_states_from_dict(self, states: Dict[str, NDArray[np.integer]]) -> None:
        """Add states from dictionary input."""
        for state_id, state in states.items():
            if state.shape != (self.N,):
                raise ValueError(f"State '{state_id}' must be {self.N}-dimensional vector, got shape {state.shape}")
            self.add_single_state(state_id, state)

    def _add_states_from_list(self, states: List[NDArray[np.integer]]) -> None:
        """Add states from list input with auto-generated IDs."""
        if not all(state.shape == (self.N,) for state in states):
            raise ValueError(f"All states must be {self.N}-dimensional vectors.")
        
        for i, state in enumerate(states):
            state_id = f"pattern_{len(self.states) + i}"
            self.add_single_state(state_id, state)

    def add_single_state(self, state_id: str, state: NDArray[np.integer]) -> None:
        """Add a single state with specified ID."""
        if state.shape != (self.N,):
            raise ValueError(f"State must be {self.N}-dimensional vector, got shape {state.shape}")
        
        self.states[state_id] = state.copy()
        self.compute_hopfield_weights()
    
    def get_state_by_id(self, state_id: str) -> np.ndarray:
        """Retrieve a specific state by ID."""
        if state_id not in self.states:
            raise KeyError(f"State '{state_id}' not found. Available states: {list(self.states.keys())}")
        return self.states[state_id].copy()

    def list_states(self) -> List[str]:
        """Get list of all state IDs."""
        return list(self.states.keys())

    def remove_state(self, state_id: str) -> None:
        """Remove a state by ID."""
        if state_id not in self.states:
            raise KeyError(f"State '{state_id}' not found")
        
        del self.states[state_id]
        self.compute_hopfield_weights()
    
    def compute_hopfield_weights(self) -> None:
        """
        Compute Hopfield weight matrix using the standard Hebbian learning rule:
        W_ij = (1/N) * sum_k(s_k_i * s_k_j) for i != j, and W_ii = 0
        where s_k is the k-th stored state vector.
        """
        self.weights: np.ndarray = np.zeros((self.N, self.N), dtype=float)  #* N×N matrix for inner weights

        if len(self.states) == 0:
            raise ValueError("No states available to compute weights. Add states first.")
        
        #? Standard Hopfield weight computation using Hebbian rule
        for state in self.states.values():
            #? Outer product of state with itself
            self.weights += np.outer(state, state)
        
        #? Normalize by number of patterns and remove self-connections
        self.weights = self.weights / len(self.states)
        np.fill_diagonal(self.weights, 0)  #* Set diagonal to zero (no self-connections)

    def compute_hopfield_state(self) -> str:
        """
        Iterate the Hopfield network from current state until it settles.
        Uses synchronous update rule: s_i(t+1) = sign(sum_j(W_ij * s_j(t)))
        
        Returns:
            np.ndarray: The settled state that the network converged to
        """
        if not hasattr(self, 'weights'):
            raise ValueError("No weights computed. Add states and compute weights first.")
        
        current = self.currState.copy()
        max_iterations = 1000  #? Prevent infinite loops
        
        for _ in range(max_iterations):
            #? Compute weighted input for each neuron
            weighted_input = np.dot(self.weights, current)
            
            #? Apply sign activation function (bipolar: -1, +1)
            new_state: np.ndarray = np.sign(weighted_input)
            
            #? Handle zero values (when weighted_input is exactly 0)
            #? Keep previous state for zero inputs to maintain stability
            zero_mask = (weighted_input == 0)
            new_state[zero_mask] = 1
            
            #? Check for convergence (state hasn't changed)
            if np.array_equal(current, new_state):
                self.currState = new_state.copy()
                log.debug(f"Network converged to state: {new_state}")
                return self.find_closest_state(new_state)

            current = new_state
        
        #? If we reach here, the network didn't converge
        #? This can happen with spurious states or oscillations
        log.warning(f"Network did not converge after {max_iterations} iterations")
        self.currState = current.copy()
        return self.find_closest_state(current)

    def find_closest_state(self, target_state: np.ndarray, threshold: float = 0.75) -> str:
        """Find the stored state ID that is closest to the target state."""
        if len(self.states) == 0:
            raise ValueError("No states stored in network")
        
        best_match = "Unknown pattern"
        best_similarity = -1
        
        for state_id, state in self.states.items():
            #? Compute dot product (similarity for bipolar vectors)
            similarity = np.dot(target_state, state) / self.N
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = state_id

        return best_match if best_similarity >= threshold else "Unknown pattern"
    
    def get_network_info(self) -> Dict[str, Union[int, float]]:
        """Get network statistics."""
        return {
            "num_patterns": len(self.states),
            "dimension": self.N,
            "capacity_ratio": len(self.states) / (0.14 * self.N)  #* Hopfield capacity limit
        }
        
    def compute_energy(self, state: np.ndarray) -> float:
        """Compute Hopfield energy for a given state."""
        return -0.5 * np.dot(state, np.dot(self.weights, state))
