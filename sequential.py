from typing import Any, Dict, List
import numpy as np
from numpy.typing import NDArray
import logging

from src.network import Network
from src.hopfieldNetwork import HopfieldNetwork
from src.utils import create_minterm_data, TransitionDict, generate_minterm_weights

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

class SequentialNetwork:
    def __init__(self, states: Dict[str, NDArray[np.integer]], transitions: List[TransitionDict], input_strings: List[str]) -> None:
        # Validate inputs
        if not states:
            raise ValueError("States dictionary cannot be empty")
        if len(transitions) != len(input_strings):
            raise ValueError(f"Transitions ({len(transitions)}) and input_strings ({len(input_strings)}) must have same length")
        
        # Check all states have same dimension
        state_dims = [state.shape[0] for state in states.values()]
        if not all(dim == state_dims[0] for dim in state_dims):
            raise ValueError("All states must have the same dimensions")
        
        self.states = states
        self.transitions = transitions
        self.input_strings = input_strings
        self.N = state_dims[0]  # More robust than list(states.values())[0].shape[0]
        
        self._initialise_networks()
    
    def _initialise_networks(self) -> None:
        n = len(self.transitions)
        bipolar_matrix = np.full((n, n), -1, dtype=int)
        np.fill_diagonal(bipolar_matrix, 1)
        self.inputs: Dict[str, NDArray[np.integer]] = {
            self.input_strings[i]: bipolar_matrix[i] for i in range(n)
        }
        
        minterms = create_minterm_data(self.states, self.transitions, list(self.inputs.values()))
        minterm_data = generate_minterm_weights(minterms)
        
        self.input_layer = Network(len(self.inputs))
        self.minterm_layer = Network(N = minterm_data['thresholds'].shape[0])
        self.state_layer = HopfieldNetwork(N = self.N)

        self.minterm_layer.set_lower_network(self.input_layer)
        self.minterm_layer.set_upper_network(self.state_layer)
        self.minterm_layer.set_outer_weights(
            lower=minterm_data['lowerWeights'],
            upper=minterm_data['upperWeights'],
            thresholds=minterm_data['thresholds']
        )

        self.state_layer.add_states(self.states)
        self.state_layer.set_lower_network(self.minterm_layer)
        self.state_layer.set_outer_weights(
            lower=minterm_data['lowerStateWeights'],
            upper=np.zeros((self.N, self.N)),  #* No upper weights in this case
            thresholds=minterm_data['stateThresholds']
        )
        self.state_layer.set_current_state(list(self.states.values())[0])  #* Set initial state
    
    def _valid_input(self) -> str:
        user_input = input(f"Enter a relation signal ({list(self.inputs.keys())}): ").upper()
        if user_input not in self.inputs:
            raise ValueError(f"Invalid relation signal. Expected one of {list(self.inputs.keys())}.")
        return user_input
    
    def process_input(self) -> str:
        """Process user input and compute the network state."""
        try:
            user_input = self._valid_input()
            self.input_layer.set_current_state(self.inputs[user_input])
            
            # Store intermediate states for debugging
            minterm_state = self.minterm_layer.compute_apical_state()
            self.state_layer.compute_apical_state()
            final_state = self.state_layer.compute_hopfield_state()
            
            log.debug(f"Minterm activations: {np.sum(minterm_state == 1)} active")
            log.debug(f"Final state: {final_state}")
            
            return final_state
            
        except Exception as e:
            log.error(f"Error processing input: {e}")
            return "error_state"
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        return {
            "num_states": len(self.states),
            "state_dimension": self.N,
            "num_minterms": self.minterm_layer.N,
            "num_transitions": len(self.transitions),
            "hopfield_capacity": self.state_layer.get_network_info(),
            "input_signals": list(self.inputs.keys())
        }
