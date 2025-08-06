from typing import Dict, List
import numpy as np
from numpy.typing import NDArray

from src.network import Network
from src.hopfieldNetwork import HopfieldNetwork
from src.utils import create_minterm_data, TransitionDict, generate_minterm_weights

class SequentialNetwork:
    def __init__(self, states: Dict[str, NDArray[np.integer]], transitions: List[TransitionDict], input_strings: List[str]) -> None:
        self.states = states
        self.transitions = transitions
        self.input_strings = input_strings

        self.N = list(states.values())[0].shape[0]

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
        user_input = self._valid_input()
        self.input_layer.set_current_state(self.inputs[user_input])
        
        #? Compute minterm layer state
        self.minterm_layer.compute_apical_state()
        self.state_layer.compute_apical_state()
        
        #? Compute Hopfield state
        new_state = self.state_layer.compute_hopfield_state()
        
        return new_state
