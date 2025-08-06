from typing import Optional
import numpy as np
from numpy.typing import NDArray

class Network:
    def __init__(self, N: int) -> None:
        if N <= 0:
            raise ValueError("Network size N must be positive")
        
        self.N: int = N
        self.currState: NDArray[np.integer] = np.random.choice([-1, 1], size=(N,))  #* Random initial state

        #? Initialize weights and thresholds
        self.lowerWeights: NDArray[np.floating] = np.zeros((N, N), dtype=float)  #* N×N matrix
        self.upperWeights: NDArray[np.floating] = np.zeros((N, N), dtype=float)  #* N×N matrix
        self.thresholds: NDArray[np.floating] = np.zeros((N,), dtype=float)  #* N-dimensional vector

        self.upperNetwork: Optional['Network'] = None  #* Optional upper network connection
        self.lowerNetwork: Optional['Network'] = None  #* Optional lower network connection

    def set_current_state(self, state: NDArray[np.integer]) -> None:
        if state.shape != (self.N,):
            raise ValueError(f"State must be a {self.N}-dimensional vector.")
        self.currState = state.copy()

    def set_upper_network(self, upper_network: 'Network') -> None:
        self.upperNetwork = upper_network

    def set_lower_network(self, lower_network: 'Network') -> None:
        self.lowerNetwork = lower_network
        
    def get_state(self) -> np.ndarray:
        """Get copy of current state (prevents external modification)"""
        return self.currState.copy()

    def set_outer_weights(self, lower: NDArray[np.floating], upper: NDArray[np.floating], thresholds: NDArray[np.floating]) -> None:
        #? Validate that lower and upper are well shaped matrices
        self._validate_weights(lower, upper, thresholds)
            
        self.lowerWeights = lower.copy()
        self.upperWeights = upper.copy()
        self.thresholds = thresholds.copy()
    
    def _validate_weights(self, lower: NDArray[np.floating], upper: NDArray[np.floating], 
                         thresholds: NDArray[np.floating]) -> None:
        """Validate weight matrices and thresholds."""
        # Validate dimensions
        if self.lowerNetwork is not None and lower.shape != (self.N, self.lowerNetwork.N):
            raise ValueError(f"Lower weights must be {self.N}×{self.lowerNetwork.N}, got {lower.shape}")
        
        if self.upperNetwork is not None and upper.shape != (self.N, self.upperNetwork.N):
            raise ValueError(f"Upper weights must be {self.N}×{self.upperNetwork.N}, got {upper.shape}")

        if thresholds.shape != (self.N,):
            raise ValueError(f"Thresholds must be {self.N}-dimensional, got {thresholds.shape}")
        
        # Validate data types
        if not np.issubdtype(lower.dtype, np.floating):
            raise ValueError("Lower weights must be float type")
        if not np.issubdtype(upper.dtype, np.floating):
            raise ValueError("Upper weights must be float type")
        if not np.issubdtype(thresholds.dtype, np.floating):
            raise ValueError("Thresholds must be float type")
    
    
    def compute_apical_state(self) -> np.ndarray:
        """
        Compute the apical state based on the current state and upper weights.
        This is a simplified version of how apical states might be computed.
        """
        weighted_input = np.zeros(self.N, dtype=float)
        if self.upperNetwork is not None:
            weighted_input += np.dot(self.upperWeights, self.upperNetwork.currState)
        if self.lowerNetwork is not None:
            weighted_input += np.dot(self.lowerWeights, self.lowerNetwork.currState)
        
        #? Apply thresholds to determine the apical state
        new_state = np.sign(weighted_input - self.thresholds)
        self.set_current_state(new_state)  #* Update current state
        return new_state.copy()
