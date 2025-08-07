from typing import List, Dict, Tuple
import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
MintermDict = Dict[str, NDArray[np.integer]]
WeightDict = Dict[str, NDArray[np.floating]]
TransitionDict = List[Tuple[str, str]]

def are_almost_orthogonal(vectors: list[np.ndarray], threshold: float) -> bool:
    """Check if all vectors are almost orthogonal (dot product below threshold)"""
    n_vectors = len(vectors)
    for i in range(n_vectors):
        for j in range(i + 1, n_vectors):
            dot_product = np.dot(vectors[i], vectors[j])
            if abs(dot_product) > threshold:  #? Check if exceeds orthogonality threshold
                return False
    return True

def create_minterm_data(states: Dict[str, NDArray[np.integer]], transition_data: List[TransitionDict], inputs: List[NDArray[np.integer]]) -> List[MintermDict]:
    """Create a dictionary mapping state names to minterm activation patterns.
    
    Returns:
        Dict mapping state names to 6-dimensional minterm patterns
    """
    minterms: List[MintermDict] = []
    for i, transitions in enumerate(transition_data):
        for transition in transitions:
            relation_signal: NDArray[np.integer] = np.asarray(inputs[i], dtype=int)
            input_pattern: NDArray[np.integer] = np.asarray(states.get(transition[0]), dtype=int)
            output_state: NDArray[np.integer] = np.asarray(states.get(transition[1]), dtype=int)
            minterms.append({
                "relation_signal": relation_signal,
                "input_pattern": input_pattern,
                "output_state": output_state
            })

    return minterms

def _validate_minterm_data(minterms: List[MintermDict]) -> None:
    """Validate that minterm data is consistent."""
    if not minterms:
        raise ValueError("Empty minterm list provided")
    
    # Check consistent dimensions
    first_signal_size = len(minterms[0]['relation_signal'])
    first_state_size = len(minterms[0]['input_pattern'])
    
    for i, minterm in enumerate(minterms):
        if len(minterm['relation_signal']) != first_signal_size:
            raise ValueError(f"Inconsistent signal size in minterm {i}")
        if len(minterm['input_pattern']) != first_state_size:
            raise ValueError(f"Inconsistent state size in minterm {i}")
        if len(minterm['output_state']) != first_state_size:
            raise ValueError(f"Inconsistent output state size in minterm {i}")

def _extract_dimensions(minterms: List[MintermDict]) -> Tuple[int, int, int]:
    """Extract key dimensions from minterm data."""
    num_minterms = len(minterms)
    input_size = len(minterms[0]['relation_signal'])  # e.g., 3 for [0,0,1]
    state_size = len(minterms[0]['input_pattern'])    # e.g., 100 for state vector
    return num_minterms, input_size, state_size

def _generate_minterm_weights_and_thresholds(minterms: List[MintermDict], num_minterms: int, input_size: int, state_size: int) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Generate weights and thresholds for minterm layer."""
    lower_weights = np.zeros((num_minterms, input_size))
    upper_weights = np.zeros((num_minterms, state_size))
    thresholds = np.zeros(num_minterms)
    
    for i, minterm in enumerate(minterms):
        # Set weights for relation signal matching
        lower_weights[i, :] = minterm['relation_signal']
        
        # Set weights for state pattern matching  
        upper_weights[i, :] = minterm['input_pattern']
        
        # Set threshold: require BOTH state AND signal to match
        state_norm_sq = np.linalg.norm(minterm['input_pattern']) ** 2
        signal_norm_sq = np.linalg.norm(minterm['relation_signal']) ** 2
        thresholds[i] = state_norm_sq + signal_norm_sq - 0.5
    
    return lower_weights, upper_weights, thresholds

def _calculate_state_thresholds(minterms: List[MintermDict], num_minterms: int, state_size: int) -> NDArray[np.floating]:
    """
    Calculate state layer thresholds.
    
    Note: Thresholds are set for apical transitions only. With the current minterm logic, the apical transition will look like output = state_c - sum(state_i) - offset. Here state_c is the wanted state from minterm_c while the sum is over all other minterms. If we want to have just the wanted state we want offset = - sum(state_i). This would work for a specific state_c perfectly. Since we are dealing with random vectors we can assume that the average offset for every wanted state will work for every computation. Therefore we calculated every offset_c and average them.
    """
    state_thresholds = np.zeros(state_size)
    
    # Sum contributions from all other minterms (unclear why this is needed)
    for i in range(num_minterms):
        offset_c = np.zeros(state_size)
        for j in range(num_minterms):
            if i != j:
                offset_c -= minterms[j]['output_state']
        state_thresholds += offset_c
    
    # Normalize by negative count
    state_thresholds /= num_minterms
    
    return state_thresholds

def _generate_state_weights_and_thresholds(minterms: List[MintermDict], num_minterms: int, state_size: int) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Generate weights and thresholds for state layer."""
    lower_state_weights = np.zeros((state_size, num_minterms))
    
    # Set state weights: each column maps to output state for that minterm
    for i, minterm in enumerate(minterms):
        lower_state_weights[:, i] = minterm['output_state']
    
    # Calculate state thresholds (this logic needs clarification)
    state_thresholds = _calculate_state_thresholds(minterms, num_minterms, state_size)
    
    return lower_state_weights, state_thresholds

def generate_minterm_weights(minterms: List[MintermDict]) -> WeightDict:
    """
    Generate all weights for minterm layer and state layer connections.
    
    Args:
        minterms: List of minterm dictionaries from create_minterm_data()
    
    Returns:
        Dictionary containing:
        - lowerWeights: Minterm layer weights from input layer
        - upperWeights: Minterm layer weights from state layer  
        - thresholds: Minterm layer thresholds
        - lowerStateWeights: State layer weights from minterm layer
        - stateThresholds: State layer thresholds
    """
    # Validate input data
    _validate_minterm_data(minterms)
    
    # Extract dimensions
    num_minterms, input_size, state_size = _extract_dimensions(minterms)
    
    # Generate minterm layer components
    lower_weights, upper_weights, thresholds = _generate_minterm_weights_and_thresholds(
        minterms, num_minterms, input_size, state_size
    )
    
    # Generate state layer components
    lower_state_weights, state_thresholds = _generate_state_weights_and_thresholds(
        minterms, num_minterms, state_size
    )
    
    return {
        "lowerWeights": lower_weights,
        "upperWeights": upper_weights,
        "thresholds": thresholds,
        "lowerStateWeights": lower_state_weights,
        "stateThresholds": state_thresholds
    }
