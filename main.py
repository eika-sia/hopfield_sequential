import numpy as np
import logging
from src.state_generator import generate_states
from sequential import SequentialNetwork

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# Parameters
N = 100  #* Dimension of vectors
K = 5    #* Number of vectors to find
ORTHOGONALITY_THRESHOLD = 0  #* Maximum allowed dot product

#? State generator - Find K N-dimensional almost orthogonal vectors
#? Convert to numpy array for easier manipulation
states = np.array(generate_states(N, K, ORTHOGONALITY_THRESHOLD))
log.debug(f"Final shape: {states.shape}")

STATES_INTERNAL = {
    "maomao": states[0],
    "lakan": states[1],
    "jinshi": states[2],
    "gyokugou": states[3],
    "emporor": states[4],
}

#? Transition initialization
FATHER_OF = [("maomao", "lakan"), ("jinshi", "emporor")]
LIKES = [("jinshi", "maomao"), ("emporor", "gyokugou"), ("gyokugou", "maomao"), ("maomao", "gyokugou"), ("lakan", "maomao")]
BULLIES = [("gyokugou", "jinshi"), ("maomao", "jinshi"), ("lakan", "jinshi")]

input_signals = ["FATHER_OF", "LIKES", "BULLIES"]

sequential_network = SequentialNetwork(
    states=STATES_INTERNAL,
    transitions=[FATHER_OF, LIKES, BULLIES],
    input_strings=input_signals
)

while True:
    output = sequential_network.process_input()
    print(output)
