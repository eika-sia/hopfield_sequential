import numpy as np
import logging
from src.utils import are_almost_orthogonal
from typing import List
from numpy.typing import NDArray

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

def generate_states(N: int, K: int, orthogonality_threshold: float) -> List[NDArray[np.floating]]:
    if N <= 0 or K <= 0:
        raise ValueError("N and K must be positive")
    if orthogonality_threshold < 0:
        raise ValueError("Threshold must be non-negative")

    log.info(f"Searching for {K} almost orthogonal {N}-dimensional bipolar vectors...")
    log.info(f"Orthogonality threshold: {orthogonality_threshold}")

    vectors: list[np.ndarray] = []
    attempts = 0

    for i in range(K):
        log.debug(f"Generating vector {i+1}/{K}...")
        found_valid = False
        
        while not found_valid:
            attempts += 1
            #? Generate random bipolar vector (-1, 1)
            new_vector = np.random.choice([-1, 1], N)

            #? Check orthogonality with existing vectors
            candidate_vectors = vectors + [new_vector]
            
            if are_almost_orthogonal(candidate_vectors, orthogonality_threshold):
                vectors.append(new_vector)
                found_valid = True
                log.debug(f"Found valid vector {i+1} after {attempts} total attempts")
            
            #if attempts % 1000 == 0:
                #log.info(f"Still searching... {attempts} attempts so far")

    log.info(f"Successfully found {K} almost orthogonal vectors!")
    log.debug(f"Total attempts: {attempts}")

    #? Verify orthogonality
    log.debug("Verification - Dot products between vectors:")
    for i in range(K):
        for j in range(i + 1, K):
            dot_prod = np.dot(vectors[i], vectors[j])
            log.debug(f"Vector {i+1} · Vector {j+1} = {dot_prod}")
    return vectors
