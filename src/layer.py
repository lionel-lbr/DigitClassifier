import math
import random
from typing import List


def init_weights(n_neurons: int, n_inputs: int) -> tuple[list[list[float]], list[float]]:
    """Return (weights, biases) with Xavier/Glorot init for a single layer."""
    std = math.sqrt(1 / n_inputs)
    W: List[List[float]] = []
    for _ in range(n_neurons):
        row = [random.gauss(0, std) for _ in range(n_inputs)]
        W.append(row)
    b = [0.0 for _ in range(n_neurons)]
    return W, b


def forward_pass(x: List[int], W: List[List[float]], b: List[float]) -> List[float]:
    """Compute one-layer outputs: weighted sum of inputs plus bias per neuron."""
    if not W:
        return []

    n_inputs = len(W[0])
    if len(x) != n_inputs:
        raise ValueError(f"Expected input length {n_inputs}, got {len(x)}")

    outputs: List[float] = []
    for weights, bias in zip(W, b):
        if len(weights) != n_inputs:
            raise ValueError("Inconsistent weight vector length")
        activation = sum(w * xi for w, xi in zip(weights, x)) + bias
        outputs.append(activation)

    return outputs


def softmax(logits: List[float]) -> List[float]:
    """Convert raw scores to probabilities in a numerically stable way."""
    if not logits:
        return []

    shift = max(logits)
    exps = [math.exp(x - shift) for x in logits]
    denom = sum(exps)
    return [val / denom for val in exps]
