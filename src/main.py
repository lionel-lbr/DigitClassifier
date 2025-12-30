from typing import List
import random
import math

from data import NUM_CLASSES, load_dataset
from layer import init_weights, forward_pass, softmax


def cross_entropy_loss(target: List[int], probs: List[float]) -> float:
    """
    Compute cross-entropy loss for a one-hot target vector and predicted probs.

    Expected: target length equals probs length; target contains a single 1.
    """
    if len(target) != len(probs):
        raise ValueError("Target and probability vectors must have same length")

    epsilon = 1e-12  # protect log
    loss = 0.0
    for t, p in zip(target, probs):
        if t not in (0, 1):
            raise ValueError("Target must be one-hot encoded (0 or 1 values)")
        loss -= t * math.log(max(p, epsilon))
    return loss


def predict_proba(x: List[int], W: List[List[float]], b: List[float]) -> List[float]:
    """Return class probabilities for a single input."""
    logits = forward_pass(x, W, b)
    return softmax(logits)


def predict_digit(x: List[int], W: List[List[float]], b: List[float]) -> int:
    """Return the predicted class index for a single input."""
    probs = predict_proba(x, W, b)
    if not probs:
        raise ValueError("No probabilities computed for prediction")
    return max(range(len(probs)), key=probs.__getitem__)


def gradient_step(
    x: List[int], target: List[int], W: List[List[float]], b: List[float], lr: float
) -> float:
    """
    Single SGD step: forward, loss, then weight/bias update using softmax + cross-entropy.

    Updates W and b in-place. Returns the scalar loss for logging.
    """
    logits = forward_pass(x, W, b)
    probs = softmax(logits)
    loss = cross_entropy_loss(target, probs)

    # For softmax + cross-entropy, dL/dlogits = probs - target
    delta = [p - t for p, t in zip(probs, target)]

    # Update weights and biases
    for i, (d, weights) in enumerate(zip(delta, W)):
        for j, xj in enumerate(x):
            weights[j] -= lr * d * xj
        b[i] -= lr * d

    return loss


def render_image(img: List[int]) -> str:
    """Render a flat 64-length image into 8 lines of '0'/'#' characters."""
    if len(img) != 64:
        raise ValueError(f"Expected 64-length image, got {len(img)}")
    lines = ["  01234567"]
    index = 0
    for i in range(0, 64, 8):
        row = "".join("#" if px else " " for px in img[i : i + 8])
        lines.append(f"{index}: {row}")
        index += 1
    print("\n".join(lines))


def main() -> None:
    """Train a single-layer softmax classifier."""
    images, hot_vectors = load_dataset()
    if not images:
        raise ValueError("No digit data available to train on")

    n_inputs = len(images[0])
    W, b = init_weights(NUM_CLASSES, n_inputs)
    learning_rate = 0.1
    best_train_loss = float("inf")
    patience = 5
    min_delta = 1e-4
    wait = 0
    max_epochs = 1000

    train_indices = list(range(len(images)))

    for epoch in range(1, max_epochs + 1):
        random.shuffle(train_indices)

        train_losses = []
        for idx in train_indices:
            digit = images[idx]
            hot_vector = hot_vectors[idx]
            loss = gradient_step(digit, hot_vector, W, b, learning_rate)
            train_losses.append(loss)

        avg_train_loss = sum(train_losses) / len(train_losses)
        improved = avg_train_loss < best_train_loss - min_delta
        if improved:
            best_train_loss = avg_train_loss
            wait = 0
        else:
            wait += 1

        print(
            f"Epoch {epoch:4d} | avg_loss={avg_train_loss:.6f} "
            f"| best={best_train_loss:.6f} | wait={wait}"
        )

        if wait >= patience:
            print("Stopping: no improvement on training loss")
            break

    correct = 0
    for digit, hot_vector in zip(images, hot_vectors):
        prediction = predict_digit(digit, W, b)
        label_idx = hot_vector.index(1)
        if prediction == label_idx:
            correct += 1
    accuracy = correct / len(images)
    print(f"Training set accuracy: {accuracy:.2f}")
    print("Training complete.")


if __name__ == "__main__":
    main()
