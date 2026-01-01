from pathlib import Path
from typing import List, Tuple

NUM_CLASSES = 10


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data-sets"


def _one_hot(label: int, num_classes: int = NUM_CLASSES) -> List[int]:
    if not (0 <= label < num_classes):
        raise ValueError(f"Label {label} out of range for {num_classes} classes")
    vec = [0 for _ in range(num_classes)]
    vec[label] = 1
    return vec


def _load_digit_path(file_path: Path) -> List[int]:
    """Load an 8x8 digit text file of 0/1 characters into a flat list of ints."""
    if not file_path.exists():
        raise FileNotFoundError(f"Digit file not found: {file_path}")

    digit: List[int] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            if any(ch not in {"0", "1"} for ch in stripped):
                raise ValueError(f"Invalid character in digit file: {file_path}")

            digit.extend([int(ch) for ch in stripped])

    if not digit:
        raise ValueError(f"Digit file is empty: {file_path}")

    return digit


def load_digit_file(filename: str | Path) -> List[int]:
    """
    Load a digit file from an absolute/relative path or from the bundled data dir.
    """
    candidate = Path(filename)
    if candidate.is_absolute():
        return _load_digit_path(candidate)

    if candidate.exists():
        return _load_digit_path(candidate)

    return _load_digit_path(_data_dir() / candidate)


def load_dataset() -> Tuple[List[List[int]], List[List[int]]]:
    """Return (images, hot_vectors) for digit_0.txt ... digit_9.txt."""
    images: List[List[int]] = []
    hot_vectors: List[List[int]] = []
    for digit in range(NUM_CLASSES):
        filename = f"digit-samples/digit_{digit}.txt"
        pixels = _load_digit_path(_data_dir() / filename)
        images.append(pixels)
        hot_vectors.append(_one_hot(digit))
    return images, hot_vectors
