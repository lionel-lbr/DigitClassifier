# Toy Digit Classifier

Lightweight 8×8 digit classifier implemented from scratch in Python. It uses a single-layer softmax model trained on the bundled `data-sets/digit-samples/digit_*.txt` bitmaps and provides a small CLI to train and predict.

## Setup

- Requirements: Python 3.9+ (no external dependencies).
- Clone and enter the project:
  ```bash
  git clone https://github.com/lionel-lbr/DigitClassifier.git && cd DigitClassifier
  ```
- (Optional) Use a virtual environment if you want to keep things isolated.

## Usage

Train on the bundled digits and save a model (you can choose any path, e.g. repo root):

```bash
python src/main.py train --out model.json
```

Optional training controls:

- Limit epochs: `--max-epochs 300` (default: 1000)
- Early-stop patience: `--patience 10` (default: 5)

Predict a digit using a saved model and an 8×8 text file of 0/1 values:

```bash
python src/main.py predict --model model.json --input digit-samples/digit_4.txt
```

Render the input before predicting:

```bash
python src/main.py predict --model model.json --input digit-samples/digit_4.txt --render
```

Notes:

- The `--input` file can be any path; relative paths are looked up first as-given and then within `data-sets/`.
- Model files are JSON and store `n_neurons`, `n_inputs`, `weights`, and `bias`.
- The training script prints loss progress and final training accuracy.
