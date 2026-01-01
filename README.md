# Toy Digit Classifier

Lightweight 8×8 digit classifier implemented from scratch in Python. It uses a single-layer softmax model trained on the bundled `data-sets/digit-samples/digit_*.txt` bitmaps and provides a small CLI to train and predict.

## Setup

- Requirements: Python 3.9+ (no external dependencies).
- Clone and enter the project:
  ```bash
  git clone https://github.com/lionel-lbr/toy-digit-classifier.git && cd toy-digit-classifier
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

## Network topology and limitations

The model is a single-layer perceptron (logistic regression) with 64 inputs—one per pixel of the 8×8 bitmap—and 10 output neurons, one per digit class.

A forward pass computes logits `z = Wx + b` and applies softmax, `softmax(z_i) = exp(z_i) / sum_j exp(z_j)`, to obtain a probability distribution.

Targets are one-hot vectors of length 10.

Training consists of feeding each input through the forward pass, computing softmax probabilities, comparing them to the one-hot target, and applying the gradient update; one full sweep over the 10 training samples constitutes an epoch.

The loss minimized is cross-entropy, optimized with stochastic gradient descent; for softmax + cross-entropy the gradient with respect to each logit is `(probs - one_hot)`, yielding weight updates `w_ij -= lr * (probs_i - target_i) * x_j` and bias updates `b_i -= lr * (probs_i - target_i)`. The loss is calculated on a whole epoch and used to decide when to stop training.

Early stopping monitors average loss with a patience of five epochs by default. Although training accuracy typically reaches 100% after roughly 50 epochs, additional epochs (often into the hundreds) reduce the loss further before patience halts training.

Because the dataset contains only one exemplar per digit, the model largely memorizes these patterns; it will often classify heavily degraded versions of the provided digits (many 1s flipped to 0s) as the originals.

For example try the following command they'll all output 4.

```bash
python src/main.py predict --model model.json --input digit_4_unknown_1.txt --render
```

```bash
python src/main.py predict --model model.json --input digit_4_unknown_2.txt --render
```

```bash
python src/main.py predict --model model.json --input digit_4_unknown_3.txt --render
```

## Future evolutions

To correctly handle degraded digits (and move beyond memorization), the model needs more capacity and richer feature extraction. Two natural directions:

- Introduce hidden layers and treat the network as a small MLP (e.g., Multilayer Perceptron with one or two dense layers with nonlinear activations) so it can learn abstractions beyond the raw pixel patterns.
- Move to a simple CNN (Convolutional Neural Network) with convolution + pooling blocks to capture local spatial structure and invariance to small degradations or noise.\*\*\*
