# PyTorch MNIST Digit Classifier

I wrote this to get comfortable with the PyTorch training loop, so I kept it small: one file and nothing hidden. It trains a tiny neural network on MNIST and lets you test it on your own handwritten digits.

## Quick start

```bash
git clone https://github.com/aghojogh/pytorch_minst.git
cd pytorch_minst
pip install -r requirements.txt
python mnist.py
```

The first run downloads MNIST into a `data/` folder, trains for 3 epochs, prints the test accuracy after each one, and saves the weights to `mnist_model.pt`. It uses your GPU if you have one and the CPU otherwise.

On Windows, if `pip` isn't recognized, use `python -m pip install -r requirements.txt` instead.

## Training options

You can change the training settings from the command line:

```bash
python mnist.py train --epochs 5 --lr 0.001 --batch-size 128
```

- `--epochs`: how many passes over the training data (default 3)
- `--lr`: learning rate (default 0.001)
- `--batch-size`: images per training step (default 64)

## Try it on your own digit

Draw a digit, save it as an image, and run:

```bash
python mnist.py predict my_digit.png
```

MNIST digits are white on a black background. If your image is black ink on white paper, the model will struggle, so invert it first. The script handles grayscale and resizing to 28x28, but it can't fix the colors.

## How it works

The model is two linear layers with a ReLU in between. It flattens the 28x28 image into 784 numbers, passes them through a hidden layer of 128 neurons, and ends with 10 outputs, one score per digit. The highest score is the prediction.

Training uses cross-entropy loss and the Adam optimizer.

Everything is in `mnist.py`, split into a few small functions: `get_loaders`, `build_model`, `evaluate`, `train`, and `predict`. It's short enough to read from top to bottom.

## Things I might try next

- Swap in a small CNN
- Add data augmentation
- Plot the loss curve
- Try Fashion-MNIST

## License

MIT, see [LICENSE](LICENSE).
