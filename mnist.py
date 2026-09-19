
"""
MNIST handwritten digit classifier built with PyTorch (single-file project).

Usage:
    python mnist.py                      # train with default settings
    python mnist.py train --epochs 5 --lr 0.001 --batch-size 128
    python mnist.py predict my_digit.png # classify your own image
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MODEL_PATH = "mnist_model.pt"


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
def get_loaders(batch_size=64):
    """Build DataLoaders for the MNIST training and test sets."""
    tfm = transforms.ToTensor()  # image -> tensor, pixel values scaled to [0, 1]

    train_ds = datasets.MNIST("data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tfm)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------
def build_model():
    """A small fully connected network for 28x28 grayscale digits."""
    return nn.Sequential(
        nn.Flatten(),             # (1, 28, 28) -> 784
        nn.Linear(28 * 28, 128),  # hidden layer
        nn.ReLU(),                # non-linearity
        nn.Linear(128, 10),       # output layer: one score per digit 0-9
    )


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
def evaluate(model, dataloader, device):
    """Return the model's accuracy (0 to 1) on the given dataloader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------
def train(epochs, lr, batch_size):
    """Train the model, report test accuracy each epoch, and save the weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_dl, test_dl = get_loaders(batch_size)
    model = build_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        acc = evaluate(model, test_dl, device)
        print(f"epoch {epoch + 1}, loss = {loss.item():.4f}, test acc = {acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
def predict(image_path):
    """Classify a single image using the saved model."""
    from PIL import Image  # only needed for prediction

    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # The image should show a light digit on a dark background (like MNIST).
    tfm = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    x = tfm(Image.open(image_path)).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    print(f"Predicted digit: {probs.argmax().item()} (confidence {probs.max().item():.2%})")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MNIST digit classifier with PyTorch")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="train the model")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--batch-size", type=int, default=64)

    p_pred = sub.add_parser("predict", help="classify an image")
    p_pred.add_argument("image", help="path to the image file")

    args = parser.parse_args()

    if args.command is None:
        train(epochs=3, lr=1e-3, batch_size=64)  # default: train
    elif args.command == "train":
        train(args.epochs, args.lr, args.batch_size)
    else:
        predict(args.image)


if __name__ == "__main__":
    main()