import os
import sys
import time
import json
import argparse
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_cat_dog_data
from model import build_model

class EarlyStopping:
    """
    Early stopping to stop the training when validation loss stops improving.
    Matches the Keras EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    shown in the notebook image.
    """
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"--> EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0

def plot_training_results(history, save_path="training_results.png"):
    """
    Plots training and validation accuracy and loss side-by-side,
    matching the exact visual format in the notebook screenshot.
    """
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs, history["train_acc"], label="Training Accuracy", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_acc"], label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    ax1.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True)

    # Loss Plot
    ax2.plot(epochs, history["train_loss"], label="Training Loss", color="#1f77b4", linewidth=2)
    ax2.plot(epochs, history["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2)
    ax2.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Results plot saved to: {save_path}")

def train_model(data_dir="archive/animals",
                backbone="mobilenet_v2",
                epochs=6,
                batch_size=32,
                learning_rate=0.001,
                patience=3,
                output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("   CAT VS DOG CLASSIFIER - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Backbone: {backbone}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {learning_rate}")

    # 1. Loading and Preprocessing Data
    print("\n[Step 1/5] Loading and splitting dataset...")
    train_loader, val_loader, classes, class_to_idx = load_cat_dog_data(
        data_dir=data_dir,
        img_size=224,
        batch_size=batch_size,
        train_split=0.8,
        seed=42
    )
    print(f"Classes: {classes} (mapping: {class_to_idx})")
    print(f"Training samples: {len(train_loader.dataset)} (batches: {len(train_loader)})")
    print(f"Validation samples: {len(val_loader.dataset)} (batches: {len(val_loader)})")

    # 2. Build Model
    print(f"\n[Step 2/5] Building {backbone} transfer learning model...")
    model = build_model(backbone=backbone, num_classes=len(classes), pretrained=True, freeze_base=True)
    model = model.to(device)

    # 3. Loss & Optimizer (Compile step from image)
    print("\n[Step 3/5] Compiling model (Loss: CrossEntropy, Optimizer: Adam)...")
    criterion = nn.CrossEntropyLoss()
    # Only train parameters that require grad (the custom classifier head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)

    early_stopping = EarlyStopping(patience=patience)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0

    # 4. Training loop with Early Stopping
    print(f"\n[Step 4/5] Training model with Early Stopping (patience={patience})...\n")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc

        print(f"Epoch {epoch:02d}/{epochs:02d} - loss: {epoch_train_loss:.4f} - accuracy: {epoch_train_acc:.4f} "
              f"- val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_acc:.4f}")

        # Check early stopping
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}!")
            break

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f}s.")

    # Restore best model weights
    if early_stopping.best_model_weights is not None:
        model.load_state_dict(early_stopping.best_model_weights)
        print("Restored best model weights from early stopping checkpoint.")

    # 5. Final Accuracy & Visualizations (Matching notebook output)
    print("\n" + "=" * 60)
    print("Printing Final Training Accuracy and Validation Accuracy")
    print("=" * 60)
    final_train_acc = history["train_acc"][-1]
    final_val_acc = history["val_acc"][-1]
    print(f"Final Training Accuracy:   {final_train_acc * 100:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc * 100:.2f}%")
    print(f"Best Validation Accuracy:  {best_val_acc * 100:.2f}%")
    print("=" * 60)

    # Save model checkpoint
    model_save_path = os.path.join(output_dir, "best_cat_dog_model.pth")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "backbone": backbone,
        "best_val_acc": best_val_acc,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "history": history
    }
    torch.save(checkpoint, model_save_path)
    print(f"Model checkpoint saved to: {model_save_path}")

    # Save history json
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "classes": classes,
            "backbone": backbone,
            "final_train_acc": final_train_acc,
            "final_val_acc": final_val_acc,
            "best_val_acc": best_val_acc,
            "history": history
        }, f, indent=4)

    # Save visual comparison plots
    plot_path = os.path.join(output_dir, "training_results.png")
    plot_training_results(history, save_path=plot_path)

    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cat vs Dog Classifier")
    parser.add_argument("--data_dir", type=str, default="archive/animals", help="Path to animals dataset directory")
    parser.add_argument("--backbone", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "vgg16", "custom_cnn"], help="Backbone model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        output_dir=args.output_dir
    )
