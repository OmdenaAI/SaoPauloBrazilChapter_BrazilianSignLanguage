import os
import csv
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from models.landmark.utils import load_config
from typing import Union, Dict


def train(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: Union[str, Dict],
):
    config = load_config(config, "training_config")
    device = config["device"]
    num_epochs = config["num_epochs"]

    model = model.to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    log_data = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for idx, batch in enumerate(train_loader):
            x = batch.squeeze(0).to(device)  # [T, D] or [B, T, D]
            if x.ndim == 2:
                x = x.unsqueeze(0)

            label = train_loader.dataset.data.iloc[idx]["label"]
            y = torch.tensor([label], dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ----- validation step -----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                x = batch.squeeze(0).to(device)
                if x.ndim == 2:
                    x = x.unsqueeze(0)
                label = test_loader.dataset.data.iloc[idx]["label"]
                y = torch.tensor([label], dtype=torch.long).to(device)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        log_data.append(
            {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
        )

        # ----- early stopping -----
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config['patience']}")
            if patience_counter >= config["patience"]:
                print("Early stopping triggered.")
                break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # ----- evaluate -----
    acc = evaluate(model, test_loader, device)

    print(f"\n Best Epoch: {best_epoch} | Test Accuracy: {acc:.4f}")

    # ----- optional logging to file -----
    if "log_path" in config:
        log_path = config["log_path"]
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(log_data)
        print(f"Training log saved to: {log_path}")

    return acc, best_epoch, log_data


def evaluate(model, val_loader: torch.utils.data.DataLoader, device: str = "cuda"):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            x = batch.squeeze(0).to(device)
            if x.ndim == 2:
                x = x.unsqueeze(0)
            label = val_loader.dataset.data.iloc[idx]["label"]
            output = model(x)
            pred = torch.argmax(output, dim=1).item()

            y_true.append(label)
            y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
