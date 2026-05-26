import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from inference.pavement_classifier import build_model, select_device


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pavement wet/dry classifier")
    parser.add_argument("--data-dir", default="data", help="Dataset with train/dry, train/wet, val/dry, val/wet")
    parser.add_argument("--output", default="data/models/pavement-wetdry-efficientnet-b0.pt")
    parser.add_argument("--architecture", default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_large", "resnet18"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device(select_device())
    train_loader, val_loader = build_loaders(Path(args.data_dir), args.batch_size)
    model = build_model(args.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, val_loader, device)
        print(f"epoch={epoch + 1} val_f1={metrics['f1']:.3f} precision={metrics['precision']:.3f} recall={metrics['recall']:.3f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "architecture": args.architecture, "classes": ["dry", "wet"]}, output)

    print(f"best val_f1={best_f1:.3f}; wrote {args.output}")


def build_loaders(data_dir: Path, batch_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_transform)
    if train_ds.class_to_idx != {"dry": 0, "wet": 1}:
        raise SystemExit(f"Expected classes dry=0, wet=1; got {train_ds.class_to_idx}")
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2), DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)


def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            labels = labels.cpu()
            tp += int(((predictions == 1) & (labels == 1)).sum())
            fp += int(((predictions == 1) & (labels == 0)).sum())
            fn += int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    main()
