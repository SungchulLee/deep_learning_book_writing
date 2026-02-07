"""
Hymenoptera — ImageFolder + ResNet Transfer Learning
=====================================================
Fine-tunes a pre-trained ResNet-18 on the Hymenoptera (ants vs bees) dataset
loaded via ``torchvision.datasets.ImageFolder``.

Usage
-----
    python hymenoptera.py
    python hymenoptera.py --epochs 10 --lr 0.001
"""

import argparse
import copy
import os
import time
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Hymenoptera Transfer Learning")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--test-batch-size", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--no-cuda", action="store_true", default=False)
parser.add_argument("--no-mps", action="store_true", default=False)
ARGS = parser.parse_args()

torch.manual_seed(ARGS.seed)

ARGS.use_cuda = not ARGS.no_cuda and torch.cuda.is_available()
ARGS.use_mps = not ARGS.no_mps and torch.backends.mps.is_available()
if ARGS.use_cuda:
    DEVICE = torch.device("cuda")
elif ARGS.use_mps:
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

TRAIN_KWARGS = {"batch_size": ARGS.batch_size}
TEST_KWARGS = {"batch_size": ARGS.test_batch_size}
if ARGS.use_cuda:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    TRAIN_KWARGS.update(cuda_kwargs)
    TEST_KWARGS.update(cuda_kwargs)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASSES = ("ants", "bees")
PATH = "./model.pth"

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_dataset():
    if os.path.isdir("./hymenoptera_data"):
        return
    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    zip_path = "hymenoptera_data.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")
    os.remove(zip_path)
    print("Hymenoptera dataset downloaded and extracted.")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data():
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
    }
    data_dir = "./hymenoptera_data"
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], **TRAIN_KWARGS)
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    return dataloaders, dataset_sizes, class_names


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def show_batch(dataloader, model=None):
    denorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(MEAN, STD)],
        std=[1 / s for s in STD],
    )
    images, labels = next(iter(dataloader))

    if model is not None:
        outputs = model(images.to(DEVICE))
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu()
    else:
        preds = labels

    n = min(ARGS.batch_size, 10)
    _, axes = plt.subplots(1, n, figsize=(12, 3))
    for i in range(n):
        img = denorm(images[i]).permute(1, 2, 0).numpy().clip(0, 1)
        axes[i].imshow(img)
        axes[i].axis("off")
        title = f"label: {CLASSES[labels[i]]}"
        if model is not None:
            title += f"\npred: {CLASSES[preds[i]]}"
        axes[i].set_title(title)
    plt.show()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False


def train(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes):
    since = time.time()
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(ARGS.epochs):
        print(f"Epoch {epoch}/{ARGS.epochs - 1}\n" + "-" * 10)
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(f"  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
        print()

    elapsed = time.time() - since
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")
    model.load_state_dict(best_wts)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def compute_accuracy(model, testloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, predicted = torch.max(model(images).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct // total} %")

    correct_pred = {c: 0 for c in CLASSES}
    total_pred = {c: 0 for c in CLASSES}
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, preds = torch.max(model(images), 1)
            for lab, pred in zip(labels, preds):
                if lab == pred:
                    correct_pred[CLASSES[lab]] += 1
                total_pred[CLASSES[lab]] += 1
    for cls, cnt in correct_pred.items():
        print(f"  class {cls:>5s}: {100 * cnt / total_pred[cls]:.1f} %")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    download_dataset()
    dataloaders, dataset_sizes, class_names = load_data()

    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    freeze_backbone(model)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=ARGS.lr, momentum=ARGS.momentum)
    exp_lr = lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)

    train(model, criterion, opt, exp_lr, dataloaders, dataset_sizes)

    torch.save(model.state_dict(), PATH)
    loaded = models.resnet18()
    loaded.fc = nn.Linear(loaded.fc.in_features, 2)
    loaded = loaded.to(DEVICE)
    loaded.load_state_dict(torch.load(PATH))

    show_batch(dataloaders["val"], loaded)
    compute_accuracy(loaded, dataloaders["val"])


if __name__ == "__main__":
    main()
