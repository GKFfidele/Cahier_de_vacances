import os
from collections import Counter

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision

SPECIES = ['AngelFish', 'BlueTang', 'ButterflyFish', 'ClownFish', 'GoldFish', 'Gourami',
           'MorishIdol', 'PlatyFish', 'RibbonedSweetlips', 'ThreeStripedDamselfish',
           'YellowCichlid', 'YellowTang', 'ZebraFish']


# --------------------------------------------------------------------------------------
# Reading the dataset (YOLO format)
# --------------------------------------------------------------------------------------

def list_images(dataset_dir, split):
    """List all image paths of a split.

    Arguments:
    dataset_dir -- root folder of the Kaggle dataset
    split -- "train", "valid" or "test"
    """
    image_dir = os.path.join(dataset_dir, split, "images")
    return sorted(os.path.join(image_dir, name) for name in os.listdir(image_dir))


def label_path_for(image_path):
    """Return the path of the YOLO label file matching an image path."""
    label_path = image_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
    return os.path.splitext(label_path)[0] + ".txt"


def read_yolo_boxes(label_path, width, height):
    """Parse a YOLO label file into pixel-coordinate boxes and species indices.

    A YOLO line is "class x_center y_center width height", the last four numbers being
    fractions of the image size. They are converted here into [x1, y1, x2, y2] pixels.

    Arguments:
    label_path -- path to a YOLO .txt label file
    width, height -- pixel dimensions of the corresponding image

    Returns:
    boxes -- list of [x1, y1, x2, y2] pixel coordinates, one per line
    labels -- list of species indices (0 to 12), one per line, same order as boxes
    """
    boxes, labels = [], []
    for line in open(label_path):
        parts = line.split()
        if len(parts) != 5:
            continue
        class_idx, xc, yc, w, h = parts
        xc, yc, w, h = float(xc), float(yc), float(w), float(h)
        boxes.append([(xc - w / 2) * width, (yc - h / 2) * height,
                      (xc + w / 2) * width, (yc + h / 2) * height])
        labels.append(int(class_idx))
    return boxes, labels


def dominant_species(label_path):
    """Return the name of the most frequent species in a YOLO label file."""
    _, class_indices = read_yolo_boxes(label_path, 1, 1)
    if not class_indices:
        return None
    most_common_id, _ = Counter(class_indices).most_common(1)[0]
    return SPECIES[most_common_id]


def build_fish_dataframe(dataset_dir):
    """Build a DataFrame with one row per labelled photo of the dataset.

    Arguments:
    dataset_dir -- root folder of the Kaggle dataset

    Returns:
    fish_df -- DataFrame with columns image_path, species (the dominant one) and split
    """
    rows = []
    for split in ("train", "valid", "test"):
        for image_path in list_images(dataset_dir, split):
            label_path = label_path_for(image_path)
            if not os.path.exists(label_path):
                continue
            species = dominant_species(label_path)
            if species is not None:
                rows.append({"image_path": image_path, "species": species, "split": split})
    return pd.DataFrame(rows)


def collate_fn(batch):
    """Group a list of (image, target) pairs into two tuples: images stay a list, since detection
    images and their number of boxes can differ from one sample to the next (no stacking possible)."""
    return tuple(zip(*batch))


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

def compute_map(model, data_loader):
    """Compute the mean Average Precision of a detector over a DataLoader.

    Arguments:
    model -- a torchvision detection model
    data_loader -- a DataLoader yielding (images, targets), as built with collate_fn

    Returns:
    results -- dict of metrics, including "map" and "map_50"
    """
    metric = MeanAveragePrecision(box_format="xyxy")
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            predictions = model(list(images))
            metric.update(list(predictions), list(targets))
    return metric.compute()


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

def plot_boxes(image, boxes, labels, scores=None, ax=None, title=None, box_color="lime"):
    """Draw an image with bounding boxes and species labels on top.

    Arguments:
    image -- a PIL image
    boxes -- list of boxes, each as (x1, y1, x2, y2) in pixel coordinates
    labels -- list of species names, one per box
    scores -- optional list of confidence scores (between 0 and 1), one per box
    ax -- matplotlib axis to draw on (creates a new one if None)
    title -- optional title for the axis
    box_color -- color of the boxes and their text labels
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        linewidth=2, edgecolor=box_color, facecolor="none"))
        text = label if scores is None else f"{label} {scores[i]:.2f}"
        ax.text(x1, max(y1 - 4, 0), text, color="black", fontsize=8,
                 bbox=dict(facecolor=box_color, edgecolor="none", pad=1))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)


def _make_grid(n_images, n_cols):
    """Create a grid of axes large enough for n_images, and return it as a flat list."""
    n_rows = (n_images + n_cols - 1) // n_cols
    _, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    axes = [ax for row in axes for ax in row]
    for ax in axes[n_images:]:
        ax.axis("off")
    return axes


def plot_species_counts(fish_df, split="train"):
    """Plot the number of photos per species in one split of the dataset.

    Arguments:
    fish_df -- DataFrame as returned by build_fish_dataframe
    split -- which split to count ("train", "valid" or "test")
    """
    counts = fish_df[fish_df["split"] == split]["species"].value_counts()
    counts.sort_values().plot(kind="barh", color="teal", figsize=(7, 5))
    plt.title(f"Nombre de photos par espèce ({split})")
    plt.xlabel("Nombre de photos")
    plt.tight_layout()
    plt.show()
    return counts


def plot_ground_truth_grid(image_paths, n_cols=3):
    """Plot a grid of photos with their real (hand-annotated) boxes and species.

    Arguments:
    image_paths -- list of image file paths
    n_cols -- number of columns of the grid
    """
    axes = _make_grid(len(image_paths), n_cols)
    for ax, image_path in zip(axes, image_paths):
        image = Image.open(image_path).convert("RGB")
        boxes, labels = read_yolo_boxes(label_path_for(image_path), image.width, image.height)
        plot_boxes(image, boxes, [SPECIES[i] for i in labels], ax=ax)
    plt.tight_layout()
    plt.show()


def plot_predictions_grid(image_paths, predict_fn, n_cols=3):
    """Plot a grid of photos with the boxes predicted by a model.

    The title of each photo is green when the most confident predicted species matches the
    real dominant species of the photo, red otherwise.

    Arguments:
    image_paths -- list of image file paths
    predict_fn -- function taking an image path and returning (boxes, labels, scores)
    n_cols -- number of columns of the grid
    """
    axes = _make_grid(len(image_paths), n_cols)
    for ax, image_path in zip(axes, image_paths):
        boxes, labels, scores = predict_fn(image_path)
        true = dominant_species(label_path_for(image_path))
        if labels:
            predicted = labels[max(range(len(scores)), key=scores.__getitem__)]
        else:
            predicted = "aucune détection"
        plot_boxes(Image.open(image_path).convert("RGB"), boxes, labels, scores, ax=ax)
        ax.set_title(f"Prédit : {predicted}\nVrai : {true}", fontsize=9,
                     color="green" if predicted == true else "red")
    plt.tight_layout()
    plt.show()


def plot_loss_curves(loss_history, num_epochs):
    """Plot several named training loss curves on the same figure.

    Arguments:
    loss_history -- dict mapping a loss name to a list of num_epochs values
    num_epochs -- number of epochs (x-axis)
    """
    plt.figure(figsize=(8, 5))
    for name, values in loss_history.items():
        plt.plot(range(1, num_epochs + 1), values, marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Courbes de loss (entraînement)")
    plt.legend()
    plt.tight_layout()
    plt.show()
