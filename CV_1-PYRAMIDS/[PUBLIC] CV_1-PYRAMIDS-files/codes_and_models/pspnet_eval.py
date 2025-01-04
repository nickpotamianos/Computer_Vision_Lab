import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cityscapes_dataset import Cityscapes
from pspnet import PSPNet
import os
import cv2



def calculate_metrics(pred, target, num_classes):
    """
    Calculate mean IoU (Intersection over Union) and pixel accuracy
    Args:
        pred: Predicted labels
        target: Ground truth labels
        num_classes: Number of classes
    Returns:
        mean_iou: Mean IoU score
        pixel_acc: Pixel accuracy
    """
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))

    # Populate confusion matrix
    for t, p in zip(target.flatten(), pred.flatten()):
        if t < num_classes:  # Ignore index 255
            confusion_matrix[t, p] += 1

    # Calculate IoU for each class
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - intersection
    iou = intersection / (union + 1e-10)

    # Calculate mean IoU
    mean_iou = np.mean(iou)

    # Calculate pixel accuracy
    pixel_acc = np.sum(intersection) / (np.sum(confusion_matrix) + 1e-10)

    return mean_iou, pixel_acc, iou


def evaluate_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load validation dataset
    dataset_val = Cityscapes(
        split='val',
        data_root='/content/drive/MyDrive/CV_1-PYRAMIDS-files/cityscapes_dataset/',
        data_list='/content/drive/MyDrive/CV_1-PYRAMIDS-files/cityscapes_dataset/list/cityscapes/fine_val.txt'
    )

    # Create dataloader
    val_dataloader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4)

    # Initialize model
    num_classes = 35  # Number of classes in Cityscapes dataset
    model = PSPNet(layers=50, bins=(2, 3, 6, 8), dropout=0.1, classes=num_classes,
                   zoom_factor=8, use_ppm=True, pretrained=False).to(device)

    # Load trained weights
    model.load_state_dict(torch.load('train_epoch_200_CPU.pth'))
    model.eval()

    # Initialize metrics storage
    all_ious = []
    all_pixel_accs = []
    class_ious = np.zeros(num_classes)
    num_images = len(val_dataloader)

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_dataloader):
            # Forward pass
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # Get predictions
            _, predictions = outputs.max(1)

            # Calculate metrics
            mean_iou, pixel_acc, iou_per_class = calculate_metrics(
                predictions[0].cpu().numpy(),
                labels[0].cpu().numpy(),
                num_classes
            )

            # Store metrics
            all_ious.append(mean_iou)
            all_pixel_accs.append(pixel_acc)
            class_ious += iou_per_class

            if (i + 1) % 10 == 0:
                print(f'Processed {i + 1}/{num_images} images')

    # Calculate final metrics
    final_mean_iou = np.mean(all_ious)
    final_mean_pixel_acc = np.mean(all_pixel_accs)
    final_class_ious = class_ious / num_images

    # Calculate standard deviations
    iou_std = np.std(all_ious)
    pixel_acc_std = np.std(all_pixel_accs)

    print("\nFinal Results:")
    print(f"Mean IoU: {final_mean_iou:.4f} ± {iou_std:.4f}")
    print(f"Mean Pixel Accuracy: {final_mean_pixel_acc:.4f} ± {pixel_acc_std:.4f}")

    # Print per-class IoU
    print("\nPer-class IoU:")
    with open('cityscapes_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    for i, class_iou in enumerate(final_class_ious):
        if i < len(class_names):
            print(f"{class_names[i]}: {class_iou:.4f}")


if __name__ == '__main__':
    evaluate_model()