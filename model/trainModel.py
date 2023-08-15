import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from preprocess.train import *
from preprocess.test import *
from plots.plots import *
from model.Unet import *
from model.LossFunction import *
from model.IoU_Score import *
from model.trainModel import *

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=5, device="cuda"):
    best_iou = 0.0
    best_model_weights = model.state_dict()

    train_losses = []
    train_ious = []
    train_accuracies = []
    val_losses = []
    val_ious = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_accuracy = 0.0
        num_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            pred_labels = (torch.sigmoid(outputs) > 0.5).float()

            running_iou += iou_score(pred_labels, labels)
            running_accuracy += accuracy_score(labels.cpu().numpy().flatten(), pred_labels.cpu().numpy().flatten())

        avg_train_loss = running_loss / len(train_loader.dataset)
        avg_train_iou = running_iou / num_batches
        avg_train_accuracy = running_accuracy / num_batches

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1).float()

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                pred_labels = (torch.sigmoid(outputs) > 0.5).float()

                val_iou += iou_score(pred_labels, labels)
                val_accuracy += accuracy_score(labels.cpu().numpy().flatten(), pred_labels.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)

        # Save metrics for plotting
        train_losses.append(avg_train_loss)
        train_ious.append(avg_train_iou)
        train_accuracies.append(avg_train_accuracy)
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        val_accuracies.append(avg_val_accuracy)

        # Print metrics for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} Train IoU: {avg_train_iou:.4f} Train Accuracy: {avg_train_accuracy:.4f} "
              f"Val Loss: {avg_val_loss:.4f} Val IoU: {avg_val_iou:.4f} Val Accuracy: {avg_val_accuracy:.4f}")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            best_model_weights = model.state_dict()

        scheduler.step()

    model.load_state_dict(best_model_weights)
    return model, train_losses, train_ious, train_accuracies, val_losses, val_ious, val_accuracies
