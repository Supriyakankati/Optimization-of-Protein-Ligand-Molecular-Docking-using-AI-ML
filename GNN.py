# -*- coding: utf-8 -*-
"""Model Training and Evaluation Module"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, LayerNorm
from sklearn.metrics import roc_curve, precision_recall_curve
from data_processing import load_data_split, set_seeds
from plots import plot_metrics_and_curves
from gnn_model import GNNModel

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    """
    Trains the GNN model and evaluates it on validation data.
    """
    train_losses, val_losses, train_acc, val_acc, train_rmsd, val_rmsd = [], [], [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total, rmsd = 0., 0., 0., 0.

        for data in train_loader:
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            correct += ((output > 0.5) == data.y).sum().item()
            total += data.num_graphs
            rmsd += torch.sum((output - data.y)**2)

        train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = correct / total
        rmsd_total = torch.sqrt(rmsd/len(train_loader.dataset)) #rmsd
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)
        train_rmsd.append(rmsd_total)

        val_loss, val_accuracy, rmsd = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_acc.append(val_accuracy)
        val_rmsd.append(rmsd)


        scheduler.step(val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    return train_losses, val_losses, train_acc, val_acc

def evaluate_model(model, loader, criterion):
    """
    Evaluates the model on the given dataset.
    """
    model.eval()
    total_loss, correct, total, rmsd = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in loader:
            output = model(data).squeeze()
            loss = criterion(output, data.y)
            total_loss += loss.item() * data.num_graphs
            correct += ((output > 0.5) == data.y).sum().item()
            total += data.num_graphs
            rmsd += torch.sum((output - data.y)**2)                 #RMSD

            all_preds.extend(output.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    rmsd_total = torch.sqrt(rmsd/len(train_loader.dataset))           #RMSD
    return avg_loss, accuracy, all_preds, all_labels, rmsd_total

if __name__ == "__main__":
    set_seeds(42)
    data_dir = "./data"  # Path to PDB files
    train_data, val_data, test_data = load_data_split(data_dir)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    model = GNNModel()
    criterion = nn.BCELoss(input_dim=1, hidden_dim=128, output_dim=1, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.0009, weight_decay=0.003)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    train_losses, val_losses, train_acc, val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20
    )
    
    #prediction.py called

    test_loss, test_acc, all_preds, all_labels = evaluate_model(model, test_loader, criterion)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    plot_metrics_and_curves(train_losses, val_losses, train_acc, val_acc, all_preds, all_labels)