# -*- coding: utf-8 -*-
"""Prediction Module"""

import torch
from torch_geometric.loader import DataLoader
from gnn_model import GNNModel  # Import the GNN model
from data_processing import MolecularFeatureExtractor, ProteinLigandDataset, load_data_splits, load_data_split
from memory_efficient_gnn import MemoryEfficientGNN
def predict(model_path, test_loader, model = "GCN"):
    """
    Loads a trained model and makes predictions on the test dataset.
    """
    # Load the model
    if (model == "GCN"):
        model = GNNModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        all_preds, all_labels = [], []

        with torch.no_grad():
            for data in test_loader:
                output = model(data).squeeze()
                all_preds.extend(output.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

        return all_preds, all_labels
    
    model = MemoryEfficientGNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch).squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())

    return all_preds, all_targets

if __name__ == "__main__":
    model = "GCN"
    if( model == "GCN"):
        data_dir = "./data"  # Path to PDB files
        _, _, test_data = load_data_split(data_dir)

        test_loader = DataLoader(test_data, batch_size=32)

        model_path = "./output/best_model.pt"  # Path to the saved model
        preds, labels = predict(model_path, test_loader)

        # Print results
        print("Predictions:", preds)
        print("Labels:", labels)
    else:
        PROJECT_PATH = "Medusa_Graph/data/"
        MODEL_PATH = f"{PROJECT_PATH}/model_outputs/best_model.pt"

        splits = load_data_splits(PROJECT_PATH)
        feature_extractor = MolecularFeatureExtractor()

        test_loader = DataLoader(
            ProteinLigandDataset(splits, feature_extractor, split_type='test', data_fraction=0.5),
            batch_size=8, num_workers=0
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        preds, targets = predict(MODEL_PATH, test_loader, device)

        print("Predictions:", preds)
        print("Targets:", targets)

    