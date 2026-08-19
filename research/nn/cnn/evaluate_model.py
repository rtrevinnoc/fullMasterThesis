import os
import sys
import pickle
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import WaferCNN
from train_model import WaferDataset, load_and_preprocess_data, label_map

OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_matrix.png')
PRESENTATION_PNG = '/Users/rtrevinnoc/maestria/tesis/presentation/pictures/cnn_confmat.png'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    pickle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LSWMD.pkl')
    df = load_and_preprocess_data(pickle_path)

    X = df['waferMap'].values
    y = df['label'].values

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Held-out test set size: {len(X_test)}")

    test_dataset = WaferDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    model = WaferCNN(num_classes=9).to(device)
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wafer_cnn.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            out = model(images)
            _, p = torch.max(out, 1)
            preds.extend(p.cpu().numpy().tolist())
            trues.extend(labels.numpy().tolist())

    preds = np.array(preds)
    trues = np.array(trues)

    inv = {v: k for k, v in label_map.items()}
    class_names = [inv[i] for i in range(9)]

    cm = confusion_matrix(trues, preds, labels=list(range(9)))
    cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True).clip(min=1)
    acc = (preds == trues).mean() * 100
    print(f"Test accuracy: {acc:.2f}%")
    print(classification_report(trues, preds, target_names=class_names, zero_division=0))

    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(class_names, rotation=40, ha='right')
    ax.set_yticklabels(class_names)


    for i in range(9):
        for j in range(9):
            v = cm_norm[i, j]
            if v >= 0.005:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color='white' if v > 0.5 else 'black', fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches='tight')
    fig.savefig(PRESENTATION_PNG, dpi=160, bbox_inches='tight')
    print(f"Saved {OUT_PNG}")
    print(f"Saved {PRESENTATION_PNG}")

if __name__ == "__main__":
    main()
