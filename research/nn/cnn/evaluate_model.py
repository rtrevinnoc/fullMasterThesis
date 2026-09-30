import os
import sys
import pickle
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import WaferCNN
from train_model import WaferDataset, load_and_preprocess_data, label_map, stratified_split


def confusion_matrix(trues, preds, labels):
    # sklearn.metrics is unavailable here: see the note in train_model.py's
    # stratified_split (scipy Fortran extensions fail to dlopen on this
    # machine's macOS 27 beta).
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(trues, preds):
        cm[t, p] += 1
    return cm


def classification_report(trues, preds, target_names, zero_division=0):
    trues = np.asarray(trues)
    preds = np.asarray(preds)
    n = len(target_names)
    cm = confusion_matrix(trues, preds, labels=list(range(n)))
    lines = [f"{'':14s}{'precision':>10s}{'recall':>10s}{'f1-score':>10s}{'support':>10s}"]
    f1s = []
    for i, name in enumerate(target_names):
        support = int(cm[i, :].sum())
        tp = int(cm[i, i])
        pred_pos = int(cm[:, i].sum())
        precision = tp / pred_pos if pred_pos > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
        lines.append(f"{name:14s}{precision:10.2f}{recall:10.2f}{f1:10.2f}{support:10d}")
    acc = float((trues == preds).mean())
    lines.append("")
    lines.append(f"{'accuracy':14s}{'':10s}{'':10s}{acc:10.2f}{len(trues):10d}")
    lines.append(f"{'macro avg':14s}{'':10s}{'':10s}{np.mean(f1s):10.2f}{len(trues):10d}")
    return "\n".join(lines)

OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_matrix.png')
PRESENTATION_PNG = '/Users/rtrevinnoc/maestria/tesis/presentation/pictures/cnn_confmat.png'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    pickle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LSWMD.pkl')
    df = load_and_preprocess_data(pickle_path)

    X = df['waferMap'].values
    y = df['label'].values

    _, X_test, _, y_test = stratified_split(X, y, test_size=0.2, seed=42)
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
