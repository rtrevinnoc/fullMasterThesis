import os
import sys
import pickle
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Workaround for older pandas pickle
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

from model import WaferCNN

# Mapping
label_map = {
    'none': 0,
    'Center': 1,
    'Donut': 2,
    'Edge-Loc': 3,
    'Edge-Ring': 4,
    'Loc': 5,
    'Near-full': 6,
    'Random': 7,
    'Scratch': 8
}

def load_and_preprocess_data(pickle_path):
    print(f"Loading data from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        df = pickle.load(f, encoding='latin1')
    
    print("Filtering data...")
    # Filter for samples where failureType is not empty
    def get_failure_type(x):
        if isinstance(x, np.ndarray) and x.size > 0:
            return str(x[0][0])
        if isinstance(x, list) and len(x) > 0:
            return str(x[0][0])
        return None

    df['failureType_str'] = df['failureType'].apply(get_failure_type)
    df = df[df['failureType_str'].notnull()]
    
    # Map to integers
    df['label'] = df['failureType_str'].map(label_map)
    df = df[df['label'].notnull()]
    df['label'] = df['label'].astype(int)
    
    print(f"Total samples after filtering: {len(df)}")
    return df

def stratified_split(X, y, test_size=0.2, seed=42):
    # sklearn's train_test_split is unavailable here: this machine's scipy
    # build fails to dlopen (Fortran extensions rejected by the macOS 27
    # beta's dyld with a __DATA/__thread_bss validation error), so sklearn's
    # import chain (via scipy.stats/scipy.optimize) breaks at import time.
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

class WaferDataset(Dataset):
    def __init__(self, wafer_maps, labels):
        self.wafer_maps = wafer_maps
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Resize to 64x64
        img = self.wafer_maps[idx]
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
        # Add channel dimension (1, 64, 64)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        
        label = self.labels[idx]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    pickle_path = 'research/nn/cnn/LSWMD.pkl'
    df = load_and_preprocess_data(pickle_path)
    
    X = df['waferMap'].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=42)

    train_dataset = WaferDataset(X_train, y_train)
    test_dataset = WaferDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = WaferCNN(num_classes=9).to(device)
    class_counts = np.bincount(y_train, minlength=9)
    class_weights = class_counts.sum() / (9 * class_counts.clip(min=1))
    print(f"Class counts (train): {class_counts.tolist()}")
    print(f"Class weights: {np.round(class_weights, 3).tolist()}")
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss/len(train_loader))
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Accuracy after epoch {epoch+1}: {accuracy:.2f}%")
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'research/nn/cnn/wafer_cnn.pth')
            print(f"Best model saved with accuracy: {best_acc:.2f}%")

    print(f"Final Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()
