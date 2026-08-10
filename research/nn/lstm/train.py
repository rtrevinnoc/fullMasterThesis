import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from model import WaferLSTM
from sklearn.preprocessing import StandardScaler
import os

class WaferDataset(Dataset):
    def __init__(self, df, seq_length=500):
        self.seq_length = seq_length
        
        # Features: v, a, j, s
        self.features = df[['v', 'a', 'j', 's']].values
        self.targets = df['e_syn'].values
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.features = self.feature_scaler.fit_transform(self.features)
        self.targets = self.target_scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()
        
        # We'll split the continuous data into fixed-length sequences
        # For simplicity, we'll just slide with no overlap if seq_length is large
        self.n_samples = len(df) // seq_length

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        
        x = torch.FloatTensor(self.features[start:end])
        y = torch.FloatTensor(self.targets[start:end]).unsqueeze(-1)
        
        return x, y

def train():
    data_path = "research/nn/lstm/sim_data.pkl"
    if not os.path.exists(data_path):
        print("Data not found. Run generate_data.py first.")
        return
        
    df = pd.read_pickle(data_path)
    
    # Train/Test split by simulation ID
    sim_ids = df['sim_id'].unique()
    train_ids = sim_ids[:int(0.8 * len(sim_ids))]
    test_ids = sim_ids[int(0.8 * len(sim_ids)):]
    
    train_df = df[df['sim_id'].isin(train_ids)]
    test_df = df[df['sim_id'].isin(test_ids)]
    
    seq_len = 500
    train_dataset = WaferDataset(train_df, seq_length=seq_len)
    test_dataset = WaferDataset(test_df, seq_length=seq_len)
    
    # We need to share the scaler for test set
    test_dataset.feature_scaler = train_dataset.feature_scaler
    test_dataset.target_scaler = train_dataset.target_scaler
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaferLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                test_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Test Loss: {test_loss/len(test_loader):.6f}")
        
    # Save model and scalers
    torch.save(model.state_dict(), "research/nn/lstm/wafer_lstm.pth")
    # Save scalers for later use
    import pickle
    with open("research/nn/lstm/scalers.pkl", "wb") as f:
        pickle.dump({'feature': train_dataset.feature_scaler, 'target': train_dataset.target_scaler}, f)
    
    print("Training complete. Model and scalers saved.")

if __name__ == "__main__":
    train()
