import torch
import torch.nn as nn

class WaferLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=1):
        super(WaferLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Using GRU for efficiency
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initial hidden state is zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        
        # We want a prediction at every timestep
        out = self.fc(out)
        return out

if __name__ == "__main__":
    # Test model
    model = WaferLSTM()
    dummy_input = torch.randn(32, 100, 4) # (batch, seq_len, features)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be (32, 100, 1)
