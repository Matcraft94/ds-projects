# Creado por: Lucy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.clip_value = 1.0
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.2,
            bidirectional=False
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

class ModelTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 0.0001):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            self.optimizer.zero_grad()
            output = self.model(X_batch)
            
            if torch.isnan(output).any():
                print("NaN detectado en output")
                continue
            
            loss = self.criterion(output, y_batch.unsqueeze(1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, test_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch.unsqueeze(1))
                total_loss += loss.item()
                
        return total_loss / len(test_loader)