"""
Example usage of ToTf TrainingMonitor with PyTorch
This demonstrates how to use the TrainingMonitor during model training
"""

# Import PyTorch first to avoid any import conflicts
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
import os

# Add parent directory to path to import ToTf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PyTorch first to avoid conflicts
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ToTf import TrainingMonitor


# Simple Neural Network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_dummy_data(n_samples=1000, n_features=10):
    """Create dummy dataset for demonstration"""
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def train_model():
    """Example training loop with TrainingMonitor"""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create data
    train_loader = create_dummy_data(n_samples=1000)
    val_loader = create_dummy_data(n_samples=200)
    
    # Training
    epochs = 5
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        train_monitor = TrainingMonitor(
            train_loader, 
            desc=f"Training Epoch {epoch + 1}", 
            log_file=f"train_log_epoch_{epoch + 1}.csv"
        )
        
        for batch_idx, (data, target) in enumerate(train_monitor):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Log metrics
            train_monitor.log({
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Validation phase
        model.eval()
        val_monitor = TrainingMonitor(
            val_loader, 
            desc=f"Validation Epoch {epoch + 1}", 
            log_file=f"val_log_epoch_{epoch + 1}.csv"
        )
        
        with torch.no_grad():
            for data, target in val_monitor:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                # Log validation metrics
                val_monitor.log({
                    'val_loss': loss.item()
                })
    
    print("\nTraining completed!")
    print(f"Log files saved in current directory")


if __name__ == "__main__":
    train_model()
