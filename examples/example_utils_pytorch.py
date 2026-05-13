"""
Example usage of utility functions for PyTorch

Demonstrates:
1. lazy_flatten - Auto-shape flattener
2. loss_ncc - Normalized Cross-Correlation loss
3. find_lr - Learning Rate Finder
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch.utils import (
    lazy_flatten,
    get_flatten_size,
    loss_ncc,
    ncc_score,
    find_lr,
    LRFinder
)


def example_1_lazy_flatten():
    """Example 1: Using lazy_flatten to simplify Conv->Linear transition"""
    print("\n" + "="*80)
    print("Example 1: Auto-Shape Flattener (lazy_flatten)")
    print("="*80)
    
    class SimpleCNN(nn.Module):
        """CNN without manual size calculation"""
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            
            # Calculate flattened size automatically
            # Input: 32x32 -> after 2 pools: 8x8, channels: 64
            flat_size = get_flatten_size((64, 8, 8))
            print(f"Calculated flatten size: {flat_size}")
            
            self.fc1 = nn.Linear(flat_size, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = lazy_flatten(x)  # No need to calculate 64*8*8!
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleCNN()
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("\n✓ Example 1 completed!")


def example_2_ncc_loss():
    """Example 2: Using NCC loss for image similarity"""
    print("\n" + "="*80)
    print("Example 2: Normalized Cross-Correlation Loss")
    print("="*80)
    
    # Simulate medical image registration task
    print("\nScenario: Medical Image Registration")
    print("Comparing original vs. registered images")
    
    # Original image
    original = torch.randn(1, 1, 128, 128)
    
    # Case 1: Perfect registration (identical)
    perfect = original.clone()
    ncc_loss_perfect = loss_ncc(original, perfect)
    ncc_score_perfect = ncc_score(original, perfect)
    print(f"\nPerfect registration:")
    print(f"  NCC Loss: {ncc_loss_perfect.item():.6f} (lower is better)")
    print(f"  NCC Score: {ncc_score_perfect.item():.6f} (higher is better)")
    
    # Case 2: Good registration (slight noise)
    good = original + torch.randn(1, 1, 128, 128) * 0.1
    ncc_loss_good = loss_ncc(original, good)
    ncc_score_good = ncc_score(original, good)
    print(f"\nGood registration (slight noise):")
    print(f"  NCC Loss: {ncc_loss_good.item():.6f}")
    print(f"  NCC Score: {ncc_score_good.item():.6f}")
    
    # Case 3: Poor registration (random image)
    poor = torch.randn(1, 1, 128, 128)
    ncc_loss_poor = loss_ncc(original, poor)
    ncc_score_poor = ncc_score(original, poor)
    print(f"\nPoor registration (random):")
    print(f"  NCC Loss: {ncc_loss_poor.item():.6f}")
    print(f"  NCC Score: {ncc_score_poor.item():.6f}")
    
    # Case 4: Scale invariance (NCC's advantage over MSE)
    scaled = original * 2.0
    ncc_loss_scaled = loss_ncc(original, scaled)
    mse_loss = F.mse_loss(original, scaled)
    print(f"\nScale invariance test (2x brightness):")
    print(f"  NCC Loss: {ncc_loss_scaled.item():.6f} (should be low)")
    print(f"  MSE Loss: {mse_loss.item():.6f} (will be high)")
    print("  → NCC is robust to intensity scaling!")
    
    print("\n✓ Example 2 completed!")


def example_3_ncc_in_training():
    """Example 3: Training a model with NCC loss"""
    print("\n" + "="*80)
    print("Example 3: Training with NCC Loss")
    print("="*80)
    
    # Simple autoencoder for image reconstruction
    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
            self.enc1 = nn.Conv2d(1, 16, 3, padding=1)
            self.enc2 = nn.Conv2d(16, 8, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            
            # Decoder
            self.dec1 = nn.ConvTranspose2d(8, 16, 2, stride=2)
            self.dec2 = nn.Conv2d(16, 1, 3, padding=1)
        
        def forward(self, x):
            # Encode
            x = self.pool(F.relu(self.enc1(x)))
            x = self.pool(F.relu(self.enc2(x)))
            
            # Decode
            x = F.relu(self.dec1(x))
            x = torch.sigmoid(self.dec2(x))
            
            return x
    
    model = Autoencoder()
    
    # Create dummy dataset
    images = torch.randn(100, 1, 32, 32)
    dataset = TensorDataset(images, images)  # Reconstruct same image
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with NCC loss
    print("\nTraining for 3 epochs with NCC loss...")
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = model(batch_x)
            
            # NCC loss
            loss = loss_ncc(batch_y, reconstructed)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/3, NCC Loss: {avg_loss:.4f}")
    
    print("\n✓ Example 3 completed!")


def example_4_lr_finder():
    """Example 4: Using Learning Rate Finder"""
    print("\n" + "="*80)
    print("Example 4: Learning Rate Finder")
    print("="*80)
    
    # Create a simple classification model
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = lazy_flatten(x)  # Using lazy_flatten!
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = Classifier()
    
    # Create dummy MNIST-like dataset
    train_data = TensorDataset(
        torch.randn(500, 1, 28, 28),
        torch.randint(0, 10, (500,))
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Initialize optimizer with placeholder LR
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Method 1: Using LRFinder class
    print("\nMethod 1: Using LRFinder class")
    lr_finder = LRFinder(model, optimizer, criterion, device='cpu')
    lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=1.0, num_iter=50)
    
    best_lr = lr_finder.get_best_lr()
    print(f"\n✓ Suggested learning rate: {best_lr:.2e}")
    
    # Uncomment to show plot:
    # lr_finder.plot()
    
    # Method 2: Using convenience function
    print("\nMethod 2: Using find_lr() convenience function")
    best_lr_2 = find_lr(
        model, optimizer, criterion, train_loader,
        device='cpu',
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=50,
        plot=False  # Set to True to see plot
    )
    print(f"\n✓ Suggested learning rate: {best_lr_2:.2e}")
    
    # Now use the best LR for actual training
    print(f"\n💡 Now you can create optimizer with lr={best_lr:.2e}")
    print(f"   optimizer = torch.optim.Adam(model.parameters(), lr={best_lr:.2e})")
    
    print("\n✓ Example 4 completed!")


def example_5_complete_pipeline():
    """Example 5: Complete training pipeline using all utilities"""
    print("\n" + "="*80)
    print("Example 5: Complete Pipeline with All Utilities")
    print("="*80)
    
    # Define model using lazy_flatten and get_flatten_size
    class MedicalSegmentationModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
            self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
            self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            
            # Bottleneck
            flat_size = get_flatten_size((64, 16, 16))  # After one pool
            self.fc1 = nn.Linear(flat_size, 512)
            
            # Decoder (simplified)
            self.fc2 = nn.Linear(512, flat_size)
            self.dec1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec2 = nn.Conv2d(32, 1, 3, padding=1)
        
        def forward(self, x):
            # Encode
            x1 = F.relu(self.enc1(x))
            x2 = self.pool(F.relu(self.enc2(x1)))
            
            # Bottleneck
            x_flat = lazy_flatten(x2)
            x_fc = F.relu(self.fc1(x_flat))
            
            # Decode
            x_unfold = self.fc2(x_fc)
            x_unfold = x_unfold.view(-1, 64, 16, 16)
            x_dec = F.relu(self.dec1(x_unfold))
            x_out = torch.sigmoid(self.dec2(x_dec))
            
            return x_out
    
    model = MedicalSegmentationModel()
    
    # Create dummy medical imaging dataset
    images = torch.randn(200, 1, 32, 32)
    masks = torch.randint(0, 2, (200, 1, 32, 32)).float()
    dataset = TensorDataset(images, masks)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Step 1: Find optimal learning rate
    print("\n Step 1: Finding optimal learning rate...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Using NCC loss directly as criterion
    def ncc_criterion(output, target):
        return loss_ncc(target, output)
    
    best_lr = find_lr(
        model, optimizer, ncc_criterion, dataloader,
        num_iter=30, plot=False
    )
    print(f"   Found LR: {best_lr:.2e}")
    
    # Step 2: Train with optimal LR and NCC loss
    print(f"\n📚 Step 2: Training with LR={best_lr:.2e} and NCC loss...")
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    
    model.train()
    for epoch in range(2):  # Just 2 epochs for demo
        total_loss = 0
        total_score = 0
        
        for batch_img, batch_mask in dataloader:
            optimizer.zero_grad()
            
            # Forward
            output = model(batch_img)
            
            # NCC loss
            loss = loss_ncc(batch_mask, output)
            score = ncc_score(batch_mask, output)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_score += score.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_score = total_score / len(dataloader)
        print(f"   Epoch {epoch+1}: NCC Loss={avg_loss:.4f}, NCC Score={avg_score:.4f}")
    
    print("\n✓ Complete pipeline executed successfully!")
    print("\n💡 This pipeline used:")
    print("   - get_flatten_size() to calculate layer sizes")
    print("   - lazy_flatten() to simplify Conv->Linear transition")
    print("   - find_lr() to find optimal learning rate")
    print("   - loss_ncc() and ncc_score() for medical image segmentation")
    
    print("\n✓ Example 5 completed!")


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*80)
    print("PYTORCH UTILITY FUNCTIONS - EXAMPLES")
    print("="*80)
    
    example_1_lazy_flatten()
    example_2_ncc_loss()
    example_3_ncc_in_training()
    example_4_lr_finder()
    example_5_complete_pipeline()
    
    print("\n" + "="*80)
    print("✅ ALL EXAMPLES COMPLETED!")
    print("="*80)
    print("\n📖 Summary:")
    print("   1. lazy_flatten - Simplifies Conv->Linear transitions")
    print("   2. loss_ncc - Robust loss for medical imaging")
    print("   3. find_lr - Finds optimal learning rate automatically")
    print("\n💡 These utilities save time and improve model performance!")


if __name__ == "__main__":
    run_all_examples()
