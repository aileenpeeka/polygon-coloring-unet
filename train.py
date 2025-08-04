import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
from model.unet import UNet
from utils import PolygonDataset
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model():
    # Initialize wandb
    wandb.init(project="ayna-ml-assignment", config={
        "learning_rate": 1e-4,
        "batch_size": 8,
        "epochs": 20,
        "architecture": "UNet",
        "dataset": "polygon-segmentation"
    })
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Check if dataset directories exist
    train_dir = "dataset/training"
    val_dir = "dataset/validation"
    
    if not os.path.exists(train_dir):
        print(f"Warning: Training directory {train_dir} not found. Creating dummy data.")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(os.path.join(train_dir, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "outputs"), exist_ok=True)
    
    if not os.path.exists(val_dir):
        print(f"Warning: Validation directory {val_dir} not found. Creating dummy data.")
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(os.path.join(val_dir, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "outputs"), exist_ok=True)
    
    # Create datasets
    train_dataset = PolygonDataset(train_dir, transform=transform)
    val_dataset = PolygonDataset(val_dir, transform=transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    model = UNet().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    epochs = 20
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for imgs, targets in train_pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            
            # Forward pass
            preds = model(imgs)
            loss = criterion(preds, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for imgs, targets in val_pbar:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_unet_model.pth")
            print(f"New best model saved! Val Loss: {best_val_loss:.4f}")
        
        # Log to wandb
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1
        })
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), "final_unet_model.pth")
    wandb.save("best_unet_model.pth")
    wandb.save("final_unet_model.pth")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    wandb.log({"training_curves": wandb.Image('training_curves.png')})
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model

if __name__ == "__main__":
    train_model() 