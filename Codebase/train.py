import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config import Config
from dataset import SynapseDataset
from model import EMGConv1D

def train_model():
    Config.ensure_dirs()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Datasets
    print("Initializing datasets...")
    train_dataset = SynapseDataset(Config.DATA_ROOT, subjects=Config.TRAIN_SUBJECTS, mode='train')
    val_dataset = SynapseDataset(Config.DATA_ROOT, subjects=Config.VAL_SUBJECTS, mode='val')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("Error: No training data found. Check paths and subjects.")
        return

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for Windows safety
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = EMGConv1D().to(device)
    print(f"Model created with {model.get_num_params()} parameters.")
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training Loop
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())
            
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}")
        print(f"          Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        
        # Scheduler step
        scheduler.step(val_f1)
        
        # Checkpointing
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"Saved new best model to {Config.MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{Config.PATIENCE}")
            
        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    
    # Final Evaluation on Validation Set with Confusion Matrix
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()
    final_preds = []
    final_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.numpy())
            
    cm = confusion_matrix(final_labels, final_preds)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    train_model()
