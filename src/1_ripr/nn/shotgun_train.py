import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
from tqdm import tqdm

# Import the dataset we just created
from dataset import ShotgunDataset
from model import ShotgunPredictor

# --- CONFIGURATION ---
H5_FILE = '../cache/dataset.h5'
BATCH_SIZE = 32 # 32 is what I used but you'll wanna close all your tabs
LEARNING_RATE = 1e-3
EPOCHS = 60
VALIDATION_SPLIT = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_val_loss = 4
save_loss_gap = 0.1
kill_loss_gap = 0.1


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # 1. Initialize Dataset
    print("Loading Dataset...")
    full_dataset = ShotgunDataset(H5_FILE)
    
    # Calculate dimensions dynamically
    sample_x, sample_y = full_dataset[0]
    input_dim = sample_x.shape[0]
    num_stops = int(np.sqrt(sample_y.shape[0]))
    
    print(f" - Input Features: {input_dim}")
    print(f" - System Stops: {num_stops} (Output Grid: {num_stops}x{num_stops} = {num_stops**2})")
    
    # 2. Split Train/Val
    total_size = len(full_dataset)
    val_size = int(total_size * VALIDATION_SPLIT)
    train_size = total_size - val_size
    
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 3. Initialize Model
    model = ShotgunPredictor(input_dim, num_stops)

    try:
        model.load_state_dict(torch.load("backup.pth", map_location="cpu"))
        print("Loaded backup.")
    except:
        print("Fresh weights")

    model = model.to(DEVICE)  # trust me you just gotta move it after to not kill vram

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # KLDivLoss is perfect for comparing two distributions (Predicted Proportions vs Actual Proportions)
    # reduction='batchmean' mathematically aligns with the KL definition
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    # 4. Training Loop
    print("\nStarting Training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        valid_batches = 0
        
        for inputs, targets in tqdm(train_loader, desc="training", unit="batches"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # --- CRITICAL: Handle Zero-Trip Days ---
            # If a day has 0 trips, the target vector sum is 0. 
            # A probability distribution must sum to 1.
            # We mask these out to avoid training on empty days (or fitting noise).
            mask = targets.sum(dim=1) > 0
            
            if not mask.any():
                continue # Skip batch if entirely empty
                
            active_inputs = inputs[mask]
            active_targets = targets[mask]
            
            # Forward
            optimizer.zero_grad()
            outputs = model(active_inputs) # LogSoftmax outputs
            
            # Loss (KL Divergence)
            loss = criterion(outputs, active_targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            valid_batches += 1
            
        avg_train_loss = train_loss / max(1, valid_batches)
        
        # 5. Validation Step
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="validating", unit="batches"):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                # Same masking for validation
                mask = targets.sum(dim=1) > 0
                if not mask.any(): continue
                    
                active_inputs = inputs[mask]
                active_targets = targets[mask]
                
                outputs = model(active_inputs)
                loss = criterion(outputs, active_targets)
                
                val_loss += loss.item()
                val_batches += 1
                
        avg_val_loss = val_loss / max(1, val_batches)
        
        # Progress Print
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if best_val_loss - avg_val_loss >= save_loss_gap:
            best_val_loss = avg_val_loss
            print("Saving backup... ", end="")
            torch.save(model.state_dict(), 'backup.pth')
            print("[SAVED]")

        if avg_val_loss - best_val_loss >= kill_loss_gap:
            print(f"Early stopped, validation loss exceeds kill loss gap above the best validation loss.")
            break

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time:.2f} seconds.")
    
    # Save Model
    #torch.save(model.state_dict(), 'transit_model.pth')
    #print("Model saved to 'transit_model.pth'")
