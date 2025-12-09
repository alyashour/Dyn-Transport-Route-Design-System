import torch as pt
import numpy as np
import matplotlib.pyplot as plt
from time import time

from dataset import ShotgunDataset
from model import ShotgunPredictor


def visualize_comparison(ground_truth, prediction, title="Traffic Comparison"):
    """
    Plots Ground Truth and Prediction side-by-side using the Magma colormap.
    
    Args:
        ground_truth (np.array): NxN matrix of actual trip counts.
        prediction (np.array): NxN matrix of predicted trip counts.
        title (str): Title for the entire figure.
    """
    
    # Create a figure with 2 subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Determine common scale for colorbars so the comparison is fair
    # We ignore zeros for the min calculation to avoid LogNorm errors later
    gt_max = ground_truth.max()
    pred_max = prediction.max()
    global_max = max(gt_max, pred_max)
    
    # For LogNorm, vmin must be > 0. We pick a small epsilon or the min non-zero value.
    global_min = 1e-5 # Default small value
    
    # Mask zeros for visualization (optional, makes them stand out against the background)
    gt_masked = np.ma.masked_where(ground_truth <= 0, ground_truth)
    pred_masked = np.ma.masked_where(prediction <= 0, prediction)
    
    # Configuration for power-law traffic data
    cmap = 'magma'
    # Use LogNorm to make low-traffic routes visible alongside high-traffic ones
    norm = LogNorm(vmin=max(1.0, global_min), vmax=global_max) if global_max > 1 else None

    # --- Plot 1: Ground Truth ---
    # We add +1 to raw data just in case inputs are 0-based counts, ensuring LogNorm doesn't crash on 0
    im1 = axes[0].imshow(ground_truth + 1, cmap=cmap, norm=norm, interpolation='nearest')
    axes[0].set_title(f"Ground Truth")
    axes[0].set_xlabel("Destination Stop Index")
    axes[0].set_ylabel("Origin Stop Index")
    # Add colorbar specific to this plot
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="Trips (Log Scale)")

    # --- Plot 2: Prediction ---
    im2 = axes[1].imshow(prediction + 1, cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title(f"Model Prediction")
    axes[1].set_xlabel("Destination Stop Index")
    axes[1].set_ylabel("Origin Stop Index")
    # Add colorbar specific to this plot
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="Trips (Log Scale)")

    # Overall Title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = ShotgunDataset("dataset.h5")
    start = time()
    model = ShotgunPredictor()
    model.load_state_dict(pt.load("shotgun.pth", map_location="cpu"))
    model.eval()
    setup_dur = time() - start
    print(f"Model setup completed in {setup_dur:.2f} seconds")

    with pt.no_grad():
        while 1:
            sample_id = input("ID: ")
            if sample_id == "":
                break

            x, truth = dataset[int(sample_id)]
            start = time()
            pred = model(x[None, :])[0].exp()
            dur = time() - start

            mse_loss = ((truth - pred) ** 2).mean()
            print(f"MSE loss = {mse_loss} | inference time: {dur:.2f} seconds")
            visualize_comparison(truth.view(1952, 1952), pred.view(1952, 1952))
