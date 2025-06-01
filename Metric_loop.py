from torch.utils.data import Dataset
import os
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class MethaneDataset(Dataset):
    def __init__(self, folder_path, split='train'):
        """
        Args:
            folder_path (str): Path to the folder containing .npz batch files.
            split (str): 'train' or 'val'
        """
        self.X_list = []
        self.Y_list = []

        all_files = sorted(glob(os.path.join(folder_path, "batch_*.npz")))
        if not all_files:
            raise FileNotFoundError(f"No .npz files found in {folder_path}")

        split = split.lower()
        assert split in ['train', 'val'], "split must be 'train' or 'val'"

        total = len(all_files)
        split_idx = int(total * 0.8)

        if split == 'train':
            selected_files = all_files[:split_idx]
        else:
            selected_files = all_files[split_idx:]

        print(f"📦 Loading {len(selected_files)} '{split}' batch files...")

        for file in selected_files:
            data = np.load(file)
            self.X_list.append(data['X'])
            self.Y_list.append(data['Y'])

        self.X = np.concatenate(self.X_list, axis=0)
        self.Y = np.concatenate(self.Y_list, axis=0)

        print(f"✅ {split.capitalize()} dataset loaded: {self.X.shape[0]} samples")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # shape: (20, H, W)
        y = self.Y[idx]  # shape: (1, H, W)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train_model(model, npz_path, epochs=10, batch_size=16, lr=1e-3, device='cuda',
                save_dir='models', model_name='methane_model'):
    """
    Train the model and save the best version based on validation F1 score

    Args:
        model: The model to train
        npz_path: Path to the dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save model checkpoints
        model_name: Base name for saved model files

    Returns:
        model: The trained model
        history: Dictionary with training metrics
        best_model_path: Path to the best model checkpoint
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Dataset
    train_ds = MethaneDataset(npz_path, split='train')
    val_ds = MethaneDataset(npz_path, split='val')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # F1 Metric (torchmetrics)
    f1_metric = BinaryF1Score(threshold=0.5).to(device)

    # History tracking
    history = {
        'train_loss': [],
        'val_f1': []
    }

    model.to(device)

    # Variables to track best model
    best_f1 = -1.0
    best_epoch = -1
    best_model_path = ""

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            Y = Y.float()  # (B, 1, H, W)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_f1 = 0.0
        with torch.no_grad():
            for X, Y in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                X, Y = X.to(device), Y.to(device)
                Y = Y.float()
                output = model(X)
                preds = torch.sigmoid(output) > 0.5
                val_f1 += f1_metric(preds.int(), Y.int()).item()

        avg_val_f1 = val_f1 / len(val_loader)
        history['val_f1'].append(avg_val_f1)

        print(f"Epoch {epoch+1} — Train Loss: {avg_train_loss:.4f}, Val F1: {avg_val_f1:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'f1': avg_val_f1
        }, checkpoint_path)

        # Check if this is the best model so far
        if avg_val_f1 > best_f1:
            if best_model_path and os.path.exists(best_model_path):
                # Optionally remove previous best to save space
                # os.remove(best_model_path)
                pass

            best_f1 = avg_val_f1
            best_epoch = epoch + 1
            best_model_path = os.path.join(save_dir, f"{model_name}_best.pt")

            # Save as best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'f1': avg_val_f1
            }, best_model_path)

            print(f"✓ New best model saved with F1: {avg_val_f1:.4f}")

    print(f"\nTraining complete. Best model at epoch {best_epoch} with F1: {best_f1:.4f}")
    print(f"Best model saved at: {best_model_path}")

    # Return the model, history, and path to the best model
    return model, history, best_model_path


def plot_training_history(history, save_plot=True, plot_path='training_history.png'):
    """
    Plot the training loss and validation F1 score history

    Args:
        history: Dictionary containing 'train_loss' and 'val_f1' lists
        save_plot: Whether to save the plot to disk
        plot_path: Path to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot validation F1 score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_f1'], 'r-', label='Validation F1 Score')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {plot_path}")

    plt.show()

    # Print best epoch information
    best_epoch = np.argmax(history['val_f1']) + 1
    best_f1 = history['val_f1'][best_epoch - 1]
    print(f"Best model at epoch {best_epoch} with validation F1 score: {best_f1:.4f}")


def load_best_model(model, model_path):
    """
    Load the best model checkpoint into the provided model

    Args:
        model: Model architecture to load weights into
        model_path: Path to the saved model checkpoint

    Returns:
        model: Model with loaded weights
        checkpoint: The loaded checkpoint dictionary with metadata
    """
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from epoch {checkpoint['epoch']} with:")
    print(f"  - Loss: {checkpoint['loss']:.4f}")
    print(f"  - F1 Score: {checkpoint['f1']:.4f}")

    return model, checkpoint
