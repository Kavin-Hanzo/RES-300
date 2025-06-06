import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def load_segmentation_model(model_path=""):
    """
    Load the segmentation model with pre-trained weights
    
    Args:
        model_path (str): Path to the model weights file
    
    Returns:
        SimpleUNet: Loaded model ready for inference
    """
    try:
        print(f"Loading segmentation model from: {model_path}")
        model = SimpleUNet()
        
        # Load weights if path exists
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model weights loaded successfully")
        else:
            print("⚠ Model weights file not found, using random initialization")
        
        model.eval()  # Set to evaluation mode
        model.name = "SimpleUNet"
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Return model with random weights if loading fails
        model = SimpleUNet()
        model.eval()
        model.name = "SimpleUNet"
        return model

def load_sample_data(sample_path):
    """
    Load a single .npz sample file containing X (input) and Y (ground truth)
    
    Args:
        sample_path (str): Path to the .npz file
    
    Returns:
        tuple: (input_image, ground_truth_mask)
    """
    try:
        data = np.load(sample_path)
        X = data['X']  # Input image
        Y = data['Y']  # Ground truth mask
        
        print(f"✓ Loaded sample: {os.path.basename(sample_path)}")
        print(f"  Input shape: {X.shape}, Ground truth shape: {Y.shape}")
        
        return X, Y
        
    except Exception as e:
        print(f"Error loading sample {sample_path}: {str(e)}")
        return None, None

def get_input_samples(samples_directory):
    """
    Get all .npz sample files from the specified directory
    
    Args:
        samples_directory (str): Directory containing .npz files
    
    Returns:
        list: List of sample file paths
    """
    try:
        # Find all .npz files in the directory
        sample_files = glob.glob(os.path.join(samples_directory, "*.npz"))
        
        if not sample_files:
            print(f"⚠ No .npz files found in {samples_directory}")
            return []
        
        print(f"✓ Found {len(sample_files)} sample files")
        return sorted(sample_files)
        
    except Exception as e:
        print(f"Error accessing samples directory: {str(e)}")
        return []

def run_segmentation_inference(input_image, model):
    """
    Run segmentation inference on the input image
    
    Args:
        input_image (np.array): Input image array
        model (SimpleUNet): Loaded segmentation model
    
    Returns:
        dict: Dictionary containing binary_mask and probability_mask
    """
    try:
        print(f"Running inference with {model.name}")
        
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            sample_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0)
            
            # Run inference
            prediction_mask = model(sample_tensor)
            prediction_mask = prediction_mask.squeeze().cpu().numpy()
            
            # Create binary mask
            binary_pred = (prediction_mask > 0.5).astype(np.uint8) * 255
            
        return {
            "binary_mask": binary_pred,
            "probability_mask": prediction_mask
        }
        
    except Exception as e:
        print(f"Inference failed: {str(e)}")
        return None
    
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=20, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def create_segmentation_visualization(original_image, ground_truth, results, model_name, sample_name):
    """
    Create comprehensive visualization of segmentation results including ground truth
    
    Args:
        original_image (np.array): Original input image (first 3 channels for RGB display)
        ground_truth (np.array): Ground truth mask
        results (dict): Segmentation results containing binary_mask and probability_mask
        model_name (str): Name of the model used
        sample_name (str): Name of the sample being processed
    
    Returns:
        matplotlib.figure.Figure: Visualization figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Segmentation Results - {model_name} - {sample_name}', fontsize=16, fontweight='bold')
    
    # Helper function to ensure 2D array for grayscale display
    def ensure_2d(arr):
        if len(arr.shape) == 3 and arr.shape[0] == 1:
            return arr.squeeze(0)  # Remove first dimension if it's 1
        elif len(arr.shape) == 3 and arr.shape[2] == 1:
            return arr.squeeze(2)  # Remove last dimension if it's 1
        elif len(arr.shape) == 2:
            return arr
        else:
            return arr.reshape(arr.shape[-2], arr.shape[-1])  # Take last 2 dimensions
    
    # Convert multi-channel image to RGB for display (use first 3 channels)
    if original_image.shape[0] >= 3:
        display_image = np.transpose(original_image[:3], (1, 2, 0))
        # Normalize to 0-1 range for display
        display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min())
    else:
        # If less than 3 channels, use the first channel and ensure it's 2D
        display_image = ensure_2d(original_image[0:1] if original_image.shape[0] > 0 else original_image)
        # Normalize to 0-1 range for display
        display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min())
    
    # Original Image (RGB or grayscale visualization)
    if len(display_image.shape) == 3:
        axes[0, 0].imshow(display_image)
        axes[0, 0].set_title('Input Image (RGB)', fontweight='bold')
    else:
        axes[0, 0].imshow(display_image, cmap='gray')
        axes[0, 0].set_title('Input Image (Grayscale)', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground Truth Mask - ensure it's 2D
    gt_display = ensure_2d(ground_truth)
    axes[0, 1].imshow(gt_display, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Predicted Binary Mask - ensure it's 2D
    binary_display = ensure_2d(results['binary_mask'])
    axes[0, 2].imshow(binary_display, cmap='gray')
    axes[0, 2].set_title('Predicted Binary Mask', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Probability Mask - ensure it's 2D
    prob_display = ensure_2d(results['probability_mask'])
    im1 = axes[1, 0].imshow(prob_display, cmap='jet', alpha=0.8)
    axes[1, 0].set_title('Probability Mask', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Overlay: Original + Ground Truth
    if len(display_image.shape) == 3:
        overlay_gt = display_image.copy()
        gt_normalized = gt_display / np.max(gt_display) if np.max(gt_display) > 0 else gt_display
        overlay_gt[:, :, 1] = np.clip(overlay_gt[:, :, 1] + gt_normalized * 0.3, 0, 1)
    else:
        overlay_gt = np.stack([display_image, display_image, display_image], axis=2)
        gt_normalized = gt_display / np.max(gt_display) if np.max(gt_display) > 0 else gt_display
        overlay_gt[:, :, 1] = np.clip(overlay_gt[:, :, 1] + gt_normalized * 0.3, 0, 1)
    
    axes[1, 1].imshow(overlay_gt)
    axes[1, 1].set_title('Overlay: Input + Ground Truth', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Overlay: Original + Prediction
    if len(display_image.shape) == 3:
        overlay_pred = display_image.copy()
        pred_normalized = binary_display / np.max(binary_display) if np.max(binary_display) > 0 else binary_display
        overlay_pred[:, :, 0] = np.clip(overlay_pred[:, :, 0] + pred_normalized * 0.3, 0, 1)
    else:
        overlay_pred = np.stack([display_image, display_image, display_image], axis=2)
        pred_normalized = binary_display / np.max(binary_display) if np.max(binary_display) > 0 else binary_display
        overlay_pred[:, :, 0] = np.clip(overlay_pred[:, :, 0] + pred_normalized * 0.3, 0, 1)
    
    axes[1, 2].imshow(overlay_pred)
    axes[1, 2].set_title('Overlay: Input + Prediction', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig

def process_all_samples(samples_directory, model_path="", save_plots=False, output_dir="results"):
    """
    Main function to process all samples and create visualizations
    
    Args:
        samples_directory (str): Directory containing .npz sample files
        model_path (str): Path to the model weights file
        save_plots (bool): Whether to save plots to disk
        output_dir (str): Directory to save plots (if save_plots=True)
    """
    print("="*60)
    print("SEGMENTATION PIPELINE STARTED")
    print("="*60)
    
    # Step 1: Get all sample files
    sample_files = get_input_samples(samples_directory)
    if not sample_files:
        print("No samples found. Exiting.")
        return
    
    # Step 2: Load the model
    model = load_segmentation_model(model_path)
    
    # Step 3: Create output directory if saving plots
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Output directory created: {output_dir}")
    
    # Step 4: Process each sample
    print(f"\nProcessing {len(sample_files)} samples...")
    print("-" * 40)
    
    for i, sample_file in enumerate(sample_files, 1):
        sample_name = os.path.basename(sample_file).replace('.npz', '')
        print(f"\n[{i}/{len(sample_files)}] Processing: {sample_name}")
        
        # Load sample data
        X, Y = load_sample_data(sample_file)
        if X is None or Y is None:
            print(f"⚠ Skipping {sample_name} due to loading error")
            continue
        
        # Run inference
        results = run_segmentation_inference(X, model)
        if results is None:
            print(f"⚠ Skipping {sample_name} due to inference error")
            continue
        
        # Create visualization
        try:
            fig = create_segmentation_visualization(X, Y, results, model.name, sample_name)
            
            if save_plots:
                plot_path = os.path.join(output_dir, f"{sample_name}_segmentation.png")
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"✓ Plot saved: {plot_path}")
                plt.close(fig)  # Close to save memory
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error creating visualization for {sample_name}: {str(e)}")
            continue
    
    print("\n" + "="*60)
    print("SEGMENTATION PIPELINE COMPLETED")
    print("="*60)

# Example usage
if __name__ == "__main__":
    # Configuration
    SAMPLES_DIR = "samples/"  # Directory containing .npz files
    MODEL_PATH = "model.pt"  # Path to model weights
    SAVE_PLOTS = True  # Set to False to display plots instead of saving
    OUTPUT_DIR = "segmentation_results"  # Directory to save results
    
    # Run the complete pipeline
    process_all_samples(
        samples_directory=SAMPLES_DIR,
        model_path=MODEL_PATH,
        save_plots=SAVE_PLOTS,
        output_dir=OUTPUT_DIR
    )
    
    # Alternative: Process specific samples
    # sample_files = ["sample1.npz", "sample2.npz", "sample3.npz"]
    # for sample_file in sample_files:
    #     process_all_samples(sample_file, MODEL_PATH, SAVE_PLOTS, OUTPUT_DIR)