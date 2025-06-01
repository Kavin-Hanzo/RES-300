
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from typing import Tuple, Dict, Optional, List, Union

def test_segmentation_model(
    model: torch.nn.Module,
    data_path: str,
    num_samples: Optional[int] = None,
    threshold: float = 0.5,
    random_seed: int = 42,
    visualize: bool = True,
    save_results: bool = False,
    output_dir: str = "./results/"
) -> Dict[str, Union[float, np.ndarray, List[np.ndarray]]]:
    """
    Test a segmentation model on a dataset and evaluate performance metrics.

    Args:
        model: PyTorch model to test
        data_path: Path to the .npz file containing the dataset
        num_samples: Number of samples to test. If None, uses all samples
        threshold: Threshold value for binary prediction (0-1 range)
        random_seed: Random seed for reproducibility
        visualize: Whether to display visualizations
        save_results: Whether to save visualization results
        output_dir: Directory to save results if save_results is True

    Returns:
        Dictionary containing performance metrics and results
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Ensure model is in evaluation mode
    model.eval()

    # Load the dataset
    try:
        data = np.load(data_path)
        test_X = data['X'] #X
        test_Y = data['Y'] #Y
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")

    # Select samples
    if num_samples is None or num_samples > len(test_X):
        num_samples = len(test_X)
        indices = np.arange(len(test_X))
    else:
        indices = np.random.choice(len(test_X), num_samples, replace=False)

    samples = test_X[indices]
    masks = test_Y[indices]

    # Initialize containers for metrics
    all_predictions = []
    all_true_masks = []
    metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'iou': 0.0,
        'tpr': 0.0,
        'fpr': 0.0,
        'auc': 0.0,
        'confusion_matrix': None,
        'sample_results': []
    }

    # Create output directory if saving results
    if save_results:
        import os
        os.makedirs(output_dir, exist_ok=True)

    # Process each sample
    with torch.no_grad():
        for i in range(num_samples):
            # Get current sample and ground truth
            sample = samples[i]
            mask = masks[i][0]  # Assuming mask has shape (1, H, W)

            # Convert to tensor and add batch dimension
            sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)

            # Make prediction
            prediction = model(sample_tensor)
            prediction = prediction.squeeze().cpu().numpy()  # Remove batch dimension

            # Store raw prediction and ground truth for later metrics calculation
            all_predictions.append(prediction)
            all_true_masks.append(mask)

            # Apply threshold to get binary prediction
            binary_pred = (prediction > threshold).astype(np.uint8) * 255

            # Calculate per-sample metrics
            sample_metrics = calculate_metrics(mask, binary_pred)
            metrics['sample_results'].append(sample_metrics)

            # Visualize if requested
            if visualize:
                fig = visualize_results(sample, mask, prediction, binary_pred, sample_metrics, index=i)

                if save_results:
                    plt.savefig(f"{output_dir}/sample_{i}_results.png", bbox_inches='tight')

                plt.show()
                plt.close(fig)

    # Calculate overall metrics
    metrics = calculate_overall_metrics(all_true_masks, all_predictions, metrics, threshold)

    # Plot confusion matrix
    if visualize:
        plot_confusion_matrix(metrics['confusion_matrix'])

        if save_results:
            plt.savefig(f"{output_dir}/confusion_matrix.png", bbox_inches='tight')

        plt.show()

        # Plot ROC curve
        plot_roc_curve(all_true_masks, all_predictions, metrics['auc'])

        if save_results:
            plt.savefig(f"{output_dir}/roc_curve.png", bbox_inches='tight')

        plt.show()

    return metrics

def calculate_metrics(true_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics for a single sample."""
    # Ensure binary format
    true_binary = (true_mask > 0).astype(np.uint8)
    pred_binary = (pred_mask > 0).astype(np.uint8)

    # Calculate confusion matrix elements
    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)  # Also known as TPR
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # Intersection over Union (IoU)
    intersection = np.sum(pred_binary & true_binary)
    union = np.sum(pred_binary | true_binary)
    iou = intersection / (union + 1e-10)

    # True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr = recall  # TPR is the same as recall
    fpr = fp / (fp + tn + 1e-10)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'tpr': tpr,
        'fpr': fpr,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def calculate_overall_metrics(
    all_true_masks: List[np.ndarray],
    all_predictions: List[np.ndarray],
    metrics: Dict[str, Union[float, np.ndarray]],
    threshold: float
) -> Dict[str, Union[float, np.ndarray]]:
    """Calculate overall metrics across all samples."""
    # Flatten all masks and predictions for global metrics
    true_flat = np.concatenate([mask.flatten() for mask in all_true_masks])
    pred_flat_raw = np.concatenate([pred.flatten() for pred in all_predictions])
    pred_flat_binary = (pred_flat_raw > threshold).astype(np.uint8)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_flat, pred_flat_raw)
    roc_auc = auc(fpr, tpr)

    # Calculate confusion matrix
    cm = confusion_matrix(true_flat, pred_flat_binary)

    # Calculate overall metrics using the confusion matrix
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)

    metrics['accuracy'] = accuracy
    metrics['precision'] = precision
    metrics['recall'] = recall  # Same as TPR
    metrics['f1_score'] = f1
    metrics['iou'] = iou
    metrics['tpr'] = recall
    metrics['fpr'] = fp / (fp + tn + 1e-10)
    metrics['auc'] = roc_auc
    metrics['confusion_matrix'] = cm

    return metrics

def visualize_results(
    sample: np.ndarray,
    true_mask: np.ndarray,
    raw_prediction: np.ndarray,
    binary_prediction: np.ndarray,
    metrics: Dict[str, float],
    index: int
) -> plt.Figure:
    """Visualize input, ground truth, prediction and metrics."""
    fig = plt.figure(figsize=(16, 6))

    # Process input image (assuming first 3 channels are RGB)
    if sample.shape[0] >= 3:
        # Take the first 3 channels and transpose to HWC format
        smp = np.transpose(sample[:3], (1, 2, 0))
        # Normalize to 0-1 range
        smp = np.clip(smp / smp.max(), 0, 1)
    else:
        # If fewer than 3 channels, use first channel
        smp = sample[0]
        smp = np.clip(smp / smp.max(), 0, 1)

    # Plot original image
    plt.subplot(2, 3, 1)
    plt.imshow(smp)
    plt.title("Input Image")
    plt.axis('off')

    # Plot ground truth mask
    plt.subplot(2, 3, 2)
    plt.imshow(true_mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    # Plot raw predictions (probability map)
    plt.subplot(2, 3, 3)
    plt.imshow(raw_prediction, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Raw Prediction (Probability Map)")
    plt.axis('off')

    # Plot binary prediction
    plt.subplot(2, 3, 4)
    plt.imshow(binary_prediction, cmap='gray')
    plt.title("Thresholded Prediction")
    plt.axis('off')

    # Plot error map (False positives and false negatives)
    error_map = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
    # True positive (white)
    error_map[(binary_prediction > 0) & (true_mask > 0)] = [255, 255, 255]
    # False positive (red)
    error_map[(binary_prediction > 0) & (true_mask == 0)] = [255, 0, 0]
    # False negative (blue)
    error_map[(binary_prediction == 0) & (true_mask > 0)] = [0, 0, 255]

    plt.subplot(2, 3, 5)
    plt.imshow(error_map)
    plt.title("Error Map (FP=red, FN=blue)")
    plt.axis('off')

    # Display metrics
    # plt.subplot(2, 3, 6)
    # plt.axis('off')
    # info_text = (
    #     f"Sample #{index}\n\n"
    #     f"Accuracy: {metrics['accuracy']:.4f}\n"
    #     f"IoU: {metrics['iou']:.4f}\n"
    #     f"Precision: {metrics['precision']:.4f}\n"
    #     f"Recall (TPR): {metrics['tpr']:.4f}\n"
    #     f"FPR: {metrics['fpr']:.4f}\n"
    #     f"F1 Score: {metrics['f1_score']:.4f}\n\n"
    #     f"TP: {metrics['tp']}, FP: {metrics['fp']}\n"
    #     f"FN: {metrics['fn']}, TN: {metrics['tn']}"
    # )
    # plt.text(0.1, 0.5, info_text, fontsize=12, va='center')

    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm: np.ndarray) -> None:
    """Plot the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

def plot_roc_curve(all_true: List[np.ndarray], all_preds: List[np.ndarray], auc_score: float) -> None:
    """Plot the ROC curve."""
    # Flatten all masks and predictions
    true_flat = np.concatenate([mask.flatten() for mask in all_true])
    pred_flat = np.concatenate([pred.flatten() for pred in all_preds])

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(true_flat, pred_flat)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

def load_best_model(model: torch.nn.Module, best_model_path: str) -> Tuple[torch.nn.Module, dict]:
    """Load a saved model checkpoint."""
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

