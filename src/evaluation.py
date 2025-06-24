"""Evaluation utilities for the IMDB sentiment analysis model."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, roc_auc_score
)
from typing import Tuple, Dict, Any, List, Optional
import torch
from tqdm.auto import tqdm

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None
) -> None:
    """Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save the figure
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()

def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    title: str = 'ROC Curve',
    save_path: Optional[str] = None
) -> float:
    """Plot ROC curve and return AUC score.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities for the positive class
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    
    return roc_auc

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    title: str = 'Precision-Recall Curve',
    save_path: Optional[str] = None
) -> float:
    """Plot precision-recall curve and return average precision score.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities for the positive class
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Average precision score
    """
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    
    plt.figure()
    plt.step(recall, precision, where='post', alpha=0.8, color='b', 
             label=f'AP = {avg_precision:.2f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc='lower left')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    
    return avg_precision

def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = ['loss', 'accuracy'],
    save_path: Optional[str] = None
) -> None:
    """Plot training and validation metrics over epochs.
    
    Args:
        history: Dictionary containing training history
        metrics: List of metrics to plot
        save_path: Path to save the figure
    """
    epochs = range(1, len(history['train_' + metrics[0]]) + 1)
    
    for metric in metrics:
        plt.figure()
        
        # Plot training metric
        plt.plot(epochs, history['train_' + metric], 'b-', label=f'Training {metric}')
        
        # Plot validation metric if available
        if f'val_{metric}' in history:
            plt.plot(epochs, history['val_' + metric], 'r-', label=f'Validation {metric}')
        
        plt.title(f'Training and Validation {metric.capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        
        if save_path:
            metric_save_path = f"{os.path.splitext(save_path)[0]}_{metric}{os.path.splitext(save_path)[1]}"
            os.makedirs(os.path.dirname(metric_save_path), exist_ok=True)
            plt.savefig(metric_save_path, bbox_inches='tight')
        
        plt.show()

def analyze_errors(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    label_names: List[str] = None,
    num_examples: int = 5,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """Analyze model errors and return examples of different error types.
    
    Args:
        texts: List of input texts
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities for the positive class
        label_names: List of label names
        num_examples: Number of examples to show for each error type
        threshold: Confidence threshold for uncertain predictions
        
    Returns:
        Dictionary containing error analysis
    """
    if label_names is None:
        label_names = ['Negative', 'Positive']
    
    # Convert to numpy arrays if they're not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # Calculate confidence (maximum probability)
    confidences = np.maximum(y_probs, 1 - y_probs)
    
    # Find indices of different error types
    correct = y_pred == y_true
    errors = y_pred != y_true
    fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]  # False positives
    fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]  # False negatives
    uncertain_indices = np.where(confidences < 0.5 + threshold)[0]  # Uncertain predictions
    
    # Sort by confidence
    fp_sorted = sorted(zip(fp_indices, y_probs[fp_indices]), key=lambda x: x[1], reverse=True)
    fn_sorted = sorted(zip(fn_indices, y_probs[fn_indices]), key=lambda x: x[1])
    
    # Get top examples
    results = {
        'false_positives': [],
        'false_negatives': [],
        'uncertain_predictions': []
    }
    
    # Add false positives
    for idx, prob in fp_sorted[:num_examples]:
        results['false_positives'].append({
            'text': texts[idx],
            'true_label': label_names[y_true[idx]],
            'pred_label': label_names[y_pred[idx]],
            'confidence': float(confidences[idx]),
            'prob_positive': float(y_probs[idx])
        })
    
    # Add false negatives
    for idx, prob in fn_sorted[:num_examples]:
        results['false_negatives'].append({
            'text': texts[idx],
            'true_label': label_names[y_true[idx]],
            'pred_label': label_names[y_pred[idx]],
            'confidence': float(confidences[idx]),
            'prob_positive': float(y_probs[idx])
        })
    
    # Add uncertain predictions
    uncertain_sorted = sorted(zip(uncertain_indices, confidences[uncertain_indices]), 
                            key=lambda x: x[1])
    
    for idx, conf in uncertain_sorted[:num_examples]:
        results['uncertain_predictions'].append({
            'text': texts[idx],
            'true_label': label_names[y_true[idx]],
            'pred_label': label_names[y_pred[idx]],
            'confidence': float(conf),
            'prob_positive': float(y_probs[idx])
        })
    
    # Calculate error statistics
    error_stats = {
        'total_examples': len(y_true),
        'correct_predictions': int(np.sum(correct)),
        'incorrect_predictions': int(np.sum(errors)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'false_positives_count': len(fp_indices),
        'false_negatives_count': len(fn_indices),
        'uncertain_predictions_count': len(uncertain_indices),
        'average_confidence': float(np.mean(confidences)),
        'average_confidence_correct': float(np.mean(confidences[correct])) if np.any(correct) else 0.0,
        'average_confidence_incorrect': float(np.mean(confidences[errors])) if np.any(errors) else 0.0
    }
    
    return {
        'error_analysis': results,
        'statistics': error_stats
    }

def save_evaluation_report(
    report: Dict[str, Any],
    output_dir: str,
    file_name: str = 'evaluation_report.json'
) -> str:
    """Save evaluation report to a JSON file.
    
    Args:
        report: Evaluation report dictionary
        output_dir: Output directory
        file_name: Output file name
        
    Returns:
        Path to the saved report
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, file_name)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report_path

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    label_names: List[str] = None,
    output_dir: str = None
) -> Dict[str, Any]:
    """Evaluate a model and generate comprehensive evaluation metrics.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        label_names: List of label names
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing evaluation results
    """
    if label_names is None:
        label_names = ['Negative', 'Positive']
    
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Get model outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Classification report
    cls_report = classification_report(
        y_true, y_pred, 
        target_names=label_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # Precision-recall curve and average precision
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    
    # Create evaluation results dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'classification_report': cls_report,
        'confusion_matrix': cm.tolist(),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': _.tolist()
        },
        'precision_recall_curve': {
            'precision': precision_curve.tolist(),
            'recall': recall_curve.tolist(),
            'thresholds': _.tolist()
        },
        'predictions': {
            'true_labels': y_true.tolist(),
            'predicted_labels': y_pred.tolist(),
            'probabilities': y_probs.tolist()
        }
    }
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'average_precision': avg_precision
            }, f, indent=2)
        
        # Save classification report
        with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
            json.dump(cls_report, f, indent=2)
        
        # Save confusion matrix
        np.save(os.path.join(output_dir, 'confusion_matrix.npy'), cm)
        
        # Save ROC curve data
        np.savez(
            os.path.join(output_dir, 'roc_curve.npz'),
            fpr=fpr,
            tpr=tpr,
            thresholds=_
        )
        
        # Save precision-recall curve data
        np.savez(
            os.path.join(output_dir, 'precision_recall_curve.npz'),
            precision=precision_curve,
            recall=recall_curve,
            thresholds=_
        )
        
        # Save predictions
        np.savez(
            os.path.join(output_dir, 'predictions.npz'),
            true_labels=y_true,
            predicted_labels=y_pred,
            probabilities=y_probs
        )
        
        # Plot and save visualizations
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_names, yticklabels=label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight')
        plt.close()
        
        # ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), bbox_inches='tight')
        plt.close()
        
        # Precision-Recall curve
        plt.figure()
        plt.step(recall_curve, precision_curve, where='post', alpha=0.8, color='b', 
                label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), bbox_inches='tight')
        plt.close()
    
    return results
