import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns


from config import model_config, data_config

class ModelUtils:
    """Utility class for model training and evaluation."""
    
    def __init__(self, model_name: str = None, num_labels: int = None):
        """
        Initialize model utilities.
        
        Args:
            model_name: Name or path of the pretrained model
            num_labels: Number of output labels
        """
        self.model_name = model_name or model_config.model_name
        self.num_labels = num_labels or model_config.num_labels
        self.model = None
        self.tokenizer = None
        self.device = model_config.device
    
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)
        return self.model, self.tokenizer
    
    def save_model(self, output_dir: str = None):
        """
        Save the model and tokenizer.
        
        Args:
            output_dir: Directory to save the model
        """
        if output_dir is None:
            output_dir = model_config.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        if self.tokenizer is not None:
            tokenizer_dir = os.path.join(os.path.dirname(output_dir), 'tokenizer')
            os.makedirs(tokenizer_dir, exist_ok=True)
            self.tokenizer.save_pretrained(tokenizer_dir)
    
    def setup_optimizer_and_scheduler(self, train_loader, lr: float = None, weight_decay: float = 0.01):
        """
        Setup optimizer and learning rate scheduler.
        
        Args:
            train_loader: Training data loader
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        lr = lr or model_config.learning_rate
        
        # Separate parameters for different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Calculate total steps
        total_steps = len(train_loader) * model_config.num_epochs
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    @staticmethod
    def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Dictionary of metrics
        """
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None, 
                            save_path: str = None) -> None:
        """
        Plot and optionally save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save the plot
        """
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
            
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def save_metrics(metrics: Dict[str, Any], file_path: str) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metrics
            file_path: Path to save the metrics
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
