import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from torch.utils.data import DataLoader

from .data_preprocessing import prepare_data_for_training
from .model_utils import ModelUtils
from .config import model_config

class Trainer:
    """Custom trainer class for fine-tuning DistilBERT."""
    
    def __init__(self, model, device, tokenizer):
        """Initialize the trainer."""
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        gradient_accumulation_steps: int = 1,
        epoch: int = 0
    ) -> Tuple[float, float]:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1} [Training]')
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Calculate metrics
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item() * gradient_accumulation_steps  # Scale back loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'acc': f"{correct / total:.4f}"
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(
        self, 
        eval_loader: DataLoader,
        epoch: int = 0
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(eval_loader, desc=f'Epoch {epoch + 1} [Validation]')
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Calculate metrics
                loss = outputs.loss
                logits = outputs.logits
                _, predicted = torch.max(logits, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{correct / total:.4f}"
                })
        
        avg_loss = total_loss / len(eval_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)

def train(model_config):
    """Main training function."""
    # Set seed for reproducibility
    torch.manual_seed(model_config.seed)
    np.random.seed(model_config.seed)
    
    # Initialize model utilities
    model_utils = ModelUtils()
    
    # Load model and tokenizer
    model, tokenizer = model_utils.load_model()
    
    # Prepare data
    train_dataset, val_dataset, test_dataset, _ = prepare_data_for_training()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for MPS compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Setup optimizer and scheduler
    optimizer, scheduler = model_utils.setup_optimizer_and_scheduler(
        train_loader,
        lr=model_config.learning_rate
    )
    
    # Initialize trainer
    trainer = Trainer(model, model_config.device, tokenizer)
    
    # Training loop
    best_accuracy = 0.0
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(model_config.num_epochs):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1}/{model_config.num_epochs}")
        print(f"{'='*50}")
        
        # Train for one epoch
        train_loss, train_acc = trainer.train_epoch(
            train_loader,
            optimizer,
            scheduler,
            gradient_accumulation_steps=model_config.gradient_accumulation_steps,
            epoch=epoch
        )
        
        # Evaluate on validation set
        val_loss, val_acc, val_preds, val_labels = trainer.evaluate(val_loader, epoch=epoch)
        
        # Update training history
        trainer.training_history['train_loss'].append(train_loss)
        trainer.training_history['train_acc'].append(train_acc)
        trainer.training_history['val_loss'].append(val_loss)
        trainer.training_history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            patience_counter = 0
            
            # Save model
            model_utils.save_model()
            
            # Save predictions for analysis
            np.save(os.path.join(model_config.output_dir, 'val_predictions.npy'), val_preds)
            np.save(os.path.join(model_config.output_dir, 'val_labels.npy'), val_labels)
            
            print(f"\nNew best model saved! Validation Accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= model_config.early_stopping_patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = trainer.evaluate(test_loader)
    
    # Calculate metrics
    metrics = model_utils.compute_metrics(test_preds, test_labels)
    
    # Save metrics
    metrics_output_dir = os.path.join(os.path.dirname(model_config.output_dir), "reports", "evaluation_metrics.json")
    model_utils.save_metrics(metrics, metrics_output_dir)
    
    # Plot and save confusion matrix
    model_utils.plot_confusion_matrix(
        test_labels,
        test_preds,
        class_names=['Negative', 'Positive'],
        save_path=os.path.join(os.path.dirname(model_config.output_dir), "reports", "confusion_matrix.png")
    )
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Metrics: {metrics}")
    print(f"Training completed. Model and metrics saved to {model_config.output_dir}")
    
    return metrics

if __name__ == "__main__":
    train()
