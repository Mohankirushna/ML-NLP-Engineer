
#!/usr/bin/env python3
"""
Main training script for the text classification model.
"""
import os
import argparse
import json
from pathlib import Path

from src.train_model import train
from src.config import model_config, data_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a text classification model.")
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=model_config.batch_size,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=model_config.learning_rate,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=model_config.num_epochs,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=model_config.output_dir,
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=model_config.seed,
        help="Random seed"
    )
    
    return parser.parse_args()
    parser.add_argument(
        "--num-labels",
        type=int,
        default=model_config.num_labels,
        help="Number of output labels"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=model_config.max_length,
        help="Maximum sequence length"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=model_config.batch_size,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=model_config.learning_rate,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=model_config.num_epochs,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=model_config.output_dir,
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=model_config.seed,
        help="Random seed"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the training process."""
    args = parse_args()
    
    # Update config with command line arguments
    model_config.batch_size = args.batch_size
    model_config.learning_rate = args.learning_rate
    model_config.num_epochs = args.num_epochs
    model_config.output_dir = args.output_dir
    model_config.seed = args.seed
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config = {
        "model_config": {
            "model_name": model_config.model_name,
            "num_labels": model_config.num_labels,
            "max_length": model_config.max_length,
            "batch_size": model_config.batch_size,
            "learning_rate": model_config.learning_rate,
            "num_epochs": model_config.num_epochs,
            "output_dir": model_config.output_dir,
            "seed": model_config.seed
        }
    }
    
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Run training
    print("Starting training...")
    metrics = train(model_config=model_config)
    
    # Save metrics
    metrics_path = Path(args.output_dir) / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTraining completed. Model saved to {args.output_dir}")
    print(f"Validation metrics: {metrics}")

if __name__ == "__main__":
    main()
