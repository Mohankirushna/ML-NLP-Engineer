import os
import torch
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_steps: int = 0
    logging_steps: int = 50
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    save_total_limit: int = 1
    fp16: bool = True
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/trained_model")
    seed: int = 42
    gradient_accumulation_steps: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dataset_name: str = "imdb"
    early_stopping_patience: int = 3

@dataclass
class DataConfig:
    text_column: str = "text"
    label_column: str = "label"
    test_size: float = 0.2
    random_state: int = 42
    max_samples: Optional[int] = None  # For debugging with smaller dataset

# Initialize configurations
model_config = ModelConfig()
data_config = DataConfig()

# Add torch import at the end to avoid circular imports
import torch
