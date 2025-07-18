# Text Classification with DistilBERT

This project implements a text classification model using DistilBERT, a distilled version of BERT that is faster and smaller while retaining most of BERT's performance. The implementation is built using PyTorch and the Hugging Face Transformers library.

## Project Structure

```
.
├── .git/                           # Git version control
├── .gitignore                      # Git ignore file
│
├── notebooks/                     # Jupyter notebooks for exploration and analysis
│   ├── data_exploration.ipynb      # Initial data analysis and visualization
│   ├── model_training.ipynb        # Model training experiments
│   └── evaluation_analysis.ipynb   # Model evaluation and result analysis
│
├── src/                          # Source code
│   ├── __init__.py                 # Python package initialization
│   ├── train_model.py              # Main training script with custom Trainer class
│   ├── data_preprocessing.py       # Data loading and preprocessing utilities
│   ├── model_utils.py              # Model architecture and utility functions
│   ├── evaluation.py               # Model evaluation and metrics
│   └── config.py                   # Configuration settings and hyperparameters
│
├── models/                       # Trained models and checkpoints
│   ├── trained_model/              # Best trained model
│   │   ├── config.json             # Model configuration
│   │   ├── training_metrics.json   # Training history and metrics
│   │   ├── val_labels.npy          # Validation set true labels
│   │   └── val_predictions.npy     # Validation set predictions
│   └── reports/                    # Model evaluation visualizations
│       ├── confusion_matrix.png     # Confusion matrix plot
│       ├── roc_curve.png           # ROC curve
│       └── precision_recall_curve.png  # Precision-Recall curve
│
├── tokenizer/                    # Saved tokenizer files
│   ├── tokenizer_config.json       # Tokenizer configuration
│   ├── special_tokens_map.json     # Special tokens mapping
│   └── vocab.txt                   # Vocabulary file
│
├── reports/                      # Project reports and metrics
│   ├── model_report.md             # Detailed model analysis report
│   ├── confusion_matrix.png        # Confusion matrix visualization
│   └── evaluation_metrics.json     # Evaluation metrics in JSON format
│
├── requirements.txt              # Python dependencies
├── train.py                       # Training script entry point
├── submission.md                  # Project report and documentation
└── README.md                      # This file
```

## Features

- Fine-tuning of DistilBERT for text classification
- Support for binary and multi-class classification
- Efficient training with gradient accumulation
- Early stopping and model checkpointing
- Mixed-precision training support
- Comprehensive evaluation metrics

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) For GPU acceleration with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### Training
Train the model with default configuration:
```bash
python train.py
```

For custom training, you can modify the parameters in `src/config.py` or pass them as command-line arguments.

### Evaluation
Run evaluation on the test set:
```bash
python -m src.evaluation
```

### Using Jupyter Notebooks
For exploration and analysis:
```bash
jupyter lab notebooks/
```

## Configuration

The main configuration is defined in `src/config.py`. Key parameters include:

- Model: `distilbert-base-uncased` (default)
- Training: batch size, learning rate, number of epochs
- Hardware: Automatic device detection (CUDA/MPS/CPU)
- Dataset: IMDB (default), but can be customized

## Results

Model performance on the test set will be saved in the `reports/` directory, including:
- Classification report
- Confusion matrix
- Training history plots

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

