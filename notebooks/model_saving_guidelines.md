# Model Saving and Loading

## Saving the Model

After training, the model and tokenizer are automatically saved to the following locations:

```python
# Model is saved to:
models/trained_model/

# Tokenizer is saved to:
tokenizer/
```

## Loading the Model for Inference

To load the saved model for predictions:

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Load the saved model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('models/trained_model')
tokenizer = DistilBertTokenizer.from_pretrained('tokenizer')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Example prediction
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    predictions = torch.softmax(logits, dim=1)
    return predictions.cpu().numpy()

# Example usage
# sentiment = predict_sentiment("This movie was amazing!")
# print(f"Sentiment probabilities: {sentiment}")
```

## Model Artifacts

The following files are saved with the model:

- `config.json`: Model configuration
- `pytorch_model.bin`: Model weights
- `training_metrics.json`: Training history and metrics
- `val_labels.npy`: Validation set true labels
- `val_predictions.npy`: Validation set predictions

## Notes

- The model is saved in PyTorch format
- The tokenizer is saved with all necessary files for reproduction
- All necessary files are included for model deployment
- The model can be loaded on any machine with the same dependencies
