# Model Training Report

## Model Architecture

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Text Classification
- **Number of Classes**: 2
- **Sequence Length**: 128 tokens
- **Pooling Strategy**: [CLS] token representation
- **Device**: Automatically selects CUDA/MPS/CPU

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 5e-5 |
| Epochs | 3 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Warmup Steps | 0 |
| Gradient Accumulation Steps | 4 |
| Mixed Precision Training | Enabled |
| Early Stopping Patience | 3 |
| Mixed Precision | fp16 |

## Training Insights

### Data
- Training Examples: 25,000 (80% of total)
- Validation Examples: 12,500 (20% of total)
- Class Distribution (Perfectly Balanced):
  - Class 0 (Negative): 12,500 instances (50%)
  - Class 1 (Positive): 12,500 instances (50%)

### Performance
- **Accuracy**: 87.44%
- **Precision**: 87.46%
- **Recall**: 87.44%
- **F1 Score**: 87.43%

*Note: These metrics are on the test set. Training and validation metrics are tracked during training.*

## Key Observations
1. **Convergence**: [Describe how the model converged]
2. **Overfitting**: [Note any overfitting and mitigation strategies]
3. **Training Stability**: [Note any training stability issues]

## Error Analysis

### Common Error Patterns
1. **False Positives (1,737 instances)**: Negative reviews misclassified as positive, often due to:
   - Sarcastic or ironic language in negative reviews
   - Mixed sentiment reviews with positive phrases
   - Negative reviews containing positive comparisons (e.g., "better than expected but still bad")

2. **False Negatives (1,404 instances)**: Positive reviews misclassified as negative, typically because:
   - Positive reviews with negative comparisons (e.g., "not as bad as I thought")
   - Subtle positive sentiment without strong positive words
   - Positive reviews mentioning minor negatives

3. **Ambiguous Cases**: Reviews with unclear sentiment:
   - Neutral or balanced reviews
   - Mixed sentiment reviews
   - Context-dependent sentiment (e.g., "so bad it's good")
   - Sarcastic statements without clear indicators

### Confusion Matrix

|                | Predicted Negative | Predicted Positive |
|----------------|-------------------|-------------------|
| **Actual Negative** | 10,763 (TN)      | 1,737 (FP)        |
| **Actual Positive** | 1,404 (FN)       | 11,096 (TP)       |


**Key Metrics**:
- **True Positive Rate (Sensitivity/Recall)**: 88.8% (11,096 / 12,500)
- **True Negative Rate (Specificity)**: 86.1% (10,763 / 12,500)
- **False Positive Rate**: 13.9% (1,737 / 12,500)
- **False Negative Rate**: 11.2% (1,404 / 12,500)

**Class-wise Performance**:
- **Negative Class**: 86.1% correctly identified
- **Positive Class**: 88.8% correctly identified

The model shows slightly better performance in identifying positive reviews compared to negative ones, but the difference is minimal (2.7 percentage points). The balanced accuracy is approximately 87.5%, which aligns with the overall accuracy of 87.44%.

## Recommendations for Improvement

1. **Data-Level**
   - Collect more training data, especially for underrepresented classes
   - Apply data augmentation techniques
   - Address class imbalance if present

2. **Model-Level**
   - Try different pre-trained models (e.g., RoBERTa, DeBERTa)
   - Adjust learning rate and batch size
   - Implement learning rate scheduling
   - Try different sequence lengths

3. **Training**
   - Increase number of epochs with early stopping
   - Implement gradient accumulation for larger effective batch sizes
   - Try different optimizers (e.g., Adam, SGD with momentum)

## Next Steps
1. [Next immediate action]
2. [Future experiments to try]
3. [Potential A/B testing]
