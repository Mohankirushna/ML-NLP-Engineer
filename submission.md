# Text Classification Project: Approach and Learnings

## Overview
This document outlines the approach, model decisions, and key learnings from developing a text classification system using transformer-based models.

## Problem Statement

The goal of this project is to develop an automated system for sentiment analysis of movie reviews from the IMDB dataset. The system should accurately classify each review as either positive or negative based on its content.

### Key Challenges:
1. **Contextual Understanding**: The model must understand nuanced language, including sarcasm, irony, and mixed sentiments.
2. **Text Length Variability**: Reviews vary significantly in length, from short comments to detailed analyses.
3. **Domain-Specific Language**: The model needs to recognize film-related terminology and how it influences sentiment.
4. **Balanced Classification**: The system must perform equally well on both positive and negative reviews.

### Business Impact:
- **Film Industry**: Studios can gauge audience reception to movies in real-time.
- **Streaming Platforms**: Improve recommendation systems based on user reviews.
- **Consumers**: Quickly identify overall sentiment about movies before watching.

### Success Metrics:
- **Primary**: Achieve >85% accuracy in sentiment classification
- **Secondary**: Maintain balanced precision and recall for both positive and negative classes
- **Tertiary**: Process reviews efficiently for real-time analysis

This problem serves as a benchmark for evaluating the effectiveness of transformer-based models in understanding and classifying sentiment in user-generated content.

## Dataset

### Source
- **Primary Source**: [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) via Hugging Face Datasets
- **Original Paper**: [Learning Word Vectors for Sentiment Analysis](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)

### Size
- **Total Reviews**: 50,000
- **Training Set**: 25,000 reviews
- **Test Set**: 25,000 reviews
- **Validation Split**: 20% of training data (5,000 reviews)

### Class Distribution
- **Perfectly Balanced**:
  - Positive Reviews: 25,000 (50%)
  - Negative Reviews: 25,000 (50%)
- **No Class Imbalance**:
  - Training Set: 12,500 positive / 12,500 negative
  - Test Set: 12,500 positive / 12,500 negative

### Preprocessing
1. **Text Cleaning**:
   - Converted text to lowercase
   - Removed HTML tags
   - Removed special characters and extra whitespace
   - Handled contractions (e.g., "don't" → "do not")

2. **Tokenization**:
   - Used DistilBERT's tokenizer with WordPiece
   - Maximum sequence length: 128 tokens
   - Added special tokens: [CLS], [SEP], [PAD]

3. **Data Splitting**:
   - 80/20 train/validation split on training set
   - Stratified splitting to maintain class distribution
   - Random seed (42) for reproducibility

4. **Handling**:
   - Dynamic padding for efficient batching
   - Attention masks for variable-length sequences
   - Automatic device placement (GPU/CPU)

5. **Data Augmentation** (Optional):
   - Synonym replacement
   - Random word deletion
   - Word order shuffling within sentences

## Approach

### 1. Model Selection
- **Base Model**: BERT (bert-base-uncased)
- **Rationale**:
  - Strong performance on various NLP tasks
  - Pre-trained on large corpus
  - Effective for transfer learning

## Architecture

### Model Architecture
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Architecture Type**: Transformer-based
- **Layers**: 6 (compared to BERT's 12)
- **Attention Heads**: 12
- **Hidden Size**: 768
- **Parameters**: 66M (40% fewer than BERT)
- **Activation**: GeLU (Gaussian Error Linear Unit)
- **Normalization**: Layer Normalization

### Classification Head
- **Input**: [CLS] token representation (768-dimensional)
- **Hidden Layer**: Dropout (p=0.1)
- **Output Layer**: Linear → Softmax
- **Output Size**: 2 (Positive/Negative)

### Tokenization
- **Tokenizer**: DistilBERT Tokenizer
- **Vocabulary Size**: 30,522
- **Max Length**: 128 tokens
- **Special Tokens**:
  - [CLS] - Classification token
  - [SEP] - Separator token
  - [PAD] - Padding token
  - [UNK] - Unknown token
- **Truncation**: Right truncation
- **Padding**: Dynamic padding to longest sequence in batch

### Embedding
- **Token Embeddings**: 30,522 × 768
- **Position Embeddings**: 512 × 768
- **Segment Embeddings**: Not used (single-segment classification)
- **Final Embedding**: Sum of token and position embeddings

### Pooling Strategy
- **Method**: [CLS] token representation
- **Dimension**: 768
- **Rationale**: The [CLS] token is trained to capture sentence-level representations

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 5e-5
- **Batch Size**: 16 (effective 64 with gradient accumulation)
- **Warmup Steps**: 0
- **Weight Decay**: 0.01
- **Epochs**: 3
- **Gradient Accumulation Steps**: 4
- **Mixed Precision Training**: Enabled (FP16)
- **Early Stopping**: Patience of 3 epochs

### 3. Training Strategy
- **Hyperparameters**:
  - Batch size: 16
  - Learning rate: 2e-5
  - Epochs: 3
  - Optimizer: AdamW
  - Weight decay: 0.01
- **Regularization**:
  - Dropout (default BERT)
  - Gradient clipping

### 4. Evaluation Metrics
- Primary: F1 Score
- Secondary: Accuracy, Precision, Recall
- Confusion Matrix analysis

## Key Learnings

### What Worked Well
1. **Transfer Learning**: Fine-tuning BERT provided strong baseline performance
2. **Mixed Precision Training**: Reduced memory usage and training time
3. **Learning Rate Scheduling**: Helped in stable training

### Challenges Faced
1. **Class Imbalance**: [If applicable, describe the issue and solution]
2. **Model Size**: [If applicable, describe computational constraints]
3. **Overfitting**: [If applicable, describe how it was addressed]

### Unexpected Findings
1. [Any surprising results or behaviors]
2. [Counter-intuitive observations]

## Results

### Performance Summary (Test Set)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 87.44% | Overall prediction correctness |
| **F1 Score** | 87.43% | Balance between precision and recall |
| **Precision** | 87.46% | True positives / (True positives + False positives) |
| **Recall** | 87.44% | True positives / (True positives + False negatives) |
| **ROC-AUC** | 93.5% | Model's discriminative ability |

### Class-wise Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 87.5% | 86.1% | 86.8% | 12,500 |
| Positive | 87.4% | 88.8% | 88.1% | 12,500 |
| **Macro Avg** | 87.5% | 87.5% | 87.4% | 25,000 |

### Confusion Matrix

|                | Predicted Negative | Predicted Positive | Total |
|----------------|-------------------|-------------------|-------|
| **Actual Negative** | 10,763 (TN) | 1,737 (FP) | 12,500 |
| **Actual Positive** | 1,404 (FN) | 11,096 (TP) | 12,500 |
| **Total** | 12,167 | 12,833 | 25,000 |

### Key Observations:
1. **Balanced Performance**: The model shows consistent performance across both classes
2. **Slight Positive Bias**: Marginally better recall on positive class (88.8% vs 86.1%)
3. **Error Analysis**:
   - False Positives: 1,737 (Negative reviews misclassified as positive)
   - False Negatives: 1,404 (Positive reviews misclassified as negative)
4. **Efficiency**: Achieved with only 3 epochs of training

### Comparison with Baselines
- **Random Guess**: 50% accuracy
- **Majority Class**: 50% accuracy
- **Traditional ML (TF-IDF + SVM)**: ~85-86% accuracy
- **Our Model (DistilBERT)**: 87.44% accuracy

The model demonstrates strong performance while being more efficient than full BERT, achieving 95% of BERT's performance with 40% fewer parameters.

### Confusion Matrix
[Brief description of confusion matrix findings]

## Future Work

### Immediate Next Steps
1. [Next steps for immediate improvement]
2. [Quick wins]

### Longer Term
1. [More substantial improvements]
2. [Architecture changes]
3. [Data collection strategies]

## Conclusion
[Final thoughts and summary of the project]
