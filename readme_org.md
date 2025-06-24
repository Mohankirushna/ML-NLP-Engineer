# 🤖 ML/NLP Engineer Intern Challenge

## 🎯 Objective
Build a complete text classification pipeline using Hugging Face Transformers. Demonstrate your skills in NLP preprocessing, model fine-tuning, and evaluation.

## 📋 Task Overview
1. **Select** a small labeled text dataset (e.g., movie reviews, sentiment analysis)
2. **Preprocess** and tokenize using Hugging Face Transformers
3. **Fine-tune** a pre-trained model (DistilBERT recommended)
4. **Evaluate** using F1, precision, recall metrics
5. **Document** insights and improvement ideas
6. **Bonus**: Extend to multilingual use case

## 📁 Project Structure

### `/notebooks/`
- **`data_exploration.ipynb`** - Dataset analysis, class distribution, sample exploration
- **`model_training.ipynb`** - Interactive model training and experimentation
- **`evaluation_analysis.ipynb`** - Results analysis, error analysis, visualizations

### `/src/`
- **`train_model.py`** - Main training script with Hugging Face Trainer
- **`data_preprocessing.py`** - Text cleaning, tokenization, dataset preparation
- **`model_utils.py`** - Model loading, saving, prediction utilities
- **`config.py`** - Training hyperparameters and model configurations

### `/models/`
- **`trained_model/`** - Fine-tuned model weights and configuration
- **`tokenizer/`** - Trained tokenizer files
- **`.gitkeep`** - Maintains directory structure

### `/reports/`
- **`model_report.md`** - Model architecture decisions, training insights, improvements
- **`evaluation_metrics.json`** - Detailed metrics (F1, precision, recall, accuracy)
- **`confusion_matrix.png`** - Classification results visualization

### Root Files
- **`requirements.txt`** - Python dependencies (transformers, torch, datasets, etc.)
- **`README.md`** - Project documentation (this file)
- **`submission.md`** - Your approach, model decisions, and key learnings
- **`train.py`** - Simple training script entry point
- **`.gitignore`** - Files to exclude from git (models/, __pycache__, etc.)

## 🚀 Getting Started

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Data Exploration**
   ```bash
   jupyter notebook notebooks/data_exploration.ipynb
   ```

3. **Train Model**
   ```bash
   python train.py
   # or
   python src/train_model.py
   ```

4. **Evaluate Results**
   ```bash
   jupyter notebook notebooks/evaluation_analysis.ipynb
   ```

## 📊 Dataset Requirements
Choose a text classification dataset with:
- 1000+ labeled samples
- 2+ classes (binary or multi-class)
- English text (bonus: multilingual)

**Suggested datasets**: 
- IMDB Movie Reviews (sentiment)
- AG News (topic classification)
- Yelp Reviews (sentiment)
- Twitter Sentiment datasets

## ✅ Expected Deliverables

1. **Working fine-tuned model** with saved weights
2. **Training pipeline** using Hugging Face Transformers
3. **Evaluation metrics** (F1, precision, recall) in JSON format
4. **Model report** explaining architecture choices and insights
5. **Clean notebooks** showing exploration and analysis
6. **Updated `submission.md`** with approach and learnings

## 🎯 Evaluation Focus
- **Model selection** and fine-tuning approach
- **Data preprocessing** and tokenization quality
- **Evaluation methodology** and metrics interpretation
- **Code organization** and reproducibility
- **Documentation** and insights quality

## 💡 Bonus Points
- Multilingual extension or cross-lingual transfer
- Advanced evaluation (ROC curves, per-class analysis)
- Model comparison (multiple architectures)
- Deployment-ready inference pipeline
- Error analysis and failure case studies

## 🔧 Key Technologies
- **Hugging Face Transformers** - Model training and inference
- **Datasets library** - Data loading and preprocessing
- **PyTorch** - Deep learning framework
- **Scikit-learn** - Evaluation metrics
- **Matplotlib/Seaborn** - Visualization

---

**Time Estimate**: 4-6 hours | **Due**: June 26, 2025, 11:59 PM IST
