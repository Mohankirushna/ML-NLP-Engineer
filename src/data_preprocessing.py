from datasets import load_dataset, Dataset, DatasetDict
from transformers import DistilBertTokenizer
from typing import Dict, List, Tuple, Optional, Union
import torch
from config import data_config, model_config  
import numpy as np
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Handles all data loading and preprocessing for the IMDB dataset."""
    
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_config.model_name)
        self.text_column = data_config.text_column
        self.label_column = data_config.label_column
        self.max_length = model_config.max_length
        
    def load_dataset(self) -> DatasetDict:
        """Load the IMDB dataset from Hugging Face datasets."""
        print("Loading IMDB dataset...")
        dataset = load_dataset('imdb')
        
        # If max_samples is set, use a subset for debugging
        if data_config.max_samples:
            dataset = dataset.shuffle(seed=model_config.seed).select(range(data_config.max_samples))
            
        return dataset
    
    def tokenize_function(self, examples: Dict[str, list]) -> Dict[str, list]:
        """Tokenize the text data."""
        return self.tokenizer(
            examples[self.text_column],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True
        )
    
    def preprocess_data(self, dataset: Dataset) -> Dataset:
        """Preprocess the dataset with tokenization."""
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=[self.text_column]
        )
        
        # Set format for PyTorch
        tokenized_dataset.set_format(
            type='torch', 
            columns=['input_ids', 'attention_mask', 'label']
        )
        
        return tokenized_dataset
        
        return tokenized_dataset
    
    def get_data_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Get train, validation, and test splits."""
        # Load the dataset
        dataset = self.load_dataset()
        
        # Split the training set into train and validation
        train_val = dataset['train'].train_test_split(
            test_size=data_config.test_size,
            seed=model_config.seed
        )
        
        # Preprocess each split
        train_dataset = self.preprocess_data(train_val['train'])
        val_dataset = self.preprocess_data(train_val['test'])
        test_dataset = self.preprocess_data(dataset['test'])
        
        return train_dataset, val_dataset, test_dataset
    
    def get_class_weights(self, dataset: Dataset) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        labels = dataset[self.label_column]
        class_counts = np.bincount(labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
        return class_weights / class_weights.sum()

def prepare_data_for_training() -> Tuple[Dataset, Dataset, Dataset, DistilBertTokenizer]:
    """
    Main function to prepare data for training.
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, tokenizer)
    """
    preprocessor = DataPreprocessor()
    train_dataset, val_dataset, test_dataset = preprocessor.get_data_splits()
    return train_dataset, val_dataset, test_dataset, preprocessor.tokenizer
