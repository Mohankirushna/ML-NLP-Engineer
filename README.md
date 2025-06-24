# Text Classification Project

This project implements a text classification model using state-of-the-art transformer models from Hugging Face.

## Project Structure

```
.
├── notebooks/                  # Jupyter notebooks for exploration and analysis
├── src/                        # Source code
│   ├── train_model.py          # Main training script
│   ├── data_preprocessing.py   # Data processing utilities
│   ├── model_utils.py         # Model utilities
│   └── config.py              # Configuration settings
├── models/                     # Trained models and tokenizers
├── reports/                    # Evaluation reports and visualizations
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run Jupyter Lab for exploration:
   ```bash
   jupyter lab
   ```

## Usage

### Training
```bash
python src/train_model.py
```

### Evaluation
```bash
python src/evaluate.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
