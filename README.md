# advanced-time-series-transformer
# Advanced Time Series Forecasting with Deep Learning and Explainability

## Project Overview
This project implements a real-world time series forecasting system for energy demand prediction using deep learning models. Two architectures are explored:
1. LSTM-based sequence model
2. Transformer-based attention model

The goal is to compare traditional recurrent models with modern attention-based Transformers and analyze their performance and interpretability.

---

## Dataset Description
A synthetic but realistic energy time series dataset is generated using domain-inspired signals:
- Hourly timestamps (3000 time steps)
- Features:
  - Temperature (seasonal sinusoidal pattern + noise)
  - Industrial activity trend
  - Time index
- Target:
  - Energy load with trend, daily seasonality, and noise

This simulates a real-world energy forecasting scenario.

---

## Data Pipeline
1. `generate_data.py` – Generates the raw time series dataset
2. `preprocess_data.py` – Normalizes data and creates supervised learning windows
3. Train/Test split using NumPy arrays

---

## Models Implemented

### LSTM Model
- Sequence-to-one prediction
- Captures temporal dependencies
- Implemented in `lstm_model.py`

### Transformer Model
- Input embedding layer
- Positional encoding
- Multi-head self-attention layers
- Feed-forward blocks
- Implemented in `transformer_model.py`

The Transformer leverages attention mechanisms to learn long-range dependencies more effectively than LSTMs.

---

## Training Configuration
All hyperparameters are centralized in `config.py`:
- Epochs
- Learning rate
- Batch size
- Model dimensions

This ensures reproducibility and clean experiment management.

---

## Model Evaluation
Models are evaluated using Mean Squared Error (MSE).
`evaluate_models.py` compares LSTM and Transformer performance on the test set.

---

## Explainability and Attention Analysis
The Transformer model uses self-attention to weigh different time steps when making predictions.

Attention weights can be extracted from the attention layers to:
- Understand which historical time steps influence predictions
- Improve model transparency
- Support explainable AI (XAI) principles in time series forecasting

This makes the model suitable for real-world decision-making systems.

---

## Conclusion
This project demonstrates how attention-based Transformers outperform traditional LSTMs in complex time series forecasting tasks while also offering interpretability through attention mechanisms.

---

## How to Run
```bash
pip install -r requirements.txt
python generate_data.py
python preprocess_data.py
python train_lstm.py
python train_transformer.py
python evaluate_models.py
