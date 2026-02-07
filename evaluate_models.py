import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from transformer_model import TimeSeriesTransformer
from lstm_model import LSTMModel

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Load Transformer model
transformer = TimeSeriesTransformer(
    input_dim=5,
    model_dim=64,
    num_heads=4,
    num_layers=2,
    horizon=12,
    output_dim=3
)
transformer.load_state_dict(torch.load("transformer_model.pth"))
transformer.eval()

# Load LSTM model
lstm = LSTMModel(
    input_dim=5,
    hidden_dim=64,
    horizon=12,
    output_dim=3
)
lstm.load_state_dict(torch.load("lstm_model.pth"))
lstm.eval()

# Predictions
with torch.no_grad():
    transformer_preds = transformer(X_test).numpy()
    lstm_preds = lstm(X_test).numpy()
    actual = y_test.numpy()

# Metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    rmse = np.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))
    return mae, rmse

t_mae, t_rmse = calculate_metrics(actual, transformer_preds)
l_mae, l_rmse = calculate_metrics(actual, lstm_preds)

# Results

print("\n===== MODEL COMPARISON RESULTS =====\n")
print(f"Transformer -> MAE: {t_mae:.4f}, RMSE: {t_rmse:.4f}")
print(f"LSTM        -> MAE: {l_mae:.4f}, RMSE: {l_rmse:.4f}")

print("\nEvaluation completed")
