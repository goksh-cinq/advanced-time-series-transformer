import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("energy_time_series.csv")

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Sliding window parameters
SEQ_LEN = 48      # past time steps
HORIZON = 12      # future prediction steps

X, y = [], []

for i in range(len(data_scaled) - SEQ_LEN - HORIZON):
    X.append(data_scaled[i:i + SEQ_LEN])
    y.append(data_scaled[i + SEQ_LEN:i + SEQ_LEN + HORIZON, :3])

X = np.array(X)
y = np.array(y)

# Train-Test Split
split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

# Save processed arrays
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Preprocessing completed")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)
