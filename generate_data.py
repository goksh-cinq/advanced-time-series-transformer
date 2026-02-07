import numpy as np
import pandas as pd

# reproducibility
np.random.seed(42)

# number of time steps (hours)
T = 3000
time = np.arange(T)

# external features
temperature = 10 + 15 * np.sin(2 * np.pi * time / 365) + np.random.normal(0, 2, T)
industrial_activity = 50 + 0.01 * time + np.random.normal(0, 1.5, T)

# target variables
energy_load = (
    200
    + 0.05 * time
    + 20 * np.sin(2 * np.pi * time / 24)     # daily seasonality
    + 10 * np.sin(2 * np.pi * time / 168)    # weekly seasonality
    - 0.7 * temperature
    + 0.4 * industrial_activity
    + np.random.normal(0, 3, T)
)

reactive_power = 0.3 * energy_load + np.random.normal(0, 2, T)
grid_frequency = 50 + 0.01 * np.sin(energy_load / 50) + np.random.normal(0, 0.05, T)

# create dataframe
data = pd.DataFrame({
    "energy_load": energy_load,
    "reactive_power": reactive_power,
    "grid_frequency": grid_frequency,
    "temperature": temperature,
    "industrial_activity": industrial_activity
})

# save dataset
data.to_csv("energy_time_series.csv", index=False)

print("Dataset created successfully")
print(data.head())
