import numpy as np
import pandas as pd
import time
import os
import joblib

from sklearn.neural_network import MLPRegressor

# Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

train_path = os.path.join(project_root, 'data', 'train.csv')
eval_path = os.path.join(project_root, 'data', 'eval.csv')
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)

model_save_path = os.path.join(models_dir, 'mlp_surrogate_model.pkl')

# Data loading
print("Loading training data...")
train_data = pd.read_csv(train_path)

X_train = train_data[["sigma"]].values
y_train = train_data["price"].values

# Model training
print("Training MLP Surrogate Model...")
model = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation="relu",
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, model_save_path)
print(f"Model successfully saved to: {model_save_path}")

# Model evaluation
eval_data = pd.read_csv(eval_path)
X_eval = eval_data[["sigma"]].values

start_time = time.perf_counter()
ai_prices = model.predict(X_eval)
end_time = time.perf_counter()

ai_pricing_time = end_time - start_time

# Performance metrics
print("=" * 50)
print("AI pricing completed.")
print(f"Total AI pricing time for {len(X_eval)} samples: {ai_pricing_time:.6f} seconds")
print(f"Average time per pricing: {ai_pricing_time / len(X_eval):.8f} seconds")
print("=" * 50)