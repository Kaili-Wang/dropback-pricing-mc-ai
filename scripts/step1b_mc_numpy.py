import math
import random
import time
import csv
import os
import numpy as np

# Directory configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_dir = os.path.join(project_root, 'data')
os.makedirs(data_dir, exist_ok=True)

# Base parameters
r = 0.02
q = 0.018
T = 3.0
steps_per_year = 252
N = int(T * steps_per_year)
dt = 1.0 / steps_per_year

S0 = 3790.38
initial_investment = 550.0
additional_investment = 150.0
initial_cash = 450.0
cash_rate = 0.0985  

trigger_levels = np.array([0.90 * S0, 0.85 * S0, 0.80 * S0])
num_paths = 50000
fixed_seed = 2026
split_seed = 8888

sigma_start = 0.1501
sigma_step = 0.0001
num_sigma = 3000

discount_factor = math.exp(-r * T)

def _generate_z_matrix():
    """
    Pre-computes the random sequence using the standard library's random module.
    This forces the vectorized implementation to use the exact same paths as the base model,
    eliminating the noise caused by different random number generation algorithms.
    """
    z_mat = np.zeros((num_paths, N))
    for p in range(num_paths):
        local_rng = random.Random(fixed_seed + p)
        for i in range(N):
            z_mat[p, i] = local_rng.gauss(0.0, 1.0)
    return z_mat

# Initialize the Z-matrix globally once
Z_matrix = _generate_z_matrix()

def get_price_for_sigma_vectorized(sigma):
    drift_part = (r - q - 0.5 * sigma ** 2) * dt
    diffusion_coef = sigma * math.sqrt(dt)

    exponents = np.clip(drift_part + diffusion_coef * Z_matrix, -50, 50)
    log_returns = np.cumsum(exponents, axis=1)
    S_paths = S0 * np.exp(log_returns)

    step_hit = np.full((3, num_paths), N)
    hit_masks = np.zeros((3, num_paths), dtype=bool)
    entry_levels = np.zeros((3, num_paths))
    
    for i in range(3):
        hits = S_paths <= trigger_levels[i]
        hit_any = hits.any(axis=1)
        
        hit_masks[i] = hit_any
        if hit_any.any():
            first_hit_indices = hits.argmax(axis=1)
            step_hit[i, hit_any] = first_hit_indices[hit_any] + 1
            entry_levels[i, hit_any] = S_paths[hit_any, first_hit_indices[hit_any]]

    base_interest = initial_cash * cash_rate * dt * N
    lost_interest = np.sum((N - step_hit) * additional_investment * cash_rate * dt, axis=0)
    accrued_interest_vec = base_interest - lost_interest
    
    num_triggers_hit = np.sum(hit_masks, axis=0)
    final_cash = initial_cash - additional_investment * num_triggers_hit
    cash_part = final_cash + accrued_interest_vec

    S_T = S_paths[:, -1]
    equity_part = initial_investment * (S_T / S0)
    
    for i in range(3):
        mask = hit_masks[i]
        if mask.any():
            equity_part[mask] += additional_investment * (S_T[mask] / entry_levels[i, mask])

    path_values = (equity_part + cash_part) * discount_factor
    return float(np.mean(path_values))

def main():
    all_sigmas = [round(sigma_start + i * sigma_step, 6) for i in range(num_sigma)]
    training_group, evaluation_group = [], []

    random.seed(split_seed)
    for i in range(0, num_sigma, 3):
        batch = all_sigmas[i:i + 3]
        if len(batch) < 3:
            training_group.extend(batch)
            continue
        
        train_batch = random.sample(batch, 2)
        eval_batch = [s for s in batch if s not in train_batch]
        training_group.extend(train_batch)
        evaluation_group.extend(eval_batch)

    datasets = [
        ("train_numpy.csv", training_group),
        ("eval_numpy.csv", evaluation_group)
    ]

    print(f"Starting data generation (NumPy Vectorized)...")
    print(f"Total samples to compute: {len(all_sigmas)}")

    for file_name, group in datasets:
        save_path = os.path.join(data_dir, file_name)
        start_t = time.time()
        
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sigma", "price"])
            for sigma in group:
                price = get_price_for_sigma_vectorized(sigma)
                writer.writerow([f"{sigma:.6f}", f"{price:.6f}"])
        
        end_t = time.time()
        print(f"Successfully saved {len(group)} samples to {file_name} (Time: {end_t - start_t:.2f}s)")

    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()