import math
import random
import time
import csv
import os

# Path configuration
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
cash_rate = 0.0985  # 9.85% simple interest

trigger_levels = [
    0.90 * S0,
    0.85 * S0,
    0.80 * S0
]

num_paths = 50000
fixed_seed = 2026
split_seed = 8888

sigma_start = 0.1501
sigma_step = 0.0001
num_sigma = 3000

discount_factor = math.exp(-r * T)

# Monte Carlo Engine
def get_price_for_sigma(sigma):
    path_values = []

    drift_part = (r - q - 0.5 * sigma ** 2) * dt
    diffusion_coef = sigma * math.sqrt(dt)

    for p in range(num_paths):
        local_rng = random.Random(fixed_seed + p)

        s_t = S0
        cash = initial_cash

        invested_amts = [initial_investment]
        entry_lvls = [S0]

        accrued_interest = 0.0
        triggers_hit = 0

        for _ in range(N):
            z = local_rng.gauss(0.0, 1.0)

            exponent = drift_part + diffusion_coef * z
            exponent = max(min(exponent, 50), -50)

            s_t *= math.exp(exponent)
            accrued_interest += cash * cash_rate * dt

            while triggers_hit < 3 and s_t <= trigger_levels[triggers_hit]:
                cash -= additional_investment
                invested_amts.append(additional_investment)
                entry_lvls.append(s_t)
                triggers_hit += 1

        equity_part = sum(
            amt * (s_t / lvl)
            for amt, lvl in zip(invested_amts, entry_lvls)
        )
        cash_part = cash + accrued_interest

        path_values.append((equity_part + cash_part) * discount_factor)

    return sum(path_values) / len(path_values)

# Dataset generation and splitting
print("Generating and splitting dataset...")

random.seed(split_seed)

all_sigmas = [round(sigma_start + i * sigma_step, 6) for i in range(num_sigma)]

training_group = []
evaluation_group = []

for i in range(0, num_sigma, 3):
    batch = all_sigmas[i:i + 3]

    if len(batch) < 3:
        training_group.extend(batch)
        continue

    train_batch = random.sample(batch, 2)
    eval_batch = [s for s in batch if s not in train_batch]

    training_group.extend(train_batch)
    evaluation_group.extend(eval_batch)

print(f"Training set size: {len(training_group)}")
print(f"Evaluation set size: {len(evaluation_group)}")
print("=" * 50)

# CSV writing
print(">>> Computing and writing train.csv...")

train_path = os.path.join(data_dir, 'train.csv')
with open(train_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sigma", "price"])

    for idx, sigma in enumerate(training_group, 1):
        price = get_price_for_sigma(sigma)
        writer.writerow([f"{sigma:.6f}", f"{price:.6f}"])
        print(f"[Train] Completed {idx}/{len(training_group)} pricing calculations.")

print("train.csv writing completed.")
print(f"Saved at: {train_path}")
print("=" * 50)

print(">>> Computing and writing eval.csv...")
start_time = time.time()

eval_path = os.path.join(data_dir, 'eval.csv')
with open(eval_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sigma", "price"])

    for idx, sigma in enumerate(evaluation_group, 1):
        price = get_price_for_sigma(sigma)
        writer.writerow([f"{sigma:.6f}", f"{price:.6f}"])
        print(f"[Eval] Completed {idx}/{len(evaluation_group)} pricing calculations.")

elapsed_time = time.time() - start_time

print("eval.csv writing completed.")
print(f"Saved at: {eval_path}")

print("=" * 50)
print("Computation Statistics:")
print(f"Evaluation set size: {len(evaluation_group)}")
print(f"Evaluation execution time: {elapsed_time:.4f} seconds")
print("=" * 50)