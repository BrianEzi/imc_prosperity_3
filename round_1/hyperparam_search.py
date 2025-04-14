import argparse
from bayes_opt import BayesianOptimization
import subprocess
import re
import os
import time
import csv

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--version", type=int, required=True, help="Strategy version number (e.g. 3 for submission_v3.py)")
args = parser.parse_args()

version = args.version

BEST_FILE = f"best_hyperparams_v{version}.py"
LOG_FILE = f"hyperparam_log_v{version}.csv"

def write_config(params):
    with open("config_hyper.py", "w") as f:
        f.write("HYPERPARAMS = {\n")
        for key, value in params.items():
            f.write(f"    '{key}': {round(value, 4) if isinstance(value, float) else int(value)},\n")
        f.write("}\n")

def get_best_profit():
    if not os.path.exists(BEST_FILE):
        return -1
    with open(BEST_FILE, "r") as f:
        content = f.read()
        match = re.search(r"Profit:\s*(\d+)", content)
        return int(match.group(1)) if match else -1

def write_best(params, profit):
    with open(BEST_FILE, "w") as f:
        f.write("BEST_HYPERPARAMS = {\n")
        for key, value in params.items():
            f.write(f"    '{key}': {round(value, 4) if isinstance(value, float) else int(value)},\n")
        f.write("}\n")
        f.write(f"Profit: {int(profit)}\n")

def run_backtest():
    try:
        result = subprocess.run(
            ["prosperity3bt", f".\\submission_v{version}.py", "1"],
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout
        matches = re.findall(r"Total profit:\s*([\d,]+)", output)
        if matches:
            return int(matches[-1].replace(",", ""))
        else:
            print("Failed to parse profit. Raw output:\n", output)
            return 0
    except Exception as e:
        print("Backtest failed:", str(e))
        return 0

def log_to_csv(params, profit):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as csvfile:
        fieldnames = [*params.keys(), "profit"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        params_copy = params.copy()
        params_copy["profit"] = profit
        writer.writerow(params_copy)

def objective(**kwargs):
    for int_param in ["LATE_DAY_TIMESTAMP", "SQUID_VOL_THRESHOLD", "SQUID_SHORT_EMA_WINDOW", "SQUID_LONG_EMA_WINDOW"]:
        if int_param in kwargs.keys():
            kwargs[int_param] = int(round(kwargs[int_param]))
    write_config(kwargs)
    print(f"Testing: {kwargs}")
    profit = run_backtest()
    print(f"→ Profit: {profit}\n")
    log_to_csv(kwargs, profit)
    if profit > get_best_profit():
        print("New best! Saving to best_hyperparams.py...\n")
        write_best(kwargs, profit)
    return profit

# All tunable parameter bounds
pbounds = {
    # # General
    # "LATE_DAY_TIMESTAMP": (800000, 900000),
    # "LATE_DAY_SIZE_FACTOR": (0.3, 0.8),
    # "LATE_DAY_SPREAD_FACTOR": (1.2, 2.0),

    # Resin
    "RESIN_TAKE_WIDTH": (1, 10),
    "RESIN_MM_EDGE": (1, 10),

    # Kelp
    "KELP_TAKE_WIDTH": (1, 10),
    "KELP_MM_EDGE": (1, 10),
    "KELP_BETA": (0, 1),

    # # Squid Ink
    # "SQUID_TAKE_WIDTH": (5.0, 10.0),
    # "SQUID_MM_EDGE": (8.0, 12.0),
    # "SQUID_BETA": (0.5, 0.7),
    # "SQUID_VOL_THRESHOLD": (5, 20),
    # "SQUID_TREND_THRESHOLD": (1.0, 2.0),
    # "SQUID_TREND_BIAS": (0.4, 1.0),
    # "SQUID_SHORT_EMA_WINDOW": (3, 10),
    # "SQUID_LONG_EMA_WINDOW": (10, 40),
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

start = time.time()

optimizer.maximize(
    init_points=5,
    n_iter=75,
)

end = time.time()

print(f'Took {end-start} seconds.')
print("All done! Best result:")
print(optimizer.max)
