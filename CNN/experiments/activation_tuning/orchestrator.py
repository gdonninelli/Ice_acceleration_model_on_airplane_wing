import os
import subprocess
import time

CANDIDATES = [
    "relu", "tanh", "sigmoid",
    "leakyrelu-0.01", "leakyrelu-0.05", "leakyrelu-0.1", "leakyrelu-0.2", "leakyrelu-0.3"
]

RESULTS_DIR = "results/cross_validation/activation_tuning"
STATE_FILE = "activation_tuning_state.txt"

os.makedirs(RESULTS_DIR, exist_ok=True)

def read_state():
    if not os.path.exists(STATE_FILE):
        return set()
    with open(STATE_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def write_state(done_candidates):
    with open(STATE_FILE, "w") as f:
        for c in sorted(list(done_candidates)):
            f.write(c + "\n")

done = read_state()

print("===== Activation Tuning Orchestrator =====")
for candidate in CANDIDATES:
    csv_path = os.path.join(RESULTS_DIR, f"{candidate}.csv")
    if candidate in done and os.path.exists(csv_path):
        print(f"[{candidate}] Already done, skipping.")
        continue

    print(f"[{candidate}] Starting run...")
    start_time = time.time()
    
    cmd = ["mpirun", "-n", "8", "--oversubscribe", "build/experiments/activation_tuning", candidate]
    
    # We must ensure PATH has openmpi
    env = os.environ.copy()
    env["PATH"] = "/usr/lib64/openmpi/bin:" + env.get("PATH", "")
    
    try:
        subprocess.run(cmd, env=env, check=True)
        elapsed = time.time() - start_time
        print(f"[{candidate}] Finished in {elapsed:.1f} seconds.")
        
        # Verify CSV was created
        if os.path.exists(csv_path):
            done.add(candidate)
            write_state(done)
        else:
            print(f"[{candidate}] ERROR: Process succeeded but {csv_path} was not created!")
            break
            
    except subprocess.CalledProcessError as e:
        print(f"[{candidate}] ERROR: Process failed with exit code {e.returncode}")
        break

print("\n===== Orchestrator finished =====")
missing = [c for c in CANDIDATES if c not in done]
if missing:
    print(f"Missing candidates: {missing}")
else:
    print("All candidates completed successfully!")

    # Generate aggregated CSV
    agg_path = os.path.join(RESULTS_DIR, "aggregated.csv")
    with open(agg_path, "w") as out:
        out.write("candidate,fold,train_mse,val_mse,baseline_mse,epochs\n")
        for c in CANDIDATES:
            with open(os.path.join(RESULTS_DIR, f"{c}.csv"), "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    for line in lines[1:]:
                        out.write(line)
    print(f"Aggregated CSV written to {agg_path}")

