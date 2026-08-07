import pandas as pd
import numpy as np

df = pd.read_csv("results/cross_validation/activation_tuning/aggregated.csv")

baseline = "leakyrelu-0.05"

# Calculate ratios
df["ratio"] = df["val_mse"] / df["baseline_mse"]

# Group by candidate
stats = df.groupby("candidate").agg(
    mean_val_mse=("val_mse", "mean"),
    std_val_mse=("val_mse", "std"),
    mean_ratio=("ratio", "mean")
).reset_index()

baseline_df = df[df["candidate"] == baseline].sort_values("fold")
baseline_vals = baseline_df["val_mse"].values

results = []

for cand in df["candidate"].unique():
    cand_df = df[df["candidate"] == cand].sort_values("fold")
    cand_vals = cand_df["val_mse"].values
    
    paired_diffs = cand_vals - baseline_vals
    
    mean_diff = np.mean(paired_diffs)
    std_diff = np.std(paired_diffs, ddof=1) if len(paired_diffs) > 1 else 0.0
    
    # improved = negative difference (lower MSE is better)
    improved_folds = sum(1 for d in paired_diffs if d < 0)
    
    results.append({
        "candidate": cand,
        "mean_val_mse": cand_df["val_mse"].mean(),
        "std_val_mse": cand_df["val_mse"].std(),
        "mean_ratio": cand_df["ratio"].mean(),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "improved_folds": improved_folds
    })

res_df = pd.DataFrame(results)

print("| Candidate | Val MSE | Std | Val/Baseline | Paired Diff vs 0.05 | Std | Improved Folds | Beats Baseline? |")
print("|---|---|---|---|---|---|---|---|")

for _, row in res_df.sort_values("mean_val_mse").iterrows():
    cand = row["candidate"]
    
    if cand == baseline:
        diff_str = "—"
        std_str = "—"
        impr_str = "—"
        beats_str = "—"
    else:
        diff_str = f"{row['mean_diff']:+.6f}"
        std_str = f"{row['std_diff']:.6f}"
        impr_str = f"{row['improved_folds']}/5"
        
        beats = (row['improved_folds'] >= 4) and (abs(row['mean_diff']) > row['std_diff']) and (row['mean_diff'] < 0)
        beats_str = "Yes" if beats else "No"
    
    cand_fmt = f"**{cand}**" if cand == baseline else cand
    val_mse = f"{row['mean_val_mse']:.6f}"
    std_mse = f"{row['std_val_mse']:.6f}"
    ratio = f"{row['mean_ratio']:.5f}"
    
    if cand == baseline:
        val_mse = f"**{val_mse}**"
    
    print(f"| {cand_fmt} | {val_mse} | {std_mse} | {ratio} | {diff_str} | {std_str} | {impr_str} | {beats_str} |")

