"""Compute per-category metrics from saved predictions and aggregate."""
import numpy as np
import pandas as pd
import glob
import os
from sklearn.metrics import accuracy_score, log_loss

# Find all prediction files
pred_files = sorted(glob.glob("results_mnar_rep*/predictions_*.npz"))
print(f"Found {len(pred_files)} prediction files\n")

# Aggregate results across repeats
all_results = []

for pred_file in pred_files:
    print(f"Processing: {pred_file}")

    data = np.load(pred_file, allow_pickle=True)
    y_test = data['y_test']
    labels = data['labels']
    K = len(labels)

    # Extract predictions for each method
    methods = set()
    for key in data.files:
        if key.endswith('__prob'):
            method = key.replace('__prob', '')
            methods.add(method)

    # Compute per-category metrics
    for method in sorted(methods):
        prob = data[f'{method}__prob']
        y_pred = prob.argmax(axis=1)

        for k in range(K):
            mask_k = y_test == k
            if mask_k.sum() > 0:
                acc_k = accuracy_score(y_test[mask_k], y_pred[mask_k])
                ll_k = log_loss(y_test[mask_k], prob[mask_k], labels=list(range(K)))
                brier_k = np.mean((y_pred[mask_k] != y_test[mask_k]).astype(float))

                all_results.append({
                    'method': method,
                    'category': k,
                    'category_name': labels[k],
                    'accuracy': acc_k,
                    'logloss': ll_k,
                    'brier': brier_k,
                })

# Create aggregated dataframe
df_all = pd.DataFrame(all_results)

# Compute mean and std by method and category
summary = df_all.groupby(['method', 'category', 'category_name']).agg({
    'accuracy': ['mean', 'std'],
    'logloss': ['mean', 'std'],
    'brier': ['mean', 'std'],
}).round(4)

summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary = summary.reset_index()

# Save aggregated results
output_dir = "results_mnar_rep10"
os.makedirs(output_dir, exist_ok=True)

# Save full per-repeat results
full_file = os.path.join(output_dir, "category_metrics_by_repeat_NMAR_alpha2.0.csv")
df_all.to_csv(full_file, index=False)
print(f"\nSaved full results (all repeats): {full_file}")

# Save aggregated summary
summary_file = os.path.join(output_dir, "category_metrics_summary_NMAR_alpha2.0.csv")
summary.to_csv(summary_file, index=False)
print(f"Saved summary (mean/std): {summary_file}")

# Print summary table
print("\n" + "="*100)
print("NMAR (α=2.0) - Per-Category Evaluation Summary")
print("="*100)
print(summary.to_string(index=False))
