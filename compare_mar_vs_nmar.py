"""Compare MAR vs NMAR per-category metrics."""
import pandas as pd
import numpy as np
import os

# Read MAR results
mar_file = "results_rep10/category_metrics_summary_MAR.csv"
mar_df = pd.read_csv(mar_file)
mar_df = mar_df[['method', 'category', 'category_name', 'accuracy_mean', 'logloss_mean', 'brier_mean']]
mar_df.columns = ['method', 'category', 'category_name', 'accuracy', 'logloss', 'brier']
mar_df['setting'] = 'MAR'

# Read NMAR results
nmar_file = "results_mnar_rep10/category_metrics_summary_NMAR_alpha2.0.csv"
nmar_df = pd.read_csv(nmar_file)
nmar_df = nmar_df[['method', 'category', 'category_name', 'accuracy_mean', 'logloss_mean', 'brier_mean']]
nmar_df.columns = ['method', 'category', 'category_name', 'accuracy', 'logloss', 'brier']
nmar_df['setting'] = 'NMAR (α=2.0)'

# Combine for easy comparison
combined_df = pd.concat([mar_df, nmar_df], ignore_index=True)

print("="*140)
print("MAR vs NMAR COMPARISON - Per-Category Metrics")
print("="*140)

# Print full comparison table
print("\nFull Comparison (MAR vs NMAR):")
print("-"*140)
pivot_acc = combined_df.pivot_table(
    index=['method', 'category', 'category_name'],
    columns='setting',
    values='accuracy'
).round(4)
pivot_acc['Diff (NMAR-MAR)'] = (pivot_acc['NMAR (α=2.0)'] - pivot_acc['MAR']).round(4)
print("\nAccuracy:")
print(pivot_acc)

pivot_ll = combined_df.pivot_table(
    index=['method', 'category', 'category_name'],
    columns='setting',
    values='logloss'
).round(4)
pivot_ll['Diff (NMAR-MAR)'] = (pivot_ll['NMAR (α=2.0)'] - pivot_ll['MAR']).round(4)
print("\nLog Loss:")
print(pivot_ll)

pivot_br = combined_df.pivot_table(
    index=['method', 'category', 'category_name'],
    columns='setting',
    values='brier'
).round(4)
pivot_br['Diff (NMAR-MAR)'] = (pivot_br['NMAR (α=2.0)'] - pivot_br['MAR']).round(4)
print("\nBrier Score:")
print(pivot_br)

# Summary by Category (averaged across methods)
print("\n" + "="*140)
print("Summary by Category (averaged across methods):")
print("="*140)

for metric in ['accuracy', 'logloss', 'brier']:
    pivot = combined_df.pivot_table(
        index='category_name',
        columns='setting',
        values=metric
    ).round(4)
    pivot['Diff (NMAR-MAR)'] = (pivot['NMAR (α=2.0)'] - pivot['MAR']).round(4)
    print(f"\n{metric.upper()}:")
    print(pivot)

# Summary by Method (averaged across categories)
print("\n" + "="*140)
print("Summary by Method (averaged across categories):")
print("="*140)

for metric in ['accuracy', 'logloss', 'brier']:
    pivot = combined_df.pivot_table(
        index='method',
        columns='setting',
        values=metric
    ).round(4)
    pivot['Diff (NMAR-MAR)'] = (pivot['NMAR (α=2.0)'] - pivot['MAR']).round(4)
    print(f"\n{metric.upper()}:")
    print(pivot)

# Save detailed comparison
print("\n" + "="*140)

# Create detailed comparison table for export
comparison_data = []
for method in combined_df['method'].unique():
    for cat in sorted(combined_df['category'].unique()):
        mar_row = mar_df[(mar_df['method'] == method) & (mar_df['category'] == cat)]
        nmar_row = nmar_df[(nmar_df['method'] == method) & (nmar_df['category'] == cat)]

        if not mar_row.empty and not nmar_row.empty:
            cat_name = mar_row.iloc[0]['category_name']
            mar_acc = mar_row.iloc[0]['accuracy']
            nmar_acc = nmar_row.iloc[0]['accuracy']
            mar_ll = mar_row.iloc[0]['logloss']
            nmar_ll = nmar_row.iloc[0]['logloss']
            mar_br = mar_row.iloc[0]['brier']
            nmar_br = nmar_row.iloc[0]['brier']

            comparison_data.append({
                'method': method,
                'category': cat,
                'category_name': cat_name,
                'accuracy_MAR': mar_acc,
                'accuracy_NMAR': nmar_acc,
                'accuracy_diff': nmar_acc - mar_acc,
                'logloss_MAR': mar_ll,
                'logloss_NMAR': nmar_ll,
                'logloss_diff': nmar_ll - mar_ll,
                'brier_MAR': mar_br,
                'brier_NMAR': nmar_br,
                'brier_diff': nmar_br - mar_br,
            })

comparison_df = pd.DataFrame(comparison_data)
out_file = "results_rep10/comparison_MAR_vs_NMAR.csv"
comparison_df.to_csv(out_file, index=False)
print(f"Saved detailed comparison: {out_file}")

# Print key insights
print("\n" + "="*140)
print("KEY INSIGHTS:")
print("="*140)

# Methods that benefit most from NMAR
print("\nMethods with largest accuracy improvement under NMAR (positive = better under NMAR):")
method_improvement = comparison_df.groupby('method')['accuracy_diff'].mean().sort_values(ascending=False)
for method, diff in method_improvement.items():
    print(f"  {method:30s}: {diff:+.4f}")

# Categories most affected by NMAR
print("\nCategories most affected by NMAR (largest accuracy changes):")
cat_impact = comparison_df.groupby('category_name')['accuracy_diff'].apply(lambda x: abs(x).mean()).sort_values(ascending=False)
for cat, diff in cat_impact.items():
    print(f"  {cat[:60]:60s}: {diff:.4f}")

