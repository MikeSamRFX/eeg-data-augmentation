import os
import pandas as pd
import numpy as np
from pathlib import Path
import config

def digest_results(input_csv=None, output_csv=None):
    """
    Reads fold-level experiment results CSV, aggregates metrics across folds (Mean ± Std),
    saves the summarized metrics to output_csv and prints a formatted summary table.
    """
    if input_csv is None:
        input_csv = config.RESULTS_FILE
    if output_csv is None:
        output_csv = config.BASE_DIR.parent / "Digested_results.csv"

    if not os.path.exists(input_csv):
        print(f" Input results file '{input_csv}' does not exist.")
        return None

    df = pd.read_csv(input_csv)
    if df.empty:
        print(f" Input results file '{input_csv}' is empty.")
        return None

    # Keep only the most recent entry for each (Model, Data_Source, Fold)
    df = df.drop_duplicates(subset=["Model", "Data_Source", "Fold"], keep="last")

    # Group by Model and Data_Source
    grouped = df.groupby(["Model", "Data_Source"])

    digested_rows = []
    for (model, data_source), group in grouped:
        acc_mean = group["Accuracy"].mean()
        acc_std = group["Accuracy"].std()
        prec_mean = group["Precision"].mean()
        prec_std = group["Precision"].std()
        rec_mean = group["Recall_ASD"].mean()
        rec_std = group["Recall_ASD"].std()
        f1_mean = group["F1_Score"].mean()
        f1_std = group["F1_Score"].std()
        num_folds = len(group)

        digested_rows.append({
            "Model": model,
            "Data_Source": data_source,
            "Folds_Count": num_folds,
            "Accuracy_Mean": round(acc_mean, 4),
            "Accuracy_Std": round(acc_std, 4) if not np.isnan(acc_std) else 0.0,
            "Precision_Mean": round(prec_mean, 4),
            "Precision_Std": round(prec_std, 4) if not np.isnan(prec_std) else 0.0,
            "Recall_Mean": round(rec_mean, 4),
            "Recall_Std": round(rec_std, 4) if not np.isnan(rec_std) else 0.0,
            "F1_Score_Mean": round(f1_mean, 4),
            "F1_Score_Std": round(f1_std, 4) if not np.isnan(f1_std) else 0.0,
            "Acc_Formatted": f"{acc_mean*100:.2f}% ± {acc_std*100:.2f}%" if not np.isnan(acc_std) else f"{acc_mean*100:.2f}%",
            "F1_Formatted": f"{f1_mean*100:.2f}% ± {f1_std*100:.2f}%" if not np.isnan(f1_std) else f"{f1_mean*100:.2f}%",
        })

    digested_df = pd.DataFrame(digested_rows)
    digested_df.to_csv(output_csv, index=False)
    print(f"\n [DIGESTED RESULTS SAVED] -> {output_csv}\n")

    # Print clean formatted table
    print("=" * 95)
    print(f"{'Model':<15} | {'Data Source':<22} | {'Folds':<5} | {'Accuracy (Mean ± Std)':<22} | {'F1-Score (Mean ± Std)':<22}")
    print("=" * 95)
    for _, row in digested_df.iterrows():
        print(f"{row['Model']:<15} | {row['Data_Source']:<22} | {row['Folds_Count']:<5} | {row['Acc_Formatted']:<22} | {row['F1_Formatted']:<22}")
    print("=" * 95)

    return digested_df

if __name__ == "__main__":
    digest_results()
