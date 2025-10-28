# CS-657 Assignment 2 – Molecular SA (80/20)

- Loads MOSES fingerprint batches from `./moses_molecule_batches_sa`.
- Relabels molecules via the global 80th percentile of `SA_score`, then trains multiple models with different imbalance handling strategies.
- Implements class weighting, random undersampling, and random oversampling to handle class imbalance.
- Reports PR-AUC, ROC-AUC, Balanced Accuracy, MCC, Precision, Recall, F1, and training time.
- Generates PR curves and calibration curves as visualizations.
- Performs scaling experiments (100k, 500k, 1M rows) to analyze performance vs. dataset size.
- Writes comprehensive metrics, visualizations, and a detailed report for submission.

## Run

```bash
spark-submit --driver-memory 6g assignment2_sa_local.py
```

## Outputs

Generated inside `outputs/` (overwritten per run):
- `metrics_valid.json`, `metrics_test.json` – JSON metrics
- `metrics_valid.csv`, `metrics_test.csv` – CSV metrics tables
- `class_balance_overall.json`, `class_balance_by_split.json` – class distribution data
- `split_stats.txt` – split statistics summary
- `calibration_valid.csv` – calibration bin data
- `pr_curve_valid.png` – PR curve visualization
- `calibration_curve_valid.png` – calibration curve visualization
- `scaling_results.csv`, `scaling_results.json` – scaling experiment results

Other artifacts:
- `REPORT.md` – comprehensive report with intro, methods, results, and discussion.
- Spark logs in the console; check cached counts for sanity.

## Notes

- Script assumes the dataset is in `./moses_molecule_batches_sa` relative to the project directory.
- Update `DATA_DIR` constant in the script if your dataset location differs.
- Ensure matplotlib is installed for visualization generation: `pip install matplotlib>=3.5.0`
- Full run with scaling experiments may take 20-30 minutes depending on hardware.
