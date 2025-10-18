# CS-657 Assignment 2 – Molecular SA (80/20)

- Loads MOSES fingerprint batches from `/Users/vishaknandakumar/Documents/Masters/College/Fall25/CS-657/Assignment2/moses_molecule_batches_sa`.
- Relabels molecules via the global 80th percentile of `SA_score`, then trains: Logistic Regression, Random Forest, and class-weighted Logistic Regression.
- Reports PR-AUC, ROC-AUC, Balanced Accuracy, MCC, training time, selects the best model on VALID PR-AUC, and evaluates it on TEST.
- Writes split stats, metrics, class balance summaries, and a 10-bin calibration table for rubric-ready reporting.

## Run

```bash
spark-submit --driver-memory 6g assignment2_sa_local.py
```

## Outputs

Generated inside `outputs/` (overwritten per run):
- `metrics_valid.json`
- `metrics_test.json`
- `class_balance_overall.json`
- `class_balance_by_split.json`
- `split_stats.txt`
- `calibration_valid.csv`

Other artifacts:
- `REPORT.md` – auto-written summary ready for submission.
- Spark logs in the console; check cached counts for sanity.

## Notes

- Script assumes the absolute dataset path above; update the constant only if the directory moves.
- Deadline extended to **October 30** – plan your final run to capture fresh metrics before submission.
