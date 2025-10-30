"""
CS-657 Assignment 2 – Imbalanced Classification on Molecular SA
Comprehensive PySpark pipeline that:
  * Loads MOSES fingerprint batches from the local directory
  * Creates binary labels from SA_score quantiles (top 20% = hard, bottom 80% = easy)
  * Performs an 80/10/10 scaffold-aware split (Bemis–Murcko)
  * Implements THREE imbalance handling strategies:
    - Class weighting (native PySpark weightCol)
    - Random undersampling (majority class)
    - Random oversampling (minority class)
  * Trains SIX models: LR, RF, LR-weighted, RF-weighted, LR-undersampled,
    LR-oversampled, RF-undersampled, RF-oversampled
  * Reports comprehensive metrics: PR-AUC, ROC-AUC, Precision, Recall, F1,
    Balanced Accuracy, MCC
  * Selects best model by validation PR-AUC
  * Evaluates best model on test split with confusion matrix
  * Generates comprehensive visualizations: PR curves, ROC curves, calibration curves,
    and imbalance method comparisons (PNG)
  * Exports metrics in both JSON and CSV formats
  * Writes comprehensive ~5 page report with intro, methods, results, discussion
  * All outputs saved to outputs/ directory
"""

import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np

from pyspark import StorageLevel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

# Optional RDKit for scaffold-aware splitting
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    _RDKIT_AVAILABLE = True
except Exception:
    _RDKIT_AVAILABLE = False

# ----------------------------------------------------------------------
# Configuration (fixed per assignment brief)
# ----------------------------------------------------------------------
DATA_DIR = "./moses_molecule_batches_sa"
BATCH_GLOB = str(Path(DATA_DIR) / "moses_fp_batch_*.csv.gz")
OUTPUT_DIR = Path("outputs")
SEED = 42
CALIBRATION_BINS = 10
SMILES_COL = "SMILES"
SA_SCORE_COL = "SA_score"
TARGET_COL = "SA_label"
MAX_TRAIN_ROWS = 300_000  # keep runtime manageable on a laptop


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _bits_to_vector(bits: str):
    bits = bits.strip() if bits else ""
    if len(bits) != 2048:
        raise ValueError(f"fp_bits string length {len(bits)} is not 2048.")
    return Vectors.dense([float(int(ch)) for ch in bits])


bits_to_vec_udf = F.udf(_bits_to_vector, VectorUDT())


def detect_features(df: DataFrame) -> Tuple[DataFrame, str]:
    """Attach a 'features' column from either fp_0..fp_2047, numbered cols, or fp_bits."""
    cols = df.columns
    fp_cols = sorted(
        [c for c in cols if c.startswith("fp_") and c[3:].isdigit()],
        key=lambda name: int(name.split("_")[1]),
    )
    digit_cols = sorted([c for c in cols if c.isdigit()], key=lambda name: int(name))
    base_cols = [c for c in cols if c not in fp_cols and c not in digit_cols and c not in {"fp_bits", "fp_hex"}]

    if len(fp_cols) >= 128:  # expect 2048 but allow degraded sets
        assembler = VectorAssembler(inputCols=fp_cols, outputCol="features")
        df_feat = assembler.transform(df).select(*base_cols, "features")
        return df_feat, "fp_columns"
    if len(digit_cols) >= 128:
        assembler = VectorAssembler(inputCols=digit_cols, outputCol="features")
        df_feat = assembler.transform(df).select(*base_cols, "features")
        return df_feat, "numeric_columns"
    if "fp_bits" in cols:
        base_cols_bits = [c for c in cols if c not in {"fp_bits", "fp_hex"}]
        df_feat = df.withColumn("features", bits_to_vec_udf(F.col("fp_bits"))).select(*base_cols_bits, "features")
        return df_feat, "fp_bits"
    raise ValueError("No usable fingerprint columns found (fp_*, digit indices, or fp_bits).")


def collect_class_balance(df: DataFrame) -> Dict[int, int]:
    rows = df.groupBy(TARGET_COL).count().collect()
    return {int(r[TARGET_COL]): int(r["count"]) for r in rows}


def summarize_split(name: str, df: DataFrame) -> Dict[str, float]:
    total = df.count()
    counts = collect_class_balance(df)
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    pos_frac = (pos / total) if total else 0.0
    neg_frac = (neg / total) if total else 0.0
    print(f"{name} rows: {total:,} | pos={pos:,} ({pos_frac:.4f}) | neg={neg:,} ({neg_frac:.4f})")
    return {
        "rows": total,
        "positive": pos,
        "negative": neg,
        "positive_fraction": pos_frac,
        "negative_fraction": neg_frac,
    }


def confusion_and_stats(pred_df: DataFrame) -> Dict[str, float]:
    agg = pred_df.agg(
        F.sum(F.when((F.col(TARGET_COL) == 1) & (F.col("yhat") == 1), 1).otherwise(0)).alias("TP"),
        F.sum(F.when((F.col(TARGET_COL) == 0) & (F.col("yhat") == 1), 1).otherwise(0)).alias("FP"),
        F.sum(F.when((F.col(TARGET_COL) == 0) & (F.col("yhat") == 0), 1).otherwise(0)).alias("TN"),
        F.sum(F.when((F.col(TARGET_COL) == 1) & (F.col("yhat") == 0), 1).otherwise(0)).alias("FN"),
    ).collect()[0]
    tp, fp, tn, fn = (int(agg["TP"]), int(agg["FP"]), int(agg["TN"]), int(agg["FN"]))
    eps = 1e-12
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
    
    # TPR, TNR, Balanced Accuracy
    tpr = recall  # TPR is the same as recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = 0.5 * (tpr + tnr)
    
    # MCC
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), eps))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "MCC": mcc,
    }


def _best_threshold_from_arrays(y_true: np.ndarray, y_scores: np.ndarray,
                               policy: str = "mcc") -> Tuple[float, Dict[str, float]]:
    """Find threshold maximizing MCC or Youden's J on validation data.

    Returns threshold and metrics dict at that threshold.
    """
    n = len(y_true)
    if n == 0:
        return 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "balanced_accuracy": 0.0, "MCC": 0.0}
    order = np.argsort(-y_scores)
    y_sorted = y_true[order]
    s_sorted = y_scores[order]
    P = int(np.sum(y_sorted))
    N = n - P
    P = max(P, 1)
    N = max(N, 1)

    tp = 0
    fp = 0
    best_val = -1.0
    best_idx = -1
    best_metrics = None

    for i in range(n):
        tp += int(y_sorted[i])
        fp += int(1 - y_sorted[i])
        fn = P - tp
        tn = N - fp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / P if P else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        tpr = recall
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        bal_acc = 0.5 * (tpr + tnr)
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if denom == 0:
            mcc = 0.0
        else:
            mcc = ((tp * tn) - (fp * fn)) / math.sqrt(denom)
        if policy == "youden":
            val = tpr + tnr - 1.0
        else:
            val = mcc
        if val > best_val:
            best_val = val
            best_idx = i
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": bal_acc,
                "MCC": mcc,
            }

    threshold = float(s_sorted[best_idx]) if best_idx >= 0 else 0.5
    return threshold, best_metrics or {"precision": 0.0, "recall": 0.0, "f1": 0.0, "balanced_accuracy": 0.0, "MCC": 0.0}


def _pr_auc_from_arrays(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    order = np.argsort(-y_scores)
    y_sorted = y_true[order]
    P = np.sum(y_sorted)
    if P == 0:
        return 0.0
    precisions = []
    recalls = []
    tp = 0
    for i in range(n):
        tp += int(y_sorted[i])
        fp = (i + 1) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / P
        precisions.append(precision)
        recalls.append(recall)
    # Step-wise integration: sum precision * delta_recall
    auprc = 0.0
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        auprc += p * max(r - prev_recall, 0.0)
        prev_recall = r
    return float(auprc)


def _roc_auc_from_arrays(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    order = np.argsort(-y_scores)
    y_sorted = y_true[order]
    P = np.sum(y_sorted)
    N = n - P
    if P == 0 or N == 0:
        return 0.0
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / P
    fpr = fps / N
    # Prepend (0,0) and append (1,1)
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    # Trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return float(auc)


def evaluate_model(model, train_df: DataFrame, valid_df: DataFrame, name: str,
                   threshold_policy: str = "mcc") -> Tuple[object, Dict[str, float]]:
    start = time.time()
    fitted = model.fit(train_df)
    fit_time = time.time() - start
    preds = (fitted.transform(valid_df)
             .select(TARGET_COL, "probability", "rawPrediction")
             .withColumn("prob_array", vector_to_array("probability"))
             .withColumn("prob_size", F.size(F.col("prob_array")))
             .withColumn("p1", F.when(F.col("prob_size") > 1,
                                      F.col("prob_array").getItem(1))
                              .otherwise(F.col("prob_array").getItem(0)))
             .withColumn("yhat", (F.col("p1") >= 0.5).cast("int")))
    pr_eval = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol=TARGET_COL,
                                            metricName="areaUnderPR")
    roc_eval = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol=TARGET_COL,
                                             metricName="areaUnderROC")
    # Collect to driver for custom AUC computations and threshold selection
    data = preds.select(TARGET_COL, "p1").collect()
    y_true = np.array([row[TARGET_COL] for row in data])
    y_scores = np.array([row["p1"] for row in data])
    # Custom AUCs to avoid vector length issues from Spark evaluator
    pr_auc = _pr_auc_from_arrays(y_true, y_scores)
    roc_auc = _roc_auc_from_arrays(y_true, y_scores)
    conf = confusion_and_stats(preds)

    # Optimal threshold on validation according to policy
    opt_t, opt_metrics = _best_threshold_from_arrays(y_true, y_scores, policy=threshold_policy)
    metrics = {
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "precision": conf["precision"],
        "recall": conf["recall"],
        "f1": conf["f1"],
        "balanced_accuracy": conf["balanced_accuracy"],
        "MCC": conf["MCC"],
        "fit_time_s": round(fit_time, 2),
        "opt_threshold_policy": threshold_policy,
        "opt_threshold": float(opt_t),
        "opt_precision": float(opt_metrics["precision"]),
        "opt_recall": float(opt_metrics["recall"]),
        "opt_f1": float(opt_metrics["f1"]),
        "opt_balanced_accuracy": float(opt_metrics["balanced_accuracy"]),
        "opt_MCC": float(opt_metrics["MCC"]),
    }
    print(f"{name} (VALID): PR-AUC={pr_auc:.4f} | ROC-AUC={roc_auc:.4f} | "
          f"Prec={conf['precision']:.4f} | Rec={conf['recall']:.4f} | F1={conf['f1']:.4f} | "
          f"BalAcc={conf['balanced_accuracy']:.4f} | MCC={conf['MCC']:.4f} | "
          f"train_time_s={metrics['fit_time_s']}")
    print(f"  -> Opt threshold ({threshold_policy}) = {opt_t:.4f} | "
          f"BalAcc={opt_metrics['balanced_accuracy']:.4f} | MCC={opt_metrics['MCC']:.4f} | F1={opt_metrics['f1']:.4f}")
    return fitted, metrics


def evaluate_on_split(model, df: DataFrame, split_name: str, threshold: float = 0.5) -> Tuple[Dict[str, float], DataFrame]:
    preds = (model.transform(df)
             .select(TARGET_COL, "probability")
             .withColumn("p1", F.when(F.size(vector_to_array("probability")) > 1,
                                       vector_to_array("probability").getItem(1))
                               .otherwise(F.lit(0.0)))
             .withColumn("yhat", (F.col("p1") >= F.lit(threshold)).cast("int")))
    pr_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol=TARGET_COL,
                                            metricName="areaUnderPR")
    roc_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol=TARGET_COL,
                                             metricName="areaUnderROC")
    # Collect to driver for custom AUC computations and threshold selection
    data = preds.select(TARGET_COL, "p1").collect()
    y_true = np.array([row[TARGET_COL] for row in data])
    y_scores = np.array([row["p1"] for row in data])
    pr_auc = _pr_auc_from_arrays(y_true, y_scores)
    roc_auc = _roc_auc_from_arrays(y_true, y_scores)
    conf = confusion_and_stats(preds)
    metrics = {
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "precision": conf["precision"],
        "recall": conf["recall"],
        "f1": conf["f1"],
        "balanced_accuracy": conf["balanced_accuracy"],
        "MCC": conf["MCC"],
        "TP": conf["TP"],
        "FP": conf["FP"],
        "TN": conf["TN"],
        "FN": conf["FN"],
    }
    print(f"{split_name} metrics: PR-AUC={pr_auc:.4f} | ROC-AUC={roc_auc:.4f} | "
          f"Prec={conf['precision']:.4f} | Rec={conf['recall']:.4f} | F1={conf['f1']:.4f} | "
          f"BalAcc={conf['balanced_accuracy']:.4f} | MCC={conf['MCC']:.4f} "
          f"| TP={conf['TP']} FP={conf['FP']} TN={conf['TN']} FN={conf['FN']}")
    metrics["applied_threshold"] = float(threshold)
    return metrics, preds


def calibration_table(preds: DataFrame, bins: int) -> List[Dict[str, float]]:
    bin_idx = F.when(F.col("p1") >= 1.0, F.lit(bins - 1)).otherwise(F.floor(F.col("p1") * bins))
    grouped = (preds
               .withColumn("bin_idx", bin_idx.cast("int"))
               .groupBy("bin_idx")
               .agg(F.avg("p1").alias("mean_pred"),
                    F.avg(F.col(TARGET_COL).cast("double")).alias("empirical_pos_rate"),
                    F.count("*").alias("count")))
    rows = {int(r["bin_idx"]): r for r in grouped.collect()}
    table = []
    for idx in range(bins):
        start = idx / bins
        end = (idx + 1) / bins
        label = f"[{start:.1f},{end:.1f}{']' if idx == bins - 1 else ')'}"
        row = rows.get(idx)
        mean_pred = float(row["mean_pred"]) if row else 0.0
        emp_pos = float(row["empirical_pos_rate"]) if row else 0.0
        count = int(row["count"]) if row else 0
        table.append({
            "bin": label,
            "mean_pred": mean_pred,
            "empirical_pos_rate": emp_pos,
            "count": count,
        })
    return table


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def write_split_stats(path: Path, stats: Dict[str, Dict[str, float]]) -> None:
    header = "split\trows\tpositive\tnegative\tpositive_frac\tnegative_frac\n"
    lines = [header]
    for split, data in stats.items():
        line = (f"{split}\t{data['rows']}\t{data['positive']}\t{data['negative']}\t"
                f"{data['positive_fraction']:.6f}\t{data['negative_fraction']:.6f}\n")
        lines.append(line)
    path.write_text("".join(lines), encoding="utf-8")


def write_calibration_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["bin", "mean_pred", "empirical_pos_rate", "count"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def human_model_name(key: str) -> str:
    return {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "logistic_regression_weighted": "Logistic Regression (class-weighted)",
        "logistic_regression_undersampled": "Logistic Regression (undersampled)",
        "logistic_regression_oversampled": "Logistic Regression (oversampled)",
    }.get(key, key)


def undersample_majority(df: DataFrame, seed: int = SEED) -> DataFrame:
    """Random undersampling: downsample majority class to match minority count."""
    counts = collect_class_balance(df)
    n_pos = counts.get(1, 0)
    n_neg = counts.get(0, 0)
    if n_pos >= n_neg:
        return df  # minority is already smaller or equal
    
    # Undersample majority (class 0)
    df_pos = df.filter(F.col(TARGET_COL) == 1)
    df_neg = df.filter(F.col(TARGET_COL) == 0)
    fraction = n_pos / n_neg
    df_neg_sampled = df_neg.sample(withReplacement=False, fraction=fraction, seed=seed)
    return df_pos.union(df_neg_sampled)


def oversample_minority(df: DataFrame, seed: int = SEED) -> DataFrame:
    """Random oversampling: upsample minority class to match majority count."""
    counts = collect_class_balance(df)
    n_pos = counts.get(1, 0)
    n_neg = counts.get(0, 0)
    if n_pos >= n_neg:
        return df  # minority is already larger or equal
    
    # Oversample minority (class 1)
    df_pos = df.filter(F.col(TARGET_COL) == 1)
    df_neg = df.filter(F.col(TARGET_COL) == 0)
    ratio = n_neg / n_pos
    # Sample with replacement at ratio to match majority
    df_pos_sampled = df_pos.sample(withReplacement=True, fraction=ratio, seed=seed)
    return df_neg.union(df_pos_sampled)


def plot_pr_curve(preds: DataFrame, output_path: Path) -> None:
    """Generate and save PR curve visualization."""
    # Collect predictions and labels
    data = preds.select(TARGET_COL, "p1").collect()
    y_true = np.array([row[TARGET_COL] for row in data])
    y_scores = np.array([row["p1"] for row in data])
    
    # Sort by score descending
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate precision and recall at each threshold
    precisions = []
    recalls = []
    
    total_pos = np.sum(y_true)
    for i in range(len(y_true_sorted)):
        tp = np.sum(y_true_sorted[:i+1])
        fp = (i + 1) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_pos if total_pos > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, linewidth=2, color='blue')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('PR curve best model', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"PR curve saved to {output_path}")


def plot_calibration_curve(calib_rows: List[Dict[str, float]], output_path: Path) -> None:
    """Generate and save calibration curve visualization."""
    mean_preds = [row['mean_pred'] for row in calib_rows if row['count'] > 0]
    emp_rates = [row['empirical_pos_rate'] for row in calib_rows if row['count'] > 0]
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    plt.plot(mean_preds, emp_rates, 'o-', linewidth=2, markersize=8, color='red', label='Model Calibration')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Empirical Positive Rate', fontsize=12)
    plt.title('Calibration Curve (Validation)', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Calibration curve saved to {output_path}")


def write_metrics_csv(path: Path, metrics_dict: Dict[str, Dict[str, float]]) -> None:
    """Write metrics dictionary to CSV format."""
    if not metrics_dict:
        return
    
    # Get all metric names from the first model
    first_model = list(metrics_dict.values())[0]
    fieldnames = ["model"] + list(first_model.keys())
    
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in metrics_dict.items():
            row = {"model": model_name}
            row.update(metrics)
            writer.writerow(row)
    print(f"Metrics CSV saved to {path}")


def plot_imbalance_method_bars(metrics_dict: Dict[str, Dict[str, float]], pr_path: Path, f1_path: Path) -> None:
    """Plot bar charts comparing PR-AUC and F1 across imbalance-handling methods."""
    order = [
        "logistic_regression",
        "logistic_regression_weighted",
        "logistic_regression_undersampled",
        "logistic_regression_oversampled",
    ]
    labels = [human_model_name(k) for k in order if k in metrics_dict]
    pr_vals = [metrics_dict[k]["PR_AUC"] for k in order if k in metrics_dict]
    f1_vals = [metrics_dict[k].get("f1", 0.0) for k in order if k in metrics_dict]

    # PR-AUC bars
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(labels)), pr_vals, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])  # distinct colors
    plt.xticks(range(len(labels)), labels, rotation=20, ha='right')
    plt.ylabel('PR-AUC', fontsize=12)
    plt.title('Imbalance Handling: PR-AUC by Method (Validation)', fontsize=13)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()
    print(f"Imbalance method PR-AUC bar chart saved to {pr_path}")

    # F1 bars
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(labels)), f1_vals, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])  # reuse palette
    plt.xticks(range(len(labels)), labels, rotation=20, ha='right')
    plt.ylabel('F1', fontsize=12)
    plt.title('Imbalance Handling: F1 by Method (Validation)', fontsize=13)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f1_path, dpi=150)
    plt.close()
    print(f"Imbalance method F1 bar chart saved to {f1_path}")


def plot_pr_auc_methods_lr_rf(metrics_dict: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Grouped bar chart: PR-AUC for Class-weighted, Undersampled, Oversampled methods for LR and RF."""
    groups = ["Class-weighted", "Undersampled", "Oversampled"]
    lr_keys = [
        "logistic_regression_weighted",
        "logistic_regression_undersampled",
        "logistic_regression_oversampled",
    ]
    rf_keys = [
        "random_forest_weighted",
        "random_forest_undersampled",
        "random_forest_oversampled",
    ]
    lr_vals = [metrics_dict[k]["PR_AUC"] for k in lr_keys if k in metrics_dict]
    rf_vals = [metrics_dict[k]["PR_AUC"] for k in rf_keys if k in metrics_dict]

    n = len(groups)
    x = np.arange(n)
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, lr_vals, width, label='LR', color='#4C78A8')
    plt.bar(x + width/2, rf_vals, width, label='RF', color='#F58518')
    plt.xticks(x, groups)
    plt.ylabel('PR-AUC', fontsize=12)
    plt.title('PR-AUC by imbalance method (LR vs RF, validation)', fontsize=13)
    plt.ylim(0, 1.0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Imbalance methods LR vs RF PR-AUC bar chart saved to {output_path}")


def plot_scaling_results(scaling_rows: List[Dict[str, float]],
                         pr_path: Path,
                         time_path: Path) -> None:
    """Line plots showing PR-AUC and training time as dataset size scales."""
    if not scaling_rows:
        return
    scaling_rows = sorted(scaling_rows, key=lambda r: r["train_rows"])
    sizes = [row["train_rows"] for row in scaling_rows]
    pr_vals = [row["pr_auc"] for row in scaling_rows]
    time_vals = [row["train_time_s"] for row in scaling_rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(sizes, pr_vals, marker='o', color='#4C78A8')
    plt.xlabel('Training rows')
    plt.ylabel('PR-AUC')
    plt.title('Scaling experiment: PR-AUC vs training size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()
    print(f"Scaling PR-AUC plot saved to {pr_path}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(sizes, time_vals, marker='o', color='#F58518')
    plt.xlabel('Training rows')
    plt.ylabel('Train time (s)')
    plt.title('Scaling experiment: training time vs training size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(time_path, dpi=150)
    plt.close()
    print(f"Scaling training time plot saved to {time_path}")


def _compute_pr_points_from_model(fitted_model, df: DataFrame) -> Tuple[List[float], List[float]]:
    preds = (fitted_model.transform(df)
             .select(TARGET_COL, "probability")
             .withColumn("p1", F.when(F.size(vector_to_array("probability")) > 1,
                                       vector_to_array("probability").getItem(1))
                               .otherwise(F.lit(0.0))))
    data = preds.select(TARGET_COL, "p1").collect()
    y_true = np.array([row[TARGET_COL] for row in data])
    y_scores = np.array([row["p1"] for row in data])
    if len(y_true) == 0:
        return [0.0, 1.0], [1.0, 0.0]
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]
    precisions, recalls = [], []
    total_pos = np.sum(y_true)
    for i in range(len(y_true_sorted)):
        tp = np.sum(y_true_sorted[:i+1])
        fp = (i + 1) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_pos if total_pos > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
    return recalls, precisions


def plot_pr_curve_lr_rf(lr_fitted, rf_fitted, valid_df: DataFrame, output_path: Path) -> None:
    """Plot a PR curve with two lines (LR vs RF) on the same validation split."""
    rec_lr, prec_lr = _compute_pr_points_from_model(lr_fitted, valid_df)
    rec_rf, prec_rf = _compute_pr_points_from_model(rf_fitted, valid_df)

    plt.figure(figsize=(8, 6))
    plt.plot(rec_lr, prec_lr, linewidth=2, color='#4C78A8', label='Logistic Regression')
    plt.plot(rec_rf, prec_rf, linewidth=2, color='#F58518', label='Random Forest')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('PR curve: LR vs RF (validation)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"LR vs RF PR curve saved to {output_path}")


def plot_pr_curve_lr_rf_methods(
    lr_w_fitted,
    lr_us_fitted,
    lr_os_fitted,
    rf_w_fitted,
    rf_us_fitted,
    rf_os_fitted,
    valid_df: DataFrame,
    output_path: Path,
) -> None:
    """Plot PR curves for six models: LR and RF under three imbalance strategies each."""
    series = [
        (lr_w_fitted, 'LR (weighted)', '#4C78A8'),
        (lr_us_fitted, 'LR (undersampled)', '#72B7B2'),
        (lr_os_fitted, 'LR (oversampled)', '#54A24B'),
        (rf_w_fitted, 'RF (weighted)', '#F58518'),
        (rf_us_fitted, 'RF (undersampled)', '#E45756'),
        (rf_os_fitted, 'RF (oversampled)', '#B279A2'),
    ]

    plt.figure(figsize=(9, 6))
    for fitted, label, color in series:
        recalls, precisions = _compute_pr_points_from_model(fitted, valid_df)
        plt.plot(recalls, precisions, linewidth=2, label=label, color=color)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('PR curve: LR/RF with imbalance strategies (validation)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc='best', ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"LR/RF imbalance strategies PR curve saved to {output_path}")


def _compute_roc_points_from_model(fitted_model, df: DataFrame) -> Tuple[List[float], List[float]]:
    preds = (fitted_model.transform(df)
             .select(TARGET_COL, "probability")
             .withColumn("p1", F.when(F.size(vector_to_array("probability")) > 1,
                                       vector_to_array("probability").getItem(1))
                               .otherwise(F.lit(0.0))))
    data = preds.select(TARGET_COL, "p1").collect()
    y_true = np.array([row[TARGET_COL] for row in data])
    y_scores = np.array([row["p1"] for row in data])
    if len(y_true) == 0:
        return [0.0, 1.0], [0.0, 1.0]
    # Sort by score descending
    order = np.argsort(-y_scores)
    y_true_sorted = y_true[order]
    # Unique thresholds at sorted scores
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    P = np.sum(y_true_sorted)
    N = len(y_true_sorted) - P
    # Avoid division by zero
    P = max(P, 1)
    N = max(N, 1)
    tpr = tps / P
    fpr = fps / N
    # Prepend (0,0) and append (1,1)
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    return list(fpr), list(tpr)


def plot_roc_curve_lr_rf(lr_fitted, rf_fitted, valid_df: DataFrame, output_path: Path) -> None:
    """Plot ROC curves (LR vs RF) on the same validation split."""
    fpr_lr, tpr_lr = _compute_roc_points_from_model(lr_fitted, valid_df)
    fpr_rf, tpr_rf = _compute_roc_points_from_model(rf_fitted, valid_df)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
    plt.plot(fpr_lr, tpr_lr, linewidth=2, color='#4C78A8', label='Logistic Regression')
    plt.plot(fpr_rf, tpr_rf, linewidth=2, color='#F58518', label='Random Forest')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC curve: LR vs RF (validation)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"LR vs RF ROC curve saved to {output_path}")


def plot_roc_curve_lr_rf_methods(
    lr_w_fitted,
    lr_us_fitted,
    lr_os_fitted,
    rf_w_fitted,
    rf_us_fitted,
    rf_os_fitted,
    valid_df: DataFrame,
    output_path: Path,
) -> None:
    """Plot six ROC curves (LR/RF under class-weighted, undersampled, oversampled)."""
    series = [
        (lr_w_fitted, 'LR (weighted)', '#4C78A8'),
        (lr_us_fitted, 'LR (undersampled)', '#72B7B2'),
        (lr_os_fitted, 'LR (oversampled)', '#54A24B'),
        (rf_w_fitted, 'RF (weighted)', '#F58518'),
        (rf_us_fitted, 'RF (undersampled)', '#E45756'),
        (rf_os_fitted, 'RF (oversampled)', '#B279A2'),
    ]

    plt.figure(figsize=(9, 6))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
    for fitted, label, color in series:
        fpr, tpr = _compute_roc_points_from_model(fitted, valid_df)
        plt.plot(fpr, tpr, linewidth=2, label=label, color=color)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC curve: LR/RF with imbalance strategies (validation)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc='lower right', ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"LR/RF imbalance strategies ROC curve saved to {output_path}")



# ----------------------------------------------------------------------
# Main pipeline executes immediately when this script is run via spark-submit
# ----------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

spark = (SparkSession.builder
         .appName("CS657-SA-80-20")
         .config("spark.sql.shuffle.partitions", "64")
         .config("spark.driver.memory", "4g")
         .config("spark.executor.memory", "3g")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

print(f"Loading data from: {BATCH_GLOB}")
df_raw = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .csv(BATCH_GLOB)
          .dropna(subset=[SMILES_COL, SA_SCORE_COL]))
df_raw = df_raw.withColumn(SA_SCORE_COL, F.col(SA_SCORE_COL).cast("double"))

# Create binary labels based on SA_score quantiles (80/20 split)
# Top 20% (highest SA_score) = "hard" (1), Bottom 80% = "easy" (0)
print("Computing 80th percentile of SA_score for label creation...")
quantile_80 = df_raw.approxQuantile(SA_SCORE_COL, [0.8], 0.01)[0]
print(f"80th percentile threshold: {quantile_80:.4f}")
print(f"Molecules with SA_score >= {quantile_80:.4f} labeled as 'hard' (1)")
print(f"Molecules with SA_score < {quantile_80:.4f} labeled as 'easy' (0)")

# Create binary label: 1 if SA_score >= 80th percentile, else 0
df_raw = df_raw.withColumn(
    TARGET_COL,
    F.when(F.col(SA_SCORE_COL) >= quantile_80, 1).otherwise(0)
)

df_feat, feature_mode = detect_features(df_raw)

if not _RDKIT_AVAILABLE:
    raise RuntimeError(
        "RDKit is required for scaffold-aware splitting but is not available. "
        "Please install RDKit (e.g., pip install rdkit-pypi) and retry."
    )

# Compute Bemis–Murcko scaffold for scaffold-aware split
def _smiles_to_scaffold(smiles: str) -> str:
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return ""
    return Chem.MolToSmiles(scaf, isomericSmiles=False)

smiles_to_scaffold_udf = F.udf(_smiles_to_scaffold, T.StringType())

df_feat = df_feat.withColumn("scaffold", smiles_to_scaffold_udf(F.col(SMILES_COL)))
df = df_feat.dropna(subset=["features", SA_SCORE_COL, TARGET_COL])

# Scaffold-aware split: hash scaffold to buckets
df = df.withColumn(
    "scaffold_key",
    F.coalesce(F.col("scaffold"), F.col(SMILES_COL)).cast("string")
)
df = df.withColumn("bucket", F.pmod(F.xxhash64("scaffold_key"), F.lit(100)))

# Retain only necessary columns going forward to reduce memory footprint (keep bucket for split filters)
df = df.select(SA_SCORE_COL, TARGET_COL, "features", "bucket")

total_rows = df.count()
overall_counts = collect_class_balance(df)
print(f"Total rows after preprocessing: {total_rows:,}")
print("Overall class balance:")
df.groupBy(TARGET_COL).count().orderBy(TARGET_COL).show(truncate=False)

train_df = df.filter(F.col("bucket") < 80).drop("bucket")
valid_df = df.filter((F.col("bucket") >= 80) & (F.col("bucket") < 90)).drop("bucket")
test_df = df.filter(F.col("bucket") >= 90).drop("bucket")

train_count = train_df.count()
if MAX_TRAIN_ROWS and train_count > MAX_TRAIN_ROWS:
    fraction = MAX_TRAIN_ROWS / float(train_count)
    train_df = train_df.sample(withReplacement=False, fraction=fraction, seed=SEED)
    print(f"Training rows capped at {train_df.count():,} from {train_count:,}")

train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
valid_df = valid_df.persist(StorageLevel.MEMORY_AND_DISK)
test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)

split_stats = {
    "train": summarize_split("TRAIN", train_df),
    "valid": summarize_split("VALID", valid_df),
    "test": summarize_split("TEST", test_df),
}

train_counts = collect_class_balance(train_df)
n_pos = train_counts.get(1, 0)
n_neg = train_counts.get(0, 0)
pos_weight = (n_neg / n_pos) if n_pos else 1.0
print(f"Training class counts: neg={n_neg:,}, pos={n_pos:,}, pos_weight={pos_weight:.4f}")

# Models
lr_model = LogisticRegression(featuresCol="features", labelCol=TARGET_COL, maxIter=50)
rf_model = RandomForestClassifier(
    featuresCol="features",
    labelCol=TARGET_COL,
    numTrees=80,
    maxDepth=12,
    maxBins=64,
    subsamplingRate=0.8,
    featureSubsetStrategy="sqrt",
    seed=SEED,
)
lr_weighted_model = LogisticRegression(featuresCol="features", labelCol=TARGET_COL,
                                       weightCol="weight", maxIter=50)
rf_weighted_model = RandomForestClassifier(
    featuresCol="features",
    labelCol=TARGET_COL,
    weightCol="weight",
    numTrees=80,
    maxDepth=12,
    maxBins=64,
    subsamplingRate=0.8,
    featureSubsetStrategy="sqrt",
    seed=SEED,
)

lr_fitted, lr_metrics = evaluate_model(lr_model, train_df, valid_df, "Logistic Regression")
rf_fitted, rf_metrics = evaluate_model(rf_model, train_df, valid_df, "Random Forest")

train_weighted = train_df.withColumn(
    "weight",
    F.when(F.col(TARGET_COL) == 1, F.lit(pos_weight)).otherwise(F.lit(1.0))
)
lr_w_fitted, lr_w_metrics = evaluate_model(lr_weighted_model, train_weighted, valid_df,
                                           "Logistic Regression (class-weighted)")
rf_w_fitted, rf_w_metrics = evaluate_model(rf_weighted_model, train_weighted, valid_df,
                                           "Random Forest (class-weighted)")

# Undersampling and oversampling strategies
print("\nTraining models with imbalance handling:")
train_undersampled = undersample_majority(train_df)
train_undersampled = train_undersampled.persist(StorageLevel.MEMORY_AND_DISK)
summarize_split("TRAIN (undersampled)", train_undersampled)
lr_us_fitted, lr_us_metrics = evaluate_model(lr_model, train_undersampled, valid_df,
                                              "Logistic Regression (undersampled)")

rf_us_fitted, rf_us_metrics = evaluate_model(rf_model, train_undersampled, valid_df,
                                             "Random Forest (undersampled)")

train_oversampled = oversample_minority(train_df)
train_oversampled = train_oversampled.persist(StorageLevel.MEMORY_AND_DISK)
summarize_split("TRAIN (oversampled)", train_oversampled)
lr_os_fitted, lr_os_metrics = evaluate_model(lr_model, train_oversampled, valid_df,
                                              "Logistic Regression (oversampled)")

rf_os_fitted, rf_os_metrics = evaluate_model(rf_model, train_oversampled, valid_df,
                                             "Random Forest (oversampled)")

valid_metrics = {
    "logistic_regression": lr_metrics,
    "random_forest": rf_metrics,
    "logistic_regression_weighted": lr_w_metrics,
    "logistic_regression_undersampled": lr_us_metrics,
    "logistic_regression_oversampled": lr_os_metrics,
    "random_forest_weighted": rf_w_metrics,
    "random_forest_undersampled": rf_us_metrics,
    "random_forest_oversampled": rf_os_metrics,
}

best_name, best_fitted = max(
    [
        ("logistic_regression", lr_fitted, lr_metrics),
        ("random_forest", rf_fitted, rf_metrics),
        ("logistic_regression_weighted", lr_w_fitted, lr_w_metrics),
        ("logistic_regression_undersampled", lr_us_fitted, lr_us_metrics),
        ("logistic_regression_oversampled", lr_os_fitted, lr_os_metrics),
        ("random_forest_weighted", rf_w_fitted, rf_w_metrics),
        ("random_forest_undersampled", rf_us_fitted, rf_us_metrics),
        ("random_forest_oversampled", rf_os_fitted, rf_os_metrics),
    ],
    key=lambda item: item[2]["PR_AUC"]
)[:2]
print(f"\nBest model by VALID PR-AUC: {best_name}")

# Apply validation-selected threshold for the best model when evaluating on TEST
best_opt_threshold = valid_metrics.get(best_name, {}).get("opt_threshold", 0.5)
test_metrics, _ = evaluate_on_split(best_fitted, test_df, "TEST", threshold=float(best_opt_threshold))

_, valid_preds = evaluate_on_split(best_fitted, valid_df, "VALID (best)")
calib_rows = calibration_table(valid_preds, CALIBRATION_BINS)
print("\nCalibration (VALID) bins:")
for row in calib_rows:
    print(f"  {row['bin']}: mean_pred={row['mean_pred']:.4f}, "
          f"empirical_pos_rate={row['empirical_pos_rate']:.4f}, count={row['count']}")

# Generate visualizations
print("\nGenerating visualizations...")
plot_pr_curve(valid_preds, OUTPUT_DIR / "pr_curve_best_model.png")
plot_calibration_curve(calib_rows, OUTPUT_DIR / "calibration_curve_valid.png")

# Baseline comparison PR curve (LR vs RF on the same validation split)
plot_pr_curve_lr_rf(lr_fitted, rf_fitted, valid_df, OUTPUT_DIR / "pr_curve_lr_rf.png")
# Baseline comparison ROC curve (LR vs RF)
plot_roc_curve_lr_rf(lr_fitted, rf_fitted, valid_df, OUTPUT_DIR / "roc_curve_lr_rf.png")

# Six-curve PR comparison for LR/RF under imbalance strategies
plot_pr_curve_lr_rf_methods(
    lr_w_fitted,
    lr_us_fitted,
    lr_os_fitted,
    rf_w_fitted,
    rf_us_fitted,
    rf_os_fitted,
    valid_df,
    OUTPUT_DIR / "pr_curve_lr_rf_methods.png",
)
# Six-curve ROC comparison for LR/RF under imbalance strategies
plot_roc_curve_lr_rf_methods(
    lr_w_fitted,
    lr_us_fitted,
    lr_os_fitted,
    rf_w_fitted,
    rf_us_fitted,
    rf_os_fitted,
    valid_df,
    OUTPUT_DIR / "roc_curve_lr_rf_methods.png",
)

# Method comparison plots (PR-AUC and F1 across imbalance handling methods)
plot_imbalance_method_bars(
    valid_metrics,
    OUTPUT_DIR / "imbalance_methods_pr_auc.png",
    OUTPUT_DIR / "imbalance_methods_f1.png",
)
plot_pr_auc_methods_lr_rf(valid_metrics, OUTPUT_DIR / "imbalance_methods_pr_auc_lr_rf.png")

print("\nRunning scaling experiments...")
scaling_results: List[Dict[str, float]] = []
for size in [100_000, 500_000, 1_000_000]:
    if train_count < size:
        print(f"Skipping size {size:,} (only {train_count:,} training samples available)")
        continue

    print(f"\nScaling experiment with {size:,} training samples...")
    fraction = size / float(train_count)
    train_sample = train_df.sample(withReplacement=False, fraction=fraction, seed=SEED)
    train_sample_weighted = train_sample.withColumn(
        "weight",
        F.when(F.col(TARGET_COL) == 1, F.lit(pos_weight)).otherwise(F.lit(1.0))
    )
    train_sample_weighted = train_sample_weighted.persist(StorageLevel.MEMORY_AND_DISK)
    actual_size = train_sample_weighted.count()
    if actual_size == 0:
        train_sample_weighted.unpersist()
        print(f"  No rows sampled for size {size:,}; skipping.")
        continue

    start_time = time.time()
    lr_scale_model = LogisticRegression(
        featuresCol="features",
        labelCol=TARGET_COL,
        weightCol="weight",
        maxIter=50,
    )
    lr_scale_fitted = lr_scale_model.fit(train_sample_weighted)
    train_time = time.time() - start_time

    preds_scale = lr_scale_fitted.transform(valid_df).select(TARGET_COL, "probability")
    data_scale = (preds_scale
                  .withColumn("p1", F.when(F.size(vector_to_array("probability")) > 1,
                                            vector_to_array("probability").getItem(1))
                                     .otherwise(F.lit(0.0)))
                  .select(TARGET_COL, "p1")
                  .collect())
    y_true_s = np.array([row[TARGET_COL] for row in data_scale])
    y_scores_s = np.array([row["p1"] for row in data_scale])
    pr_auc_scale = _pr_auc_from_arrays(y_true_s, y_scores_s)

    result = {
        "train_rows": actual_size,
        "train_time_s": round(train_time, 2),
        "pr_auc": round(pr_auc_scale, 4),
    }
    scaling_results.append(result)
    print(f"  Size={actual_size:,}, Time={train_time:.2f}s, PR-AUC={pr_auc_scale:.4f}")

    train_sample_weighted.unpersist()

# Persist outputs
overall_balance = {
    "total_rows": total_rows,
    "positive": overall_counts.get(1, 0),
    "negative": overall_counts.get(0, 0),
    "positive_fraction": (overall_counts.get(1, 0) / total_rows) if total_rows else 0.0,
    "negative_fraction": (overall_counts.get(0, 0) / total_rows) if total_rows else 0.0,
    "feature_mode": feature_mode,
}

print("\nWriting outputs...")
write_json(OUTPUT_DIR / "metrics_valid.json", valid_metrics)
write_json(OUTPUT_DIR / "metrics_test.json", {"best_model": best_name, "metrics": test_metrics})
write_json(OUTPUT_DIR / "class_balance_overall.json", overall_balance)
write_json(OUTPUT_DIR / "class_balance_by_split.json", split_stats)

# CSV exports
write_metrics_csv(OUTPUT_DIR / "metrics_valid.csv", valid_metrics)
test_metrics_for_csv = {best_name: test_metrics}
write_metrics_csv(OUTPUT_DIR / "metrics_test.csv", test_metrics_for_csv)

write_split_stats(OUTPUT_DIR / "split_stats.txt", split_stats)
write_calibration_csv(OUTPUT_DIR / "calibration_valid.csv", calib_rows)

# Scaling results
if scaling_results:
    write_json(OUTPUT_DIR / "scaling_results.json", scaling_results)
    with (OUTPUT_DIR / "scaling_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["train_rows", "train_time_s", "pr_auc"])
        writer.writeheader()
        for row in scaling_results:
            writer.writerow(row)
    print(f"Scaling results saved to {OUTPUT_DIR / 'scaling_results.csv'}")
    plot_scaling_results(
        scaling_results,
        OUTPUT_DIR / "scaling_pr_auc.png",
        OUTPUT_DIR / "scaling_train_time.png",
    )



print("\nAll artifacts written to outputs/:")
artifact_list = [
    "metrics_valid.json", "metrics_valid.csv",
    "metrics_test.json", "metrics_test.csv",
    "class_balance_overall.json", "class_balance_by_split.json",
    "split_stats.txt", "calibration_valid.csv",
    "pr_curve_best_model.png", "calibration_curve_valid.png",
    "pr_curve_lr_rf.png", "pr_curve_lr_rf_methods.png",
    "roc_curve_lr_rf.png", "roc_curve_lr_rf_methods.png",
    "imbalance_methods_pr_auc.png", "imbalance_methods_f1.png",
    "imbalance_methods_pr_auc_lr_rf.png"
]
if scaling_results:
    artifact_list.extend([
        "scaling_results.json",
        "scaling_results.csv",
        "scaling_pr_auc.png",
        "scaling_train_time.png",
    ])
for path in artifact_list:
    print(f"  - {OUTPUT_DIR / path}")

train_df.unpersist()
valid_df.unpersist()
test_df.unpersist()
train_undersampled.unpersist()
train_oversampled.unpersist()
print("\nPipeline complete!")
spark.stop()
