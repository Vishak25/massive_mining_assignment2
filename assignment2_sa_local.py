"""
CS-657 Assignment 2 – Imbalanced Classification on Molecular SA
Comprehensive PySpark pipeline that:
  * Loads MOSES fingerprint batches from the local directory
  * Relabels using the 80th percentile SA_score threshold (target80)
  * Performs an 80/10/10 random split
  * Implements THREE imbalance handling strategies:
    - Class weighting (native PySpark weightCol)
    - Random undersampling (majority class)
    - Random oversampling (minority class)
  * Trains FIVE models: LR, RF, LR-weighted, LR-undersampled, LR-oversampled
  * Reports comprehensive metrics: PR-AUC, ROC-AUC, Precision, Recall, F1, 
    Balanced Accuracy, MCC
  * Selects best model by validation PR-AUC
  * Evaluates best model on test split with confusion matrix
  * Generates visualizations: PR curve and calibration curve (PNG)
  * Performs scaling experiments (100k, 500k, 1M rows)
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
from pyspark.sql import DataFrame, SparkSession, functions as F

# ----------------------------------------------------------------------
# Configuration (fixed per assignment brief)
# ----------------------------------------------------------------------
DATA_DIR = "/Users/vishaknandakumar/Documents/Masters/College/Fall25/CS-657/Assignment2/moses_molecule_batches_sa"
BATCH_GLOB = str(Path(DATA_DIR) / "moses_fp_batch_*.csv.gz")
OUTPUT_DIR = Path("outputs")
SEED = 42
CALIBRATION_BINS = 10
SMILES_COL = "SMILES"
SA_SCORE_COL = "SA_score"
TARGET_COL = "target80"
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


def evaluate_model(model, train_df: DataFrame, valid_df: DataFrame, name: str) -> Tuple[object, Dict[str, float]]:
    start = time.time()
    fitted = model.fit(train_df)
    fit_time = time.time() - start
    preds = (fitted.transform(valid_df)
             .select(TARGET_COL, "probability", "rawPrediction")
             .withColumn("p1", vector_to_array("probability").getItem(1))
             .withColumn("yhat", (F.col("p1") >= 0.5).cast("int")))
    pr_eval = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol=TARGET_COL,
                                            metricName="areaUnderPR")
    roc_eval = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol=TARGET_COL,
                                             metricName="areaUnderROC")
    pr_auc = pr_eval.evaluate(preds)
    roc_auc = roc_eval.evaluate(preds)
    conf = confusion_and_stats(preds)
    metrics = {
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "precision": conf["precision"],
        "recall": conf["recall"],
        "f1": conf["f1"],
        "balanced_accuracy": conf["balanced_accuracy"],
        "MCC": conf["MCC"],
        "fit_time_s": round(fit_time, 2),
    }
    print(f"{name} (VALID): PR-AUC={pr_auc:.4f} | ROC-AUC={roc_auc:.4f} | "
          f"Prec={conf['precision']:.4f} | Rec={conf['recall']:.4f} | F1={conf['f1']:.4f} | "
          f"BalAcc={conf['balanced_accuracy']:.4f} | MCC={conf['MCC']:.4f} | "
          f"train_time_s={metrics['fit_time_s']}")
    return fitted, metrics


def evaluate_on_split(model, df: DataFrame, split_name: str) -> Tuple[Dict[str, float], DataFrame]:
    preds = (model.transform(df)
             .select(TARGET_COL, "probability", "rawPrediction")
             .withColumn("p1", vector_to_array("probability").getItem(1))
             .withColumn("yhat", (F.col("p1") >= 0.5).cast("int")))
    pr_eval = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol=TARGET_COL,
                                            metricName="areaUnderPR")
    roc_eval = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol=TARGET_COL,
                                             metricName="areaUnderROC")
    pr_auc = pr_eval.evaluate(preds)
    roc_auc = roc_eval.evaluate(preds)
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
    plt.title('Precision-Recall Curve (Validation)', fontsize=14)
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


def write_report(path: Path,
                 overall_balance: Dict[str, float],
                 split_stats: Dict[str, Dict[str, float]],
                 valid_metrics: Dict[str, Dict[str, float]],
                 best_model_key: str,
                 test_metrics: Dict[str, float],
                 calibration_rows: List[Dict[str, float]],
                 pos_weight: float,
                 scaling_results: List[Dict[str, float]] = None) -> None:
    lines: List[str] = []
    lines.append("# CS-657 Assignment 2 – Imbalanced Molecular SA Classifier\n")

    lines.append("## Data & Relabeling\n")
    if overall_balance["feature_mode"] == "fp_columns":
        feature_desc = "fp_0..fp_2047 columns"
    elif overall_balance["feature_mode"] == "numeric_columns":
        feature_desc = "digit-indexed fingerprint columns (0..2047)"
    else:
        feature_desc = "2048-bit fp_bits string"
    lines.append(f"- Rows ingested after cleaning: **{overall_balance['total_rows']:,}**\n")
    lines.append(f"- Feature extraction mode: **{overall_balance['feature_mode']}** ({feature_desc})\n")
    lines.append(f"- 80th percentile threshold q80({SA_SCORE_COL}): **{overall_balance['q80']:.4f}**\n")
    pos_frac = overall_balance["positive_fraction"]
    neg_frac = overall_balance["negative_fraction"]
    lines.append(f"- Labeling yields target80 positive fraction ≈ **{pos_frac:.4f}** (negative fraction **{neg_frac:.4f}**)\n")

    lines.append("\n## Split Summary\n")
    lines.append("| Split | Rows | Positive | Negative | Pos Fraction |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: |\n")
    for key in ["train", "valid", "test"]:
        stats = split_stats[key]
        lines.append(f"| {key.title()} | {stats['rows']:,} | {stats['positive']:,} | {stats['negative']:,} | {stats['positive_fraction']:.4f} |\n")

    lines.append("\n## Validation Metrics (PR-AUC focus)\n")
    lines.append("| Model | PR-AUC | ROC-AUC | Balanced Acc | MCC | Train Time (s) |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |\n")
    for key in ["logistic_regression", "logistic_regression_weighted"]:
        metrics = valid_metrics[key]
        lines.append(f"| {human_model_name(key)} | {metrics['PR_AUC']:.4f} | {metrics['ROC_AUC']:.4f} | "
                     f"{metrics['balanced_accuracy']:.4f} | {metrics['MCC']:.4f} | {metrics['fit_time_s']:.2f} |\n")
    lines.append(f"\n*Random Forest validation PR-AUC (for reference): {valid_metrics['random_forest']['PR_AUC']:.4f}*\n")

    lines.append("\n## Test Metrics (Best Model)\n")
    lines.append(f"- Selected model: **{human_model_name(best_model_key)}**\n")
    lines.append(f"- PR-AUC: **{test_metrics['PR_AUC']:.4f}**, ROC-AUC: **{test_metrics['ROC_AUC']:.4f}**\n")
    lines.append(f"- Balanced Accuracy: **{test_metrics['balanced_accuracy']:.4f}**, MCC: **{test_metrics['MCC']:.4f}**\n")
    lines.append(f"- Confusion matrix @0.5 → TP={test_metrics['TP']}, FP={test_metrics['FP']}, "
                 f"TN={test_metrics['TN']}, FN={test_metrics['FN']}\n")

    lines.append("\n## Calibration (VALID, Best Model)\n")
    lines.append("| Bin | Mean Pred | Empirical Pos Rate | Count |\n")
    lines.append("| --- | ---: | ---: | ---: |\n")
    for row in calibration_rows:
        lines.append(f"| {row['bin']} | {row['mean_pred']:.4f} | {row['empirical_pos_rate']:.4f} | {row['count']} |\n")
    lines.append("\n- Lower bins should stay near zero; monitor for over-confidence in the top bins.\n")
    lines.append("- Weighted LR tugged probabilities upward for the minority, but bins remain monotone.\n")

    lines.append("\n## Discussion & Justification\n")
    lines.append(f"- **PR-AUC priority**: With ~20% positives, PR-AUC reflects precision/recall trade-offs better than ROC-AUC; it guides which model keeps minority alerts accurate under imbalance.\n")
    lines.append(f"- **Class weights**: Using pos_weight = neg/pos ≈ {pos_weight:.2f} leverages Spark's native weighting, avoiding heavy resampling loops while boosting recall on hard (positive) molecules.\n")
    lines.append("- **Laptop readiness**: Single-pass training with capped rows and three fixed models finishes under the 6 GB driver budget; outputs feed directly into the submission packet.\n")

    path.write_text("".join(lines), encoding="utf-8")


# Import comprehensive report generator
from report_generator import write_comprehensive_report

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
if "SA_label" in df_raw.columns:
    df_raw = df_raw.drop("SA_label")
df_raw = df_raw.withColumn(SA_SCORE_COL, F.col(SA_SCORE_COL).cast("double"))

df_feat, feature_mode = detect_features(df_raw)
df = df_feat.dropna(subset=["features", SA_SCORE_COL])

q80 = df.approxQuantile(SA_SCORE_COL, [0.80], 1e-4)[0]
df = df.withColumn(TARGET_COL, (F.col(SA_SCORE_COL) >= F.lit(q80)).cast("int"))

# Retain only necessary columns going forward to reduce memory footprint
df = df.select(SA_SCORE_COL, TARGET_COL, "features")

total_rows = df.count()
overall_counts = collect_class_balance(df)
print(f"Total rows after preprocessing: {total_rows:,}")
print(f"q80({SA_SCORE_COL}) = {q80:.4f}")
print("Overall class balance:")
df.groupBy(TARGET_COL).count().orderBy(TARGET_COL).show(truncate=False)

train_df, valid_df, test_df = df.randomSplit([0.8, 0.1, 0.1], seed=SEED)

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

lr_fitted, lr_metrics = evaluate_model(lr_model, train_df, valid_df, "Logistic Regression")
rf_fitted, rf_metrics = evaluate_model(rf_model, train_df, valid_df, "Random Forest")

train_weighted = train_df.withColumn(
    "weight",
    F.when(F.col(TARGET_COL) == 1, F.lit(pos_weight)).otherwise(F.lit(1.0))
)
lr_w_fitted, lr_w_metrics = evaluate_model(lr_weighted_model, train_weighted, valid_df,
                                           "Logistic Regression (class-weighted)")

# Undersampling and oversampling strategies
print("\nTraining models with imbalance handling:")
train_undersampled = undersample_majority(train_df)
train_undersampled = train_undersampled.persist(StorageLevel.MEMORY_AND_DISK)
summarize_split("TRAIN (undersampled)", train_undersampled)
lr_us_fitted, lr_us_metrics = evaluate_model(lr_model, train_undersampled, valid_df,
                                              "Logistic Regression (undersampled)")

train_oversampled = oversample_minority(train_df)
train_oversampled = train_oversampled.persist(StorageLevel.MEMORY_AND_DISK)
summarize_split("TRAIN (oversampled)", train_oversampled)
lr_os_fitted, lr_os_metrics = evaluate_model(lr_model, train_oversampled, valid_df,
                                              "Logistic Regression (oversampled)")

valid_metrics = {
    "logistic_regression": lr_metrics,
    "random_forest": rf_metrics,
    "logistic_regression_weighted": lr_w_metrics,
    "logistic_regression_undersampled": lr_us_metrics,
    "logistic_regression_oversampled": lr_os_metrics,
}

best_name, best_fitted = max(
    [
        ("logistic_regression", lr_fitted, lr_metrics),
        ("random_forest", rf_fitted, rf_metrics),
        ("logistic_regression_weighted", lr_w_fitted, lr_w_metrics),
        ("logistic_regression_undersampled", lr_us_fitted, lr_us_metrics),
        ("logistic_regression_oversampled", lr_os_fitted, lr_os_metrics),
    ],
    key=lambda item: item[2]["PR_AUC"]
)[:2]
print(f"\nBest model by VALID PR-AUC: {best_name}")

test_metrics, _ = evaluate_on_split(best_fitted, test_df, "TEST")

_, valid_preds = evaluate_on_split(best_fitted, valid_df, "VALID (best)")
calib_rows = calibration_table(valid_preds, CALIBRATION_BINS)
print("\nCalibration (VALID) bins:")
for row in calib_rows:
    print(f"  {row['bin']}: mean_pred={row['mean_pred']:.4f}, "
          f"empirical_pos_rate={row['empirical_pos_rate']:.4f}, count={row['count']}")

# Generate visualizations
print("\nGenerating visualizations...")
plot_pr_curve(valid_preds, OUTPUT_DIR / "pr_curve_valid.png")
plot_calibration_curve(calib_rows, OUTPUT_DIR / "calibration_curve_valid.png")

# Scaling experiment
print("\nRunning scaling experiments...")
scaling_results = []
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
    
    start_time = time.time()
    lr_scale_model = LogisticRegression(featuresCol="features", labelCol=TARGET_COL,
                                        weightCol="weight", maxIter=50)
    lr_scale_fitted = lr_scale_model.fit(train_sample_weighted)
    train_time = time.time() - start_time
    
    preds_scale = lr_scale_fitted.transform(valid_df).select(TARGET_COL, "rawPrediction")
    pr_eval_scale = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
                                                   labelCol=TARGET_COL,
                                                   metricName="areaUnderPR")
    pr_auc_scale = pr_eval_scale.evaluate(preds_scale)
    
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
    "q80": q80,
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

# Generate comprehensive report
write_comprehensive_report(Path("REPORT.md"), overall_balance, split_stats, valid_metrics,
                           best_name, test_metrics, calib_rows, pos_weight, scaling_results)
print("Comprehensive report written to REPORT.md")

print("\nAll artifacts written to outputs/:")
artifact_list = [
    "metrics_valid.json", "metrics_valid.csv",
    "metrics_test.json", "metrics_test.csv",
    "class_balance_overall.json", "class_balance_by_split.json",
    "split_stats.txt", "calibration_valid.csv",
    "pr_curve_valid.png", "calibration_curve_valid.png"
]
if scaling_results:
    artifact_list.extend(["scaling_results.json", "scaling_results.csv"])
for path in artifact_list:
    print(f"  - {OUTPUT_DIR / path}")

train_df.unpersist()
valid_df.unpersist()
test_df.unpersist()
train_undersampled.unpersist()
train_oversampled.unpersist()
print("\nPipeline complete!")
spark.stop()
