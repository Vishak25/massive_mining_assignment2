# Quick Start Guide - CS-657 Assignment 2

## Prerequisites

1. **PySpark installed** (version 3.3.0 or higher)
2. **Python 3.8+** with numpy and matplotlib
3. **Dataset** in `./moses_molecule_batches_sa/` directory

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

This will install:
- pyspark>=3.3.0
- numpy>=1.21.0
- matplotlib>=3.5.0

## Running the Pipeline

### Basic Run
```bash
spark-submit --driver-memory 6g assignment2_sa_local.py
```

### Expected Runtime
- **Full pipeline:** 15-30 minutes (depending on hardware)
  - Data loading: 2-3 minutes
  - Feature extraction: 1-2 minutes
  - Model training (5 models): 5-10 minutes
  - Scaling experiments: 5-10 minutes
  - Visualization generation: 10-20 seconds
  - Report generation: < 1 second

### Console Output
You'll see detailed progress including:
- Data loading statistics
- Class balance information
- Per-model training progress with metrics
- Calibration analysis
- Scaling experiment results
- File generation confirmations

## What Gets Generated

### 📊 Metrics (Both JSON and CSV)
```
outputs/
├── metrics_valid.json       # All 5 models' validation metrics
├── metrics_valid.csv        # Same data in CSV format
├── metrics_test.json        # Best model's test performance
└── metrics_test.csv         # Same data in CSV format
```

### 📈 Visualizations
```
outputs/
├── pr_curve_valid.png           # Precision-Recall curve
└── calibration_curve_valid.png  # Calibration curve
```

### 📁 Data Summaries
```
outputs/
├── class_balance_overall.json    # Overall class distribution
├── class_balance_by_split.json   # Train/Valid/Test distributions
├── split_stats.txt               # Tab-separated statistics
└── calibration_valid.csv         # Calibration bin data
```

### 🔬 Scaling Experiment
```
outputs/
├── scaling_results.json    # Training time vs. dataset size
└── scaling_results.csv     # Same data in CSV format
```

### 📝 Comprehensive Report
```
REPORT.md    # ~5 page report with intro, methods, results, discussion
```

## Understanding the Output

### Validation Metrics CSV
Open `outputs/metrics_valid.csv` to compare all models:

| model | PR_AUC | ROC_AUC | precision | recall | f1 | balanced_accuracy | MCC | fit_time_s |
|-------|--------|---------|-----------|--------|----|--------------------|-----|------------|
| LR | ... | ... | ... | ... | ... | ... | ... | ... |
| LR-weighted | ... | ... | ... | ... | ... | ... | ... | ... |
| LR-undersampled | ... | ... | ... | ... | ... | ... | ... | ... |
| LR-oversampled | ... | ... | ... | ... | ... | ... | ... | ... |
| RF | ... | ... | ... | ... | ... | ... | ... | ... |

### Visualizations
1. **PR Curve (`pr_curve_valid.png`)**: Shows precision-recall trade-off
   - Higher and to the right = better performance
   - Area under curve = PR-AUC metric

2. **Calibration Curve (`calibration_curve_valid.png`)**: Shows probability calibration
   - Points close to diagonal = well-calibrated
   - Above diagonal = overconfident
   - Below diagonal = underconfident

### Scaling Results
Open `outputs/scaling_results.csv`:

| train_rows | train_time_s | pr_auc |
|------------|--------------|--------|
| 100000 | X.XX | 0.XXXX |
| 500000 | Y.YY | 0.YYYY |
| 1000000 | Z.ZZ | 0.ZZZZ |

Analyze:
- How training time scales (should be ~linear)
- How PR-AUC improves with more data
- Diminishing returns point

### Comprehensive Report
Open `REPORT.md` in a Markdown viewer or convert to PDF:

```bash
# Convert to PDF (requires pandoc)
pandoc REPORT.md -o REPORT.pdf
```

Report sections:
1. Introduction - Problem context and objectives
2. Methods - Data, strategies, models, metrics
3. Results - Tables and key findings
4. Discussion - Analysis and interpretation
5. Conclusion - Summary and recommendations

## Key Findings to Look For

### 1. Which imbalance strategy works best?
Compare PR-AUC across:
- Class weighting
- Undersampling  
- Oversampling

**Expected:** Class weighting typically wins (preserves all data)

### 2. Precision vs. Recall Trade-off
Check the validation metrics:
- High precision = fewer false positives
- High recall = fewer false negatives
- F1 balances both

**For drug discovery:** May want higher precision to avoid wasting resources on false positives

### 3. Model Calibration
Check calibration curve:
- Are predicted probabilities trustworthy?
- Can we use them for risk-based decision making?

### 4. Scaling Behavior
From scaling results:
- Is more data worth the computational cost?
- Where are diminishing returns?

## Troubleshooting

### Out of Memory
```bash
# Increase driver memory
spark-submit --driver-memory 8g assignment2_sa_local.py

# Or reduce MAX_TRAIN_ROWS in the script (line 39)
MAX_TRAIN_ROWS = 200_000  # instead of 300_000
```

### Scaling Experiment Takes Too Long
Edit `assignment2_sa_local.py` line 585:
```python
# Reduce experiment sizes
for size in [100_000, 300_000]:  # instead of [100_000, 500_000, 1_000_000]
```

### Matplotlib Display Issues
The script uses `matplotlib.use('Agg')` for non-interactive backend.
Visualizations save to PNG files automatically.

### Missing Dataset
Ensure dataset directory structure:
```
moses_molecule_batches_sa/
├── moses_fp_batch_0.csv.gz
├── moses_fp_batch_1.csv.gz
├── ...
└── moses_fp_batch_N.csv.gz
```

## Next Steps

1. **Review Console Output** - Check for any errors or warnings
2. **Open REPORT.md** - Read the comprehensive analysis
3. **Examine Visualizations** - Look at the PNG files
4. **Analyze Metrics CSVs** - Compare models quantitatively
5. **Review Scaling Results** - Understand computational trade-offs

## For Submission

Include these files:
- ✅ `assignment2_sa_local.py` - Main script
- ✅ `report_generator.py` - Report generation module
- ✅ `requirements.txt` - Dependencies
- ✅ `README.md` - Project overview
- ✅ `REPORT.md` - Comprehensive report (or convert to PDF)
- ✅ `outputs/` directory - All generated artifacts
  - Metrics (JSON + CSV)
  - Visualizations (PNG)
  - Calibration data
  - Scaling results

Optional but recommended:
- ✅ `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- ✅ `QUICKSTART.md` - This guide

---

**Questions?** Check:
1. Console output for detailed progress
2. REPORT.md for comprehensive analysis
3. IMPLEMENTATION_SUMMARY.md for technical details

