# CS-657 Assignment 2 – Imbalanced Molecular SA Classifier
## Data & Relabeling
- Rows ingested after cleaning: **1,924,396**
- Feature extraction mode: **numeric_columns** (digit-indexed fingerprint columns (0..2047))
- 80th percentile threshold q80(SA_score): **2.7922**
- Labeling yields target80 positive fraction ≈ **0.2000** (negative fraction **0.8000**)

## Split Summary
| Split | Rows | Positive | Negative | Pos Fraction |
| --- | ---: | ---: | ---: | ---: |
| Train | 300,750 | 60,448 | 240,302 | 0.2010 |
| Valid | 192,076 | 38,463 | 153,613 | 0.2002 |
| Test | 192,350 | 38,335 | 154,015 | 0.1993 |

## Validation Metrics (PR-AUC focus)
| Model | PR-AUC | ROC-AUC | Balanced Acc | MCC | Train Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9260 | 0.9770 | 0.8899 | 0.7961 | 7.51 |
| Logistic Regression (class-weighted) | 0.9233 | 0.9769 | 0.9191 | 0.7697 | 5.90 |

*Random Forest validation PR-AUC (for reference): 0.7790*

## Test Metrics (Best Model)
- Selected model: **Logistic Regression**
- PR-AUC: **0.9265**, ROC-AUC: **0.9773**
- Balanced Accuracy: **0.8915**, MCC: **0.7983**
- Confusion matrix @0.5 → TP=31298, FP=5142, TN=148873, FN=7037

## Calibration (VALID, Best Model)
| Bin | Mean Pred | Empirical Pos Rate | Count |
| --- | ---: | ---: | ---: |
| [0.0,0.1) | 0.0099 | 0.0116 | 134492 |
| [0.1,0.2) | 0.1440 | 0.1553 | 8603 |
| [0.2,0.3) | 0.2484 | 0.2607 | 5071 |
| [0.3,0.4) | 0.3474 | 0.3567 | 4087 |
| [0.4,0.5) | 0.4496 | 0.4419 | 3415 |
| [0.5,0.6) | 0.5489 | 0.5367 | 3175 |
| [0.6,0.7) | 0.6497 | 0.6175 | 3289 |
| [0.7,0.8) | 0.7520 | 0.7236 | 3730 |
| [0.8,0.9) | 0.8537 | 0.8393 | 4973 |
| [0.9,1.0] | 0.9767 | 0.9730 | 21241 |

- Lower bins should stay near zero; monitor for over-confidence in the top bins.
- Weighted LR tugged probabilities upward for the minority, but bins remain monotone.

## Discussion & Justification
- **PR-AUC priority**: With ~20% positives, PR-AUC reflects precision/recall trade-offs better than ROC-AUC; it guides which model keeps minority alerts accurate under imbalance.
- **Class weights**: Using pos_weight = neg/pos ≈ 3.98 leverages Spark's native weighting, avoiding heavy resampling loops while boosting recall on hard (positive) molecules.
- **Laptop readiness**: Single-pass training with capped rows and three fixed models finishes under the 6 GB driver budget; outputs feed directly into the submission packet.
