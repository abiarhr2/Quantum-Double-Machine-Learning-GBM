
# QNN Hyperparameter Optimization

## 1. Objective

This experiment evaluates Quantum Neural Network (QNN) configurations for estimating two nuisance functions used in Double Machine Learning (DML).

- `g(X) = P(Y = 1 | X)`: tumor type (GSM vs GBM)
- `m(X) = P(D = 1 | X)`: edema presence

The objective is to minimize out-of-fold predictive log-loss:

`LL_total = LL_g + LL_m`

Lower log-loss indicates better prediction.

This experiment does not estimate the DML coefficient theta, standard errors, confidence intervals, or p-values.

## 2. Dataset

Input file: `data/GBM_GBS.xlsx`

| Characteristic | Value |
|---|---|
| Patients | 183 |
| Radiomic features | 12 |
| Tumor classes | GBM and GSM |
| Edema present | 161 |
| Edema absent | 22 |
| Cross-validation | 5-fold StratifiedKFold |

Cross-validation folds are stratified on edema status.

Missing values are replaced using median imputation fitted on the training fold. QNN features are scaled to [-π, π] using MinMaxScaler fitted on the training fold.

## 3. QNN Architecture

The model is implemented using PennyLane and the `lightning.qubit` simulator.

```text
12 radiomic features
         |
         v
Median imputation
         |
         v
MinMaxScaler [-pi, pi]
         |
         v
AngleEmbedding (RY)
         |
         v
12 quantum qubits
         |
         v
StronglyEntanglingLayers
         |
         v
PauliZ measurement (qubit 0)
         |
         v
Predicted probability
         |
         v
Binary cross-entropy
```

Training uses full-batch Adam optimization with adjoint differentiation.

Each epoch represents one Adam parameter-update step.

The number of trainable parameters is:

`Parameters = 12 qubits × 3 rotations × number of layers`

## 4. Stage 1: Initial Hyperparameter Screening

### Objective

Evaluate learning rates and training durations using a single quantum layer.

### Configurations

| Parameter | Values |
|---|---|
| Learning rates | 0.01, 0.05, 0.1, 0.2 |
| Epochs | 40, 80 |
| Quantum layers | 1 |
| Random seed | 42 |
| Cross-validation folds | 5 |

Eight configurations were evaluated.

Each configuration required five folds and two nuisance models, giving 80 QNN training runs.

### Stage 1 Results

| Learning rate | Epochs | Total log-loss |
|---|---:|---:|
| 0.01 | 80 | **1.3834** |
| 0.01 | 40 | 1.3863 |
| 0.05 | 80 | 1.3877 |
| 0.05 | 40 | 1.3907 |
| 0.20 | 40 | 1.3925 |
| 0.10 | 80 | 1.3934 |
| 0.20 | 80 | 1.3941 |
| 0.10 | 40 | 1.3947 |

**Stage 1 finding:** Learning rate 0.01 with 80 epochs achieved the lowest combined log-loss among the eight tested configurations.

The three best Stage 1 candidates were:

1. `lr0.01_ep80_L1`
2. `lr0.01_ep40_L1`
3. `lr0.05_ep80_L1`

These were the candidates recorded in `stage2_plan.json`.

## 5. Stage 2: QNN Refinement

### Planned methodology

The Python script generates Stage 2 candidates by:

1. Selecting the three best Stage 1 configurations.
2. Testing 1, 2 and 3 quantum layers.
3. Adding longer training when convergence diagnostics justify it.
4. Repeating each configuration using seeds 1, 42 and 123.
5. Comparing predictive performance using identical validation folds.

### Actual recorded Stage 2 results

The supplied `stage2_results.csv` contains only one configuration:

| Parameter | Value |
|---|---|
| Configuration | `lr0.05_ep120_L3` |
| Learning rate | 0.05 |
| Epochs | 120 |
| Quantum layers | 3 |
| Qubits | 12 |
| Trainable parameters | 108 |
| Seeds | 1, 42, 123 |
| Cross-validation folds | 5 |

This corresponds to 30 QNN training runs.

The Stage 2 plan and results show that this configuration was evaluated separately from the full automatically generated search.

The supplied current Python script contains the automatic search logic, while the saved Stage 2 outputs contain only the single configuration. Therefore, the exact code version or execution procedure that restricted Stage 2 to this configuration is not established by the current script alone.

### Results Across Seeds

| Seed | Total log-loss |
|---|---:|
| 1 | 1.1607 |
| 42 | 1.1218 |
| 123 | 1.1417 |
| **Mean** | **1.1414** |
| Standard deviation | 0.0195 |

The recorded Stage 2 configuration was successful across all three seeds.

## 6. Predictive Performance

Results averaged across the three Stage 2 seeds.

| Metric | Tumor type g(X) | Edema m(X) |
|---|---:|---:|
| Log-loss | 0.6263 | 0.5151 |
| AUC | 0.7024 | 0.4919 |
| Brier score | 0.2190 | 0.1647 |

**Combined QNN log-loss: 1.1414**

The QNN achieved moderate discrimination for tumor type, but its edema AUC was approximately 0.49, indicating no useful discrimination in this evaluation.

## 7. Comparison With Classical Baselines

The QNN was compared against:

- Optimized classical MLP: 32 hidden neurons, alpha = 10, maximum 500 iterations, no early stopping.
- Prevalence-only model: predicts the proportion of positive cases in each training fold.

All models were evaluated using the same validation folds for each seed.

### Results

| Model | LL_g | LL_m | LL_total |
|---|---:|---:|---:|
| Optimized MLP | 0.4598 | 0.3828 | **0.8426** |
| Prevalence baseline | 0.6962 | 0.3678 | 1.0640 |
| QNN (0.05, 120, 3) | 0.6263 | 0.5151 | 1.1414 |

### Interpretation

The optimized MLP achieved the lowest combined log-loss.

The QNN predicted tumor type better than the prevalence baseline, but performed worse on edema prediction.

For edema:

`LL_m(QNN) - LL_m(prevalence) = +0.1474`

For combined prediction:

`LL_total(QNN) - LL_total(MLP) = +0.2988`

Positive differences indicate worse predictive performance for the QNN.

The QNN did not outperform the MLP or the prevalence-only model on combined log-loss.

## 8. Training Convergence

Figure: `training_loss_curves.png`

The figure displays average training binary cross-entropy over 120 epochs for both nuisance functions.

### Tumor type g(X)

Training loss decreased gradually from approximately 0.68 to 0.57.

The curve continued decreasing near epoch 120, although the improvements became smaller.

### Edema m(X)

Training loss decreased rapidly during the first 40 epochs, then flattened near 0.46.

The model showed a relatively stable training trajectory toward the end.

### Recorded Diagnostics

| Metric | Value |
|---|---:|
| Mean initial training loss | 0.7023 |
| Mean final training loss | 0.5125 |
| Relative improvement near end | 0.0038 |
| Fraction flagged not converged | 0% |
| Fraction flagged unstable | 0% |

The script's convergence criteria did not flag the selected configuration as unstable or insufficiently converged.

However, decreasing training loss does not guarantee strong out-of-fold predictive performance.

## 9. Computational Execution

The experiments were prepared for execution on the DICO cluster using SLURM.

### Stage 1

Script: `job_qnn_stage1.sh`

### Stage 2

Script: `job_qnn_stage2.sh`

Both scripts request:

| Resource | Value |
|---|---|
| Tasks | 1 |
| CPUs | 4 |
| Memory | 16 GB |
| Time limit | 4 hours |
| Python environment | `~/venv_qdml` |

### Recorded Computational Cost

| Metric | Value |
|---|---:|
| Total logged QNN trainings | 350 |
| Failed trainings | 0 |
| Cumulative training time | 7.50 hours |
| Mean training duration | 77 seconds |

The training log contains configurations beyond the final three-row Stage 2 CSV, including additional one-, two-, and three-layer configurations.

Consequently, the cumulative runtime reflects all logged training records, not only the final Stage 2 configuration.

## 10. Saved Results

| File | Description |
|---|---|
| `03_QNN_Hyperparameter_Optimization.py` | QNN optimization implementation |
| `job_qnn_stage1.sh` | SLURM Stage 1 job |
| `job_qnn_stage2.sh` | SLURM Stage 2 job |
| `stage1_results.csv` | Eight initial configurations |
| `stage2_plan.json` | Saved refinement plan |
| `stage2_results.csv` | Three-seed evaluation of the selected QNN |
| `qnn_configuration_ranking.csv` | QNN and baseline performance summary |
| `best_qnn_config.json` | Recorded best Stage 2 configuration |
| `optimization_summary.txt` | Detailed numerical summary |
| `train_log.jsonl` | Individual training records |
| `training_loss_curves.png` | QNN training loss curves |
| `logloss_comparison.png` | QNN, MLP and prevalence comparison |

## 11. Main Findings

1. Stage 1 tested eight configurations and identified learning rate 0.01 with 80 epochs as the best single-layer candidate.

2. The final saved Stage 2 evaluation focused on a three-layer QNN with learning rate 0.05 and 120 epochs.

3. This configuration used 12 qubits and 108 trainable parameters.

4. Across three random seeds, the QNN achieved a mean combined log-loss of 1.1414.

5. Training was numerically stable according to the script's diagnostics.

6. The optimized MLP achieved substantially better combined predictive performance (0.8426).

7. The QNN did not outperform the prevalence-only baseline for edema prediction.

8. The saved Stage 2 ranking contains only one QNN configuration. Its designation as "best" therefore does not establish that it outperformed all other configurations previously tested.

## 12. Relationship to the DML Bootstrap Experiment

The configuration `lr0.05_ep120_L3` can be used as the optimized QNN architecture in the later DML bootstrap comparison.

However, this hyperparameter optimization experiment evaluates nuisance prediction only.

The final bootstrap analysis is a separate experiment used to estimate the DML coefficient and its uncertainty.

Predictive performance and DML inference must therefore be reported separately.
