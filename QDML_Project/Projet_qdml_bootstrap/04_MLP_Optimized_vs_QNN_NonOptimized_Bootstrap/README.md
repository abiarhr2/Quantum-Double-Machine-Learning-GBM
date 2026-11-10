
# Optimized MLP vs Non-Optimized QNN: Paired DML Bootstrap

## 1. Objective

This experiment compares a classical optimized Multilayer Perceptron (MLP) with an original, non-optimized Quantum Neural Network (QNN) using Double Machine Learning (DML).

The objective is to estimate the association between edema and tumor type and evaluate whether the two models produce different DML estimates.

Both models use the same dataset, radiomic features, cross-fitting methodology, and bootstrap samples.

**Important:** The MLP uses previously selected hyperparameters. The QNN uses its original configuration, not the optimized configuration from the QNN hyperparameter search.

## 2. Dataset

**Input file:** `data/GBM_GBS.xlsx`

| Characteristic | Description |
|---|---|
| Patients | 183 |
| Features | 12 radiomic variables |
| Outcome Y | Tumor type: GSM = 1, GBM = 0 |
| Treatment D | Edema: present = 1, absent = 0 |
| Cross-fitting | 5 folds |
| Bootstrap iterations | 1,000 by default |

Only the 12 selected radiomic features are used as covariates.

Age and other clinical variables are excluded.

## 3. Models Compared

### Model 1: Optimized MLP

The MLP configuration was selected during the previous classical model-selection experiments.

| Parameter | Value |
|---|---|
| Architecture | 1 hidden layer |
| Hidden neurons | 32 |
| Activation | ReLU |
| Alpha (L2 regularization) | 10.0 |
| Maximum iterations | 500 |
| Early stopping | Disabled |
| Random seed | 42 by default |

Preprocessing:

1. Replace missing values with the training-fold median.
2. Standardize features using StandardScaler.
3. Train the MLP.
4. Predict probabilities on the held-out fold.

Preprocessing is performed inside each training fold to avoid data leakage.

### Model 2: Non-Optimized QNN

This is the original QNN configuration, before the hyperparameter optimization experiment.

| Parameter | Value |
|---|---|
| Qubits | 12 |
| Quantum layers | 1 |
| Trainable parameters | 36 |
| Learning rate | 0.2 |
| Training epochs | 40 |
| Optimizer | Adam |
| Loss function | Binary cross-entropy |
| Simulator | PennyLane lightning.qubit |
| Gradient method | Adjoint differentiation |

Quantum architecture:

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
12 qubits
        |
        v
StronglyEntanglingLayers
        (1 layer)
        |
        v
PauliZ measurement
        |
        v
Predicted probability
```

The QNN is trained separately for each nuisance function.

## 4. Double Machine Learning

DML uses two nuisance functions:

- `g(X) = P(Y = 1 | X)`: probability of GSM tumor type.
- `m(X) = P(D = 1 | X)`: probability of edema.

For each model, the procedure is:

1. Split patients into five folds.
2. Train the nuisance models using four folds.
3. Predict tumor type and edema on the remaining fold.
4. Repeat until every patient has out-of-fold predictions.
5. Calculate the outcome and treatment residuals.
6. Estimate the DML coefficient theta.

Residuals:

```text
Y_res = Y - g_hat(X)
D_res = D - m_hat(X)
```

DML coefficient:

```text
theta = sum(D_res * Y_res) / sum(D_res²)
```

The script also calculates:

- Analytical standard error
- Z-statistic
- Two-sided p-value
- Approximate 95% confidence interval

The same estimation formula is used for both models.

## 5. Estimation on the Original Dataset

Before the bootstrap, the script can estimate theta using the original dataset.

```text
Original dataset
       |
       +---------------------+
       |                     |
       v                     v
 Optimized MLP         Original QNN
       |                     |
       v                     v
 5-fold DML             5-fold DML
       |                     |
       v                     v
 theta_MLP              theta_QNN
       |                     |
       +----------+----------+
                  |
                  v
       Original-data results
```

Output:

`resultats_bootstrap/resultats_originaux.csv`

This file contains theta, analytical standard error, p-value, confidence interval, and runtime for each model.

## 6. Paired Bootstrap Procedure

The bootstrap measures how the estimated DML coefficients vary when the patient sample changes.

The script uses 1,000 bootstrap iterations by default.

### Procedure for each iteration

1. Randomly sample patients with replacement.
2. Use the exact same resampled patients for the MLP and QNN.
3. Perform five-fold cross-fitting.
4. Retrain both nuisance functions for each model.
5. Estimate theta for the optimized MLP.
6. Estimate theta for the non-optimized QNN.
7. Save both estimates.

The process is repeated across bootstrap iterations.

```text
Original dataset (183 patients)
              |
              v
   Bootstrap resampling
       with replacement
              |
              v
       Same patient sample
              |
       +------+------+
       |             |
       v             v
 Optimized MLP   Non-optimized QNN
       |             |
       v             v
  5-fold DML      5-fold DML
       |             |
       v             v
   theta_MLP      theta_QNN
       |             |
       +------+------+
              |
              v
    Paired difference
    theta_QNN - theta_MLP
              |
              v
       Save iteration
              |
              v
      Repeat 1,000 times
```

### Why paired bootstrap?

Using identical bootstrap samples allows a direct comparison of the two models while reducing variation caused by different patient samples.

The comparison focuses on the difference between the DML coefficients produced by the two models.

### Duplicate patients

Because bootstrap sampling uses replacement, the same patient may appear multiple times.

The script uses StratifiedGroupKFold during bootstrap iterations, grouping duplicate copies of the same original patient.

This prevents copies of the same patient from appearing in both the training and validation folds.

### Invalid iterations

An iteration is excluded if:

- There are insufficient distinct patients in either edema class to form five folds.
- A model raises an exception during estimation.

If either model fails, the iteration is excluded from the paired comparison for both models.

## 7. Bootstrap Statistical Analysis

After the bootstrap iterations, the script combines the saved results.

For each model, it calculates:

| Statistic | Purpose |
|---|---|
| Bootstrap mean | Average theta across iterations |
| Bootstrap median | Median theta |
| Bootstrap SE | Standard deviation of theta |
| 95% percentile CI | 2.5th and 97.5th percentiles |
| Bootstrap p-value | Two-sided sign-based bootstrap measure |
| Valid iterations | Number of successful paired iterations |

The same statistics are calculated for:

`Difference = theta_QNN - theta_MLP`

This directly evaluates whether the models produce different estimates.

The bootstrap p-value is computed from the proportions of estimates above and below zero, with a finite-sample adjustment. It is not a conventional null-centered bootstrap hypothesis test.

## 8. Execution

### Full experiment

```bash
python 04_MLP_Optimized_vs_QNN_NonOptimized_Bootstrap.py
```

By default, this runs the original-data estimation and bootstrap iterations 0 through 999.

The analysis is a separate command.

### Run a subset of iterations

```bash
python 04_MLP_Optimized_vs_QNN_NonOptimized_Bootstrap.py --start 0 --end 50 --skip-original
```

### Run another subset in parallel

```bash
python 04_MLP_Optimized_vs_QNN_NonOptimized_Bootstrap.py --start 50 --end 100 --skip-original
```

### Analyze completed iterations

```bash
python 04_MLP_Optimized_vs_QNN_NonOptimized_Bootstrap.py --analyse-only
```

The script can resume an interrupted bootstrap range using its existing CSV file.

Each iteration is saved immediately.

## 9. Output Files

All results are stored in:

`resultats_bootstrap/`

| File | Description |
|---|---|
| `resultats_originaux.csv` | Original-data DML estimates for both models |
| `boot_START_END.csv` | Bootstrap estimates for a specific iteration range |
| `resume_bootstrap.csv` | Bootstrap statistics for MLP, QNN, and their difference |
| `thetas_bootstrap_fusionnes.csv` | Combined valid bootstrap estimates |
| `bootstrap_histogrammes.png` | Distributions of theta and paired differences |

### Generated figures

The script creates two histograms:

1. Distribution of the bootstrap DML coefficients for MLP and QNN.
2. Distribution of the paired differences `theta_QNN - theta_MLP`.

A vertical reference line at zero helps visualize the direction of the estimates.

## 10. Interpretation

The experiment addresses two questions:

**Question 1: Do both models estimate an association between edema and tumor type?**

This is evaluated using each model's theta estimate and its uncertainty.

**Question 2: Do the classical and quantum models produce different DML estimates?**

This is evaluated using the paired bootstrap distribution of:

`theta_QNN - theta_MLP`

A confidence interval for this difference that includes zero indicates that a zero difference remains compatible with the bootstrap results.

The experiment does not establish a causal effect without the necessary identification assumptions.

## 11. Relationship to the Other Experiments

```text
01-02: MLP hyperparameter selection
              |
              v
       Optimized MLP
       32 neurons
       alpha = 10
              |
              |
Original QNN   |
1 layer        |
40 epochs      |
LR = 0.2       |
      |        |
      v        v
   Paired DML bootstrap
              |
              v
   MLP vs original QNN
   theta comparison
```

This experiment uses the original QNN.

The separate QNN hyperparameter optimization experiment evaluated a configuration with:

- 3 quantum layers
- 120 epochs
- Learning rate 0.05

That optimized QNN configuration is not used in this script.

## 12. Methodological Limitations

- The dataset is small, with only 22 patients without edema.
- Bootstrap iterations may be invalid if too few distinct patients from one class are sampled.
- The original QNN uses fixed hyperparameters rather than the later optimized configuration.
- Both models estimate nuisance functions from the same 12 radiomic features.
- Bootstrap uncertainty does not resolve confounding or identification problems.
- A difference in theta between MLP and QNN does not, by itself, establish that either model is more accurate.
- The script calculates analytical standard errors and bootstrap statistics using different procedures; these should not be treated as interchangeable.

## 13. Key Points

1. The MLP is optimized; the QNN is not.
2. Both models use the same DML estimation formula.
3. Both models are retrained in every bootstrap iteration.
4. Both models receive identical resampled patients.
5. Duplicate patient copies remain in the same cross-fitting fold.
6. The default experiment performs 1,000 bootstrap iterations.
7. The final analysis compares theta estimates and their paired differences.
8. This is a statistical comparison experiment, not a hyperparameter optimization experiment.
