
# MLP Model Selection and Hyperparameter Optimization

## 1. Purpose

This folder contains experiments used to select a classical Multilayer Perceptron (MLP) for nuisance-function estimation within the Double Machine Learning (DML) framework.

The objective is to identify an MLP configuration that predicts the outcome and treatment accurately before comparing classical and quantum neural networks.

Model selection is based on out-of-fold prediction log-loss, not on the significance of the estimated treatment effect.

## 2. Dataset

**Input:** `data/GBM_GBS.xlsx`

- 183 patients
- 12 selected radiomic features
- Outcome `Y`: tumor type (1 = GSM, 0 = GBM)
- Treatment `D`: edema indicator (1 = edema present, 0 = absent)
- Covariates `X`: 12 radiomic features

The same dataset and feature list are used throughout these experiments.

## 3. Double Machine Learning

Each candidate model is evaluated inside a DML procedure.

Two nuisance functions are estimated:

- `g(X) = E[Y | X]`: predicts tumor type from radiomic features.
- `m(X) = E[D | X]`: predicts edema status from radiomic features.

The DML procedure:

1. Divides the data into five cross-fitting folds.
2. Trains separate nuisance models for `Y` and `D`.
3. Predicts observations excluded from each training fold.
4. Calculates the residuals `Y - g(X)` and `D - m(X)`.
5. Estimates the DML coefficient `theta`.
6. Calculates its analytical standard error and p-value.
7. Evaluates nuisance prediction using log-loss.

The selection criterion is:

`Total log-loss = log-loss(g) + log-loss(m)`

**Lower values indicate better out-of-fold nuisance prediction.**

The coefficient, standard error, and p-value are used for sensitivity analysis, not hyperparameter selection.

## 4. Experiment 01: MLP Configuration Screening

**Python file:** `01_MLP_Hidden_Neuron_Selection.py`

### Purpose

Explore different hidden-layer sizes and assess the sensitivity of the DML estimates to the MLP architecture.

### Configuration in the supplied Python script

| Parameter | Value |
|---|---|
| Hidden neurons | 4, 8, 12, 16, 32 |
| Hidden layers | 1 |
| Activation | ReLU |
| Alpha | 0.01 |
| Early stopping | Enabled |
| Cross-fitting folds | 5 |
| Random seeds | 10 |
| Maximum training iterations | 2,000 |

This version specifies 50 DML evaluations.

### Associated CSV results

- `01_MLP_Neuron_Selection_Detail.csv`
- `01_MLP_Neuron_Selection_Summary.csv`

**Important discrepancy:** The CSV files contain 1,350 evaluations across 45 configurations, not the 50 evaluations specified in the supplied Python script.

The CSV results therefore come from a broader configuration search or a different version of the script. Their exact generating code has not yet been identified.

### Essential results from the CSV

| Metric | Result |
|---|---|
| Configurations evaluated | 45 |
| Total DML evaluations | 1,350 |
| Best configuration | 32 neurons |
| Alpha | 10.0 |
| Early stopping | Disabled |
| Mean outcome log-loss | 0.4655 |
| Mean treatment log-loss | 0.3810 |
| Mean total log-loss | 0.8465 |
| Mean theta | 0.2026 |
| Mean analytical SE | 0.0756 |

**Finding:** Among the 45 configurations recorded in this CSV, the best mean combined log-loss was obtained using 32 hidden neurons, alpha = 10, and early stopping disabled.

This result cannot be attributed to the exact File 01 code supplied above without finding the version that generated the CSV.

## 5. Experiment 02: Extended MLP Hyperparameter Optimization

**Python file:** `02_MLP_Hyperparameter_Optimization.py`

### Purpose

Evaluate MLP architecture and L2 regularization more systematically, while comparing the MLP against simpler reference models.

### Configurations

| Parameter | Values |
|---|---|
| Hidden neurons | 8, 16, 32 |
| Alpha | 10, 30, 100 |
| Early stopping | Disabled |
| Activation | ReLU |
| Cross-fitting folds | 5 |
| Random seeds | 30 |
| Maximum training iterations | 500 |

Additional reference models:

- Logistic regression with C = 0.01, 0.1, 1, 10
- Prevalence-only prediction baseline

### Number of evaluations

- 9 MLP configurations
- 4 logistic regression configurations
- 1 prevalence-only configuration
- 30 seeds per configuration

**Total: 14 configurations × 30 seeds = 420 DML evaluations.**

### Output files

- `02_MLP_Hyperparameter_Optimization_Detail.csv`
- `02_MLP_Hyperparameter_Optimization_Summary.csv`

The CSV contents match the configuration structure and evaluation count of the supplied File 02 script.

### Essential results

| Configuration | Mean total log-loss |
|---|---:|
| MLP, 32 neurons, alpha 10 | **0.8460** |
| MLP, 16 neurons, alpha 10 | 0.8566 |
| Logistic regression, C = 0.1 | 0.8668 |

The best MLP configuration recorded in the summary is:

- Hidden layers: 1
- Hidden neurons: 32
- Activation: ReLU
- Alpha: 10.0
- Early stopping: Disabled
- Maximum training iterations: 500

Its recorded statistics are:

| Metric | Value |
|---|---:|
| Outcome log-loss | 0.4652 |
| Treatment log-loss | 0.3808 |
| Total log-loss | 0.8460 |
| Mean theta | 0.2029 |
| Mean analytical SE | 0.0757 |
| Mean p-value | 0.0083 |
| Number of model parameters | 449 |

The paired comparison procedure evaluates the difference in log-loss between each MLP and the best-performing MLP across identical seeds.

The selected configuration is the smallest MLP among configurations considered statistically indistinguishable from the best by the script's comparison rule.

## 6. Relationship Between the Two Experiments

The experiments belong to the same model-selection process:

```text
Medical dataset
      |
      v
12 radiomic features
      |
      v
Experiment 01
MLP configuration screening
      |
      v
Experiment 02
Extended MLP optimization
      |
      v
Best recorded MLP configuration
32 neurons, alpha = 10
      |
      v
Final DML / bootstrap experiments
(configuration must be verified)
```

Experiment 02 does not automatically load Experiment 01's results.

Both experiments independently evaluate candidate configurations using DML and nuisance prediction log-loss.

The fact that their CSV summaries identify the same best configuration supports the consistency of the recorded searches, but does not establish which code generated the first CSV.

## 7. Methodological Details

### Cross-fitting

Both supplied scripts use five-fold `StratifiedKFold`, stratified on edema status.

They do not use patient-group-aware `StratifiedGroupKFold`.

### Missing values

Missing radiomic values are replaced using column medians.

In both supplied scripts, median imputation occurs before cross-fitting. This can introduce preprocessing leakage.

### Standardization

File 01 standardizes the full feature matrix before cross-fitting.

File 02 uses a scikit-learn pipeline to standardize features within each training fold, which avoids leakage from scaling.

### Hyperparameter selection

The scripts use prediction log-loss to select nuisance models.

They do not select models based on theta, p-values, or whether the estimated effect is statistically significant.

### Statistical interpretation

A lower nuisance log-loss does not necessarily imply a more accurate causal-effect estimate.

The DML coefficient can be interpreted causally only under the required identification assumptions.

## 8. Relationship to the Final QDML Project

These experiments belong to the **classical MLP model-selection stage**.

They are separate from:

- Original QNN training
- Quantum circuit optimization
- The 1,000 bootstrap replications
- SLURM execution on DICO
- Final MLP/QNN/QNN_OPT statistical comparison

The final bootstrap results reported for the MLP are:

| Metric | Value |
|---|---:|
| Original-data theta | 0.1918 |
| Original-data SE | 0.0779 |
| Original-data p-value | 0.0138 |
| Bootstrap iterations | 1,000 |
| Bootstrap mean theta | 0.2000 |
| Bootstrap SE | 0.0821 |
| Bootstrap 95% CI | [0.0442, 0.3624] |
| Bootstrap p-value | 0.0160 |

These are results of the later bootstrap analysis, not the two optimization scripts.

**The final bootstrap source code must be checked to confirm that it uses the selected MLP configuration: 32 hidden neurons, alpha = 10, early stopping disabled.**

## 9. File Inventory

```text
MLP_Optimization/
|
|-- 01_MLP_Hidden_Neuron_Selection.py
|-- 01_MLP_Neuron_Selection_Detail.csv
|-- 01_MLP_Neuron_Selection_Summary.csv
|
|-- 02_MLP_Hyperparameter_Optimization.py
|-- 02_MLP_Hyperparameter_Optimization_Detail.csv
|-- 02_MLP_Hyperparameter_Optimization_Summary.csv
|
|-- README.md
```

## 10. Key Findings

1. Both experiments evaluate MLP configurations using out-of-fold nuisance prediction log-loss.
2. The first CSV records 45 configurations and 1,350 evaluations, which do not match the supplied File 01 script.
3. The second CSV records 14 configurations and 420 evaluations, consistent with File 02.
4. Both CSV summaries identify an MLP with 32 hidden neurons and alpha = 10 as the best configuration by mean combined log-loss.
5. The final MLP configuration used in the 1,000-iteration bootstrap has not yet been verified from its training code.
6. The optimization results describe nuisance prediction performance, while the final bootstrap describes the uncertainty of the DML coefficient.
