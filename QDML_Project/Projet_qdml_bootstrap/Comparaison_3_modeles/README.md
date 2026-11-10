
# Final DML Comparison: Optimized MLP vs Original QNN vs Optimized QNN

## 1. Project Objective

This experiment compares three machine learning models within a Double Machine Learning (DML) framework to investigate the association between edema and brain tumor type (GBM vs GSM).

The three models are:

1. **Optimized MLP:** Classical neural network with selected hyperparameters.
2. **Original QNN:** Quantum neural network before optimization.
3. **Optimized QNN:** Quantum neural network with optimized architecture and training hyperparameters.

The objective is to evaluate:

- The DML coefficient (theta) estimated by each model.
- The uncertainty and stability of these estimates.
- Whether QNN optimization changes the estimated coefficient.
- Whether the quantum models produce different estimates from the classical MLP.

This comparison script uses previously generated results. It does not retrain the models.

## 2. Dataset

**Dataset:** `data/GBM_GBS.xlsx`

| Characteristic | Value |
|---|---|
| Total patients | 183 |
| GBM patients | 100 |
| GSM patients | 83 |
| Radiomic features | 12 |
| Treatment variable (D) | Edema presence |
| Outcome variable (Y) | Tumor type (GSM = 1, GBM = 0) |
| Cross-fitting | 5 folds |
| Bootstrap iterations | 1,000 per model |

The same 12 radiomic features were used in all three models.

## 3. Model Configurations

| Parameter | Optimized MLP | Original QNN | Optimized QNN |
|---|---|---|---|
| Architecture | Classical MLP | Quantum circuit | Quantum circuit |
| Hidden neurons | 32 | N/A | N/A |
| Qubits | N/A | 12 | 12 |
| Quantum layers | N/A | 1 | 3 |
| Trainable quantum parameters | N/A | 36 | 108 |
| Learning rate | Scikit-learn default | 0.2 | 0.05 |
| Training iterations/epochs | 500 maximum | 40 | 120 |
| Regularization alpha | 10.0 | N/A | N/A |
| Early stopping | Disabled | N/A | N/A |
| Optimizer | Adam (MLPClassifier default) | Adam | Adam |

The MLP configuration was selected through previous classical hyperparameter experiments.

The optimized QNN configuration was evaluated in the QNN hyperparameter optimization stage.

## 4. Analysis Method

### Double Machine Learning

For each model, two nuisance functions were estimated:

- `g(X) = P(Y = 1 | X)`: probability of GSM tumor type.
- `m(X) = P(D = 1 | X)`: probability of edema.

Five-fold cross-fitting was used to generate out-of-fold predictions.

Residuals were calculated as:

`Y_res = Y - g_hat(X)`

`D_res = D - m_hat(X)`

The DML coefficient was estimated using:

`theta = sum(D_res * Y_res) / sum(D_res²)`

The same DML estimation formula was applied to all three models.

### Bootstrap Procedure

Each model was evaluated using 1,000 bootstrap iterations.

At each iteration:

1. Patients were sampled with replacement.
2. Nuisance models were retrained on the bootstrap sample.
3. Five-fold cross-fitting was performed.
4. The DML coefficient theta was calculated.
5. The estimate was saved for statistical analysis.

The optimized QNN experiment reused the bootstrap sampling scheme from the previous MLP and original QNN experiment.

The final comparison script checks bootstrap sample identities and cross-fitting compatibility before calculating paired differences involving the optimized QNN.

## 5. Final Results on the Original Dataset

These results were obtained from DML estimation on the original dataset, before bootstrap resampling.

| Metric | Optimized MLP | Original QNN | Optimized QNN |
|---|---:|---:|---:|
| Theta | 0.1918 | 0.0382 | 0.1544 |
| Analytical SE | 0.0779 | 0.0742 | 0.0836 |
| 95% CI lower | 0.0391 | -0.1072 | -0.0096 |
| 95% CI upper | 0.3446 | 0.1836 | 0.3183 |
| p-value | 0.0138 | 0.6069 | 0.0649 |

### Interpretation

**Optimized MLP**

The MLP produced the largest positive original-data estimate (theta = 0.1918).

Its analytical 95% confidence interval excludes zero, with p = 0.0138.

**Original QNN**

The original QNN produced a much smaller estimate (theta = 0.0382).

Its confidence interval includes zero, with p = 0.6069.

**Optimized QNN**

The optimized QNN produced a larger estimate than the original QNN (theta = 0.1544).

Its analytical confidence interval includes zero, with p = 0.0649.

This represents a change in the estimated coefficient following QNN optimization, but does not establish a statistically significant improvement in estimation accuracy.

## 6. Bootstrap Results (1,000 Iterations)

All three models have 1,000 valid bootstrap estimates in the final results file.

| Metric | Optimized MLP | Original QNN | Optimized QNN |
|---|---:|---:|---:|
| Bootstrap mean theta | 0.2000 | 0.0373 | 0.1222 |
| Bootstrap median theta | 0.2018 | 0.0358 | 0.1277 |
| Bootstrap SE | 0.0821 | 0.0741 | 0.0865 |
| 95% CI lower | 0.0442 | -0.1121 | -0.0547 |
| 95% CI upper | 0.3624 | 0.1868 | 0.2911 |
| Reported bootstrap p-value | 0.0160 | 0.6114 | 0.1718 |
| Valid iterations | 1,000 | 1,000 | 1,000 |

### Interpretation

**Optimized MLP**

- Mean theta: 0.2000
- Bootstrap 95% CI: [0.0442, 0.3624]
- Reported bootstrap p-value: 0.0160

The bootstrap distribution is centered around a positive coefficient of approximately 0.20.

Its percentile confidence interval excludes zero.

**Original QNN**

- Mean theta: 0.0373
- Bootstrap 95% CI: [-0.1121, 0.1868]
- Reported bootstrap p-value: 0.6114

The original QNN produces estimates close to zero.

Its percentile confidence interval includes both negative and positive values.

**Optimized QNN**

- Mean theta: 0.1222
- Bootstrap 95% CI: [-0.0547, 0.2911]
- Reported bootstrap p-value: 0.1718

The optimized QNN produces a larger positive mean coefficient than the original QNN.

However, its percentile confidence interval still includes zero.

### Overall Observation

The bootstrap mean coefficients follow this order:

`Optimized MLP (0.2000) > Optimized QNN (0.1222) > Original QNN (0.0373)`

The optimized QNN moves closer to the MLP estimate after optimization.

However, similarity to the MLP coefficient does not independently demonstrate greater statistical accuracy or better causal estimation.

## 7. Paired Bootstrap Comparisons

Paired comparisons evaluate differences between models using matched bootstrap iterations.

The difference is calculated as:

`Difference = theta_Model_A - theta_Model_B`

### Results

| Comparison | Mean difference | Bootstrap SE | 95% CI | Reported p-value |
|---|---:|---:|---|---:|
| Optimized QNN - MLP | -0.0778 | 0.0869 | [-0.2434, 0.0802] | 0.3936 |
| Optimized QNN - Original QNN | +0.0849 | 0.0661 | [-0.0520, 0.2147] | 0.2078 |
| Original QNN - MLP | -0.1627 | 0.0820 | [-0.3230, 0.0022] | 0.0539 |

All three comparisons contain 1,000 valid paired observations in the final CSV.

### Comparison 1: Optimized QNN vs MLP

Mean difference: -0.0778

The optimized QNN produces a smaller mean coefficient than the optimized MLP.

The 95% confidence interval includes zero.

The reported bootstrap p-value is 0.3936.

**Finding:** The reported paired statistics do not provide sufficient evidence of a nonzero difference at the conventional 5% threshold.

### Comparison 2: Optimized QNN vs Original QNN

Mean difference: +0.0849

The optimized QNN produces a larger mean coefficient than the original QNN.

The 95% confidence interval includes zero.

The reported bootstrap p-value is 0.2078.

**Finding:** QNN optimization changed the estimated coefficient, but the paired results do not establish a statistically significant nonzero difference at the 5% threshold.

### Comparison 3: Original QNN vs MLP

Mean difference: -0.1627

The original QNN produces a substantially smaller mean coefficient than the optimized MLP.

The 95% confidence interval is [-0.3230, 0.0022], which narrowly includes zero.

The reported bootstrap p-value is 0.0539.

**Finding:** This comparison is close to the conventional 5% threshold, but does not cross it according to the reported statistics.

## 8. Computational Performance

The final results CSV also contains execution-time information.

### Original-Data Estimation

| Model | Runtime |
|---|---:|
| Optimized MLP | 1.06 seconds |
| Original QNN | 394.44 seconds |
| Optimized QNN | 2,668.47 seconds |

The optimized QNN required substantially more computation than the classical MLP.

### Optimized QNN Bootstrap

| Metric | Value |
|---|---:|
| Mean time per iteration | 2,336.29 seconds |
| Total recorded compute time | 648.97 hours |
| Valid iterations | 1,000 |

The total compute time is cumulative across iterations. It is not necessarily the elapsed wall-clock time when jobs are executed in parallel.

The original MLP/QNN bootstrap results do not provide comparable per-iteration runtime measurements in the final CSV.

## 9. Input Files

The comparison script reads existing results from two folders.

### Previous MLP and Original QNN Experiment

`resultats_bootstrap_MLP_Optimized_Not_QNN_Optimized/`

Expected contents:

- `boot_START_END.csv`
- `resultats_originaux.csv`

### Optimized QNN Experiment

`resultats_bootstrap_QNN_OPT/`

Expected contents:

- `boot_START_END.csv`
- `config_QNN_OPT.json`
- `resultats_originaux_QNN_OPT.csv`

The script also uses `data/GBM_GBS.xlsx` to verify bootstrap sample identities.

The consolidated file `final_qdml_comparison_results.csv` contains the final reported numerical results, but is not itself an input read by the supplied comparison script.

## 10. Execution

**Python script:** `06_Compare_Three_DML_Models.py`

Run with default paths:

```bash
python 06_Compare_Three_DML_Models.py
```

Run with explicitly specified directories:

```bash
python 06_Compare_Three_DML_Models.py \
  --prev-dir resultats_bootstrap_MLP_Optimized_Not_QNN_Optimized \
  --new-dir resultats_bootstrap_QNN_OPT \
  --out-dir comparaison_3_modeles \
  --data data/GBM_GBS.xlsx \
  --B 1000
```

The script reads the previous results and generates comparison tables and figures.

No model training is performed.

## 11. Generated Files

**Output directory:** `comparaison_3_modeles/`

| File | Description |
|---|---|
| `comparaison_modeles.csv` | Original-data and bootstrap statistics for all three models |
| `differences_appariees.csv` | Paired differences and their bootstrap statistics |
| `thetas_apparies.csv` | Bootstrap estimates matched by iteration |
| `hist_thetas.png` | Bootstrap distributions of theta |
| `hist_differences.png` | Distributions of paired differences |
| `forest_ic.png` | Analytical vs bootstrap confidence intervals |

The comparison script generates these files when the required data are available and the relevant analysis conditions are satisfied.

The supplied consolidated CSV is an additional results artifact. Its creation is not part of the supplied comparison script.

## 12. Interpretation of the Figures

### Bootstrap Distributions: `hist_thetas.png`

Shows the distribution of estimated theta values across bootstrap iterations for:

- Optimized MLP
- Original QNN
- Optimized QNN

The distributions can be compared in terms of center, spread, and overlap.

### Paired Differences: `hist_differences.png`

Shows the distribution of the differences between model estimates for matched bootstrap iterations.

This is used to examine whether the estimated differences are consistently positive or negative.

### Confidence Interval Comparison: `forest_ic.png`

Shows both:

- Original-data DML estimates with analytical confidence intervals.
- Bootstrap mean estimates with percentile confidence intervals.

This allows comparison of the two uncertainty estimation approaches.

## 13. Methodological Limitations

### Small Sample Size

The dataset contains 183 patients, including only 22 without edema.

This class imbalance can make nuisance-function estimation difficult, particularly for the edema model.

### Predictive Performance vs DML Estimates

The experiment compares DML coefficients, not the predictive accuracy of the models.

A model producing a larger theta or a smaller p-value is not necessarily a better nuisance estimator.

### Bootstrap p-values

The comparison script calculates bootstrap p-values using the proportions of bootstrap estimates above and below zero.

These are sign-based measures with a finite-sample adjustment, not conventional null-centered bootstrap hypothesis tests.

Therefore, the reported p-values should be interpreted with this limitation in mind.

### Causal Interpretation

DML estimation does not automatically establish causality.

A causal interpretation of theta requires appropriate identification assumptions, including adequate control of confounding.

### Paired Comparison Validity

The final comparison script checks the bootstrap sample identities and cross-fitting fold compatibility.

The consolidated results CSV reports paired statistics but does not independently prove that all verification checks passed during execution.

## 14. Final Research Findings

### Finding 1: The optimized MLP produced the largest positive estimate

The optimized MLP achieved a bootstrap mean theta of 0.2000.

Its 95% percentile confidence interval excluded zero.

### Finding 2: The original QNN produced a coefficient close to zero

The original QNN achieved a bootstrap mean theta of 0.0373.

Its confidence interval included zero.

### Finding 3: QNN optimization increased the estimated coefficient

The optimized QNN achieved a bootstrap mean theta of 0.1222 compared with 0.0373 for the original QNN.

The paired mean difference was +0.0849.

However, the corresponding confidence interval included zero.

### Finding 4: The optimized QNN estimate was closer to the MLP estimate

The absolute difference in bootstrap mean theta decreased from:

- Original QNN vs MLP: 0.1627
- Optimized QNN vs MLP: 0.0778

This indicates greater numerical agreement between the optimized QNN and MLP estimates.

It does not establish that the optimized QNN is more accurate.

### Finding 5: None of the reported paired differences excluded zero

All three paired 95% percentile confidence intervals included zero.

The reported comparisons therefore do not establish a nonzero difference at the conventional 5% threshold.

### Finding 6: The optimized QNN required substantially more computation

The optimized QNN recorded 648.97 cumulative hours across 1,000 bootstrap iterations.

The computational cost is an important consideration when comparing classical and quantum approaches on this dataset.

## 15. Conclusion

The three-model comparison shows that the optimized MLP, original QNN, and optimized QNN produce different numerical DML estimates of the association between edema and tumor type.

The optimized MLP produced the largest positive coefficient, while the original QNN produced the smallest.

QNN optimization increased the estimated coefficient and brought it closer to the MLP estimate.

However, the reported paired bootstrap confidence intervals all include zero. The results therefore do not establish statistically significant differences between the three model estimates at the conventional 5% threshold.

The optimized QNN also required substantially greater computational resources.

These findings describe the behavior of the three DML implementations on this dataset and should not be interpreted as proof of quantum advantage or causal identification.

## 16. Project Folder Structure

```text
06_MLP_Optimized_vs_QNN_Original_vs_QNN_Optimized_Comparison/
|
|-- 06_Compare_Three_DML_Models.py
|-- README.md
|-- final_qdml_comparison_results.csv
|
|-- comparaison_3_modeles/
|   |-- comparaison_modeles.csv
|   |-- differences_appariees.csv
|   |-- thetas_apparies.csv
|   |-- hist_thetas.png
|   |-- hist_differences.png
|   |-- forest_ic.png
```

The `comparaison_3_modeles/` contents shown above are the expected outputs of the script. Their presence must be checked in the actual project directory.
