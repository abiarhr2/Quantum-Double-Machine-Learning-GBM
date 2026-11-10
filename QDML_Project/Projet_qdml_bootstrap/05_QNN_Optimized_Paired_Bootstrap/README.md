Optimized QNN Paired Bootstrap
Objective
Evaluate the optimized Quantum Neural Network (QNN) as a nuisance-function estimator in Double Machine Learning (DML), using 1,000 bootstrap iterations. The experiment estimates the association between edema and brain tumor type (GBM vs GSM), quantifies uncertainty, and preserves compatibility with earlier optimized MLP and original QNN bootstrap results.

This experiment trains only the optimized QNN. It does not retrain the MLP or original QNN. The separate comparison script calculates paired model differences.
Dataset
- Input: data/GBM_GBS.xlsx
- Patients: 183 (100 GBM, 83 GSM)
- Features: 12 radiomic variables
- Treatment D: edema
- Outcome Y: tumor type (GSM = 1)
- Cross-fitting: 5 folds
- Bootstrap: 1,000 planned iterations
Optimized QNN configuration
Setting	Value
Qubits	12
Variational layers	3
Trainable parameters	108
Epochs	120
Learning rate	0.05
Optimizer	Adam
Loss	Binary cross-entropy
Simulator	lightning.qubit
Differentiation	Adjoint
Encoding	AngleEmbedding, Y rotations
Circuit	StronglyEntanglingLayers
Measurement	PauliZ(0)
Output probability	(1 - expectation_value) / 2


DML methodology
The QNN estimates two nuisance functions independently:
- g(X) = P(Y = 1 | X): probability of GSM.
- m(X) = P(D = 1 | X): probability of edema.
For each cross-fitting fold, missing values are imputed using the training-fold median. Features are scaled to [-π, π] using training-fold parameters; validation data are transformed using those fitted parameters. The QNN is trained separately for g and m, producing out-of-fold predictions.

The residuals and DML coefficient are:
Y_res = Y - g_hat(X)
D_res = D - m_hat(X)
theta = sum(D_res * Y_res) / sum(D_res ** 2)
The original-data analysis also computes an analytical standard error, confidence interval, and normal-approximation p-value.
Paired bootstrap workflow
For each bootstrap iteration b:
1. Draw patients with replacement using the iteration-specific random seed.
2. Generate five stratified, group-aware cross-fitting folds. Repeated copies of a patient stay in the same fold.
3. Fit fold-specific preprocessing.
4. Train optimized QNNs for g(X) and m(X).
5. Generate out-of-fold predictions and estimate theta and its analytical standard error.
6. Save the coefficient, standard error, sample fingerprint, runtime, and any error.
The script uses the same bootstrap sampling scheme as the earlier MLP/original-QNN experiment. Pairing is checked using the random seed, fold fingerprint, prior results, and bootstrap sample hashes. Its reference fold fingerprint is a56f4caa2b1e. The script rejects incompatible fold configurations rather than silently treating them as paired.
Running the script
Python file: 05_QNN_Optimized_Paired_Bootstrap.py

Check compatibility without training:
python 05_QNN_Optimized_Paired_Bootstrap.py --check
Run the default bootstrap range (iterations 0–999):
python 05_QNN_Optimized_Paired_Bootstrap.py
The script also supports selected iteration ranges, resuming previously saved iterations, and --skip-original to bypass original-data estimation. Existing completed iterations are not unnecessarily retrained.
Input and output files
Inputs:
- data/GBM_GBS.xlsx
- resultats_bootstrap_MLP_Optimized_Not_QNN_Optimized/: previous optimized MLP and original QNN results, read without modification.
Default output directory: resultats_bootstrap_QNN_OPT/
Output	Description
config_QNN_OPT.json	Model settings, bootstrap settings, and fold fingerprint
resultats_originaux_QNN_OPT.csv	Original-data theta, analytical SE, CI, p-value, runtime
boot_START_END.csv	Bootstrap results for the requested range


Bootstrap CSV columns include b, QNN_OPT, QNN_OPT_se, idx_hash, temps_s, and erreur. Failed iterations are recorded rather than counted as valid coefficients.

The script does not create the three-model comparison figures or paired-difference tables. Those are produced separately by compare_qdml_results.py.
Experimental results
The following numerical results are reported in the supplied consolidated file final_qdml_comparison_results.csv. They are not recalculated by this README.
Optimized QNN on original data
Metric	Value
Theta	0.1544
Analytical SE	0.0836
95% analytical CI	[-0.0096, 0.3183]
Analytical p-value	0.0649


The estimated association is positive, but the analytical 95% confidence interval includes zero.
Optimized QNN bootstrap
Metric	Value
Valid iterations	1,000
Mean theta	0.1222
Median theta	0.1277
Bootstrap SE	0.0865
95% percentile CI	[-0.0547, 0.2911]
Reported bootstrap p-value	0.1718


The bootstrap estimate remains positive on average, but its percentile confidence interval includes zero. The reported bootstrap p-value is a sign-based measure, not a conventional null-centered bootstrap test.
Runtime
Metric	Value
Original-data runtime	2,668.47 seconds
Mean bootstrap iteration runtime	2,336.29 seconds
Total recorded bootstrap compute time	648.97 hours


Total compute time is cumulative across iterations; it does not necessarily equal elapsed wall-clock time if iterations ran in parallel.
Comparison with previous models
These values come from the separate final three-model comparison, not from this QNN-only training script.
Model	Bootstrap mean theta	95% percentile CI
Optimized MLP	0.2000	[0.0442, 0.3624]
Original QNN	0.0373	[-0.1121, 0.1868]
Optimized QNN	0.1222	[-0.0547, 0.2911]


Paired difference	Mean difference	95% percentile CI	Reported p-value
Optimized QNN − Original QNN	+0.0849	[-0.0520, 0.2147]	0.2078
Optimized QNN − Optimized MLP	-0.0778	[-0.2434, 0.0802]	0.3936


Optimization increased the QNN's mean DML coefficient from 0.0373 to 0.1222 and moved it numerically closer to the optimized MLP's mean of 0.2000. However, both paired confidence intervals include zero. The reported comparisons do not establish a nonzero paired difference at the conventional 5% threshold.
Limitations
- The sample contains only 183 patients, limiting precision.
- DML coefficients are not predictive accuracy metrics; a larger theta does not prove a better nuisance model.
- Bootstrap sign-based p-values require careful interpretation and are not equivalent to conventional null-centered bootstrap hypothesis tests.
- DML does not establish causality without appropriate identification assumptions.
- The consolidated results file contains paired statistics but does not independently prove that every runtime pairing verification passed.
- Quantum simulation and repeated cross-fitting incur substantial computational cost.
Project structure
05_QNN_Optimized_Paired_Bootstrap/
├── 05_QNN_Optimized_Paired_Bootstrap.py
├── README.md
└── resultats_bootstrap_QNN_OPT/
    ├── config_QNN_OPT.json
    ├── resultats_originaux_QNN_OPT.csv
    └── boot_START_END.csv
The directory tree shows the expected default outputs; their presence depends on the completed run.
Conclusion
The optimized 12-qubit, three-layer QNN produced a bootstrap mean DML coefficient of 0.1222 across 1,000 valid iterations, compared with 0.0373 for the original QNN. The optimized estimate was numerically closer to the optimized MLP estimate (0.2000). Nevertheless, the optimized QNN's 95% bootstrap confidence interval included zero, and the paired comparisons did not establish statistically significant differences at the conventional 5% threshold. These results do not demonstrate quantum advantage.