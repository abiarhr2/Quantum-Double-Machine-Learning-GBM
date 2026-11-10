QNN + PCA: Configuration Selection and DML Bootstrap
Goal
Evaluate whether Principal Component Analysis (PCA) can reduce the number of input features and qubits while retaining useful predictive performance for the QNN nuisance models in Double Machine Learning (DML).

The existing optimized QNN (without PCA) is the reference: 12 qubits, 3 layers, 120 epochs, learning rate 0.05. PCA changes the representation of the input data, even when all 12 components are retained.
Experiment workflow
1. Load the patient dataset using load_data() from common.py.
2. Select a QNN + PCA configuration using the configuration-selection script.
3. Save the ranking and selected configuration.
4. Run the bootstrap script using that frozen configuration.
5. Compare results using compare_qnn_pca.py.
The scripts perform configuration selection followed by bootstrap evaluation. The bootstrap does not retune the hyperparameters on each iteration.
Files and responsibilities
File	Responsibility
common.py	Shared imports, paths, dataset loading, evaluation, resampling, validity checks, identity checks, and atomic output helpers. PCA placement and DML implementation must be verified in this file.
Configuration-selection script (exact filename not provided)	Tests PCA components and QNN layers, saves per-configuration evaluations, ranks results, and writes selected.json.
Bootstrap script (exact filename not provided)	Loads selected.json, records experiment identity, evaluates original data, runs up to 1,000 bootstrap iterations, and saves each result.
compare_qnn_pca.py	Automatically invoked after all 1,000 iteration files exist. Its exact comparison and inference logic has not yet been reviewed.
README.md	Describes the workflow, execution, outputs, and interpretation.


Current configuration search
The selection script evaluates nine configurations:
- PCA components/qubits: 4, 6, 8
- QNN layers: 1, 2, 3
- Epochs: 120 by default
- Learning rate: 0.05 (fixed)
- Seeds: 42 by default; additional seeds can be passed with --seeds
For each configuration and seed, the script calls evaluate(X, Y, D, cfg, seed) from common.py and saves a JSON result.

The ranking is calculated as:
loss = (g_logloss + m_logloss) / 2
score = mean(loss across seeds) + standard_deviation(loss across seeds)
Configurations are sorted by score, then runtime, component count, and layer count. With only one seed, the script replaces undefined standard deviation with zero; stability cannot be assessed from one seed.

Design limitation: This search varies both PCA dimensionality and circuit depth. It does not independently isolate PCA's effect on the previously optimized three-layer QNN.
Recommended staged design
Stage A: Isolate PCA. Hold the previously optimized QNN hyperparameters fixed (3 layers, 120 epochs, learning rate 0.05). Test 4, 6, 8, and 12 PCA components. Compare against the original 12-feature QNN without PCA.

Stage B: Retune PCA-QNN. After identifying promising PCA dimensions, test alternative layer counts and, if computationally feasible, learning rates and epoch budgets.

Stage A and Stage B describe the recommended future experimental design; they are not what the current nine-configuration script does.
Bootstrap evaluation
The bootstrap script:
1. Loads the chosen configuration from results/selection/selected.json by default.
2. Creates or verifies results/bootstrap/identity.json to prevent incompatible restarts.
3. Evaluates the selected model on the original dataset and saves original.json, unless skipped or already present.
4. For each iteration b, draws a patient sample using indices(b, len(Y)).
5. Checks sample validity using valid(idx, D).
6. Calls evaluate(X[idx], Y[idx], D[idx], cfg, seed=b, groups=idx) for valid samples.
7. Writes each iteration atomically to iteration_0000.json through iteration_0999.json.
8. Launches compare_qnn_pca.py after all 1,000 iteration files exist.
Each iteration has an exclusive .lock file to reduce duplicate work across parallel jobs. A lock left after a crashed job may require manual investigation before restarting that iteration.

Important: The existence of 1,000 files does not mean 1,000 valid estimates. Validity and numerical failures must be checked in the saved results.
Running the experiment
Replace the two placeholders below with the actual filenames in your project.
# 1. Configuration search
python <selection_script.py> --seeds 42

# 2. Bootstrap: first 50 iterations
python <bootstrap_script.py> --start 0 --end 50

# 3. Bootstrap: next 50 iterations, without repeating original-data estimation
python <bootstrap_script.py> --start 50 --end 100 --skip-original

# 4. Full bootstrap range
python <bootstrap_script.py> --start 0 --end 1000
Selection supports --epochs, --seeds, and --out-dir. Bootstrap supports --start, --end, --skip-original, --config, and --out-dir.
Output structure
results/
├── selection/
│   ├── c4_l1_s42.json             # one JSON per configuration/seed
│   ├── ...
│   ├── configurations.csv
│   ├── ranking.csv
│   └── selected.json
└── bootstrap/
    ├── identity.json
    ├── original.json
    ├── iteration_0000.json
    ├── ...
    └── iteration_0999.json
Comparison outputs depend on compare_qnn_pca.py and have not been verified.
Scientific checks required before reporting results
- Confirm that common.py fits imputation, standardization, PCA, and quantum input scaling only on each training fold.
- Confirm that the DML estimator uses held-out real-patient predictions and matches the earlier MLP/QNN estimator.
- Confirm that bootstrap samples, fold assignments, labels, and feature definitions match the models used for paired comparisons.
- Verify that model selection and final performance assessment are separated sufficiently to limit selection bias.
- Compare nuisance prediction performance for both g(X) and m(X); do not choose a configuration solely because its DML estimate is significant.
- Compare the selected PCA-QNN against the previously optimized QNN without PCA.
- Record valid iteration counts, numerical failures, runtime, and reproducibility information.
Interpretation
A lower nuisance log-loss is evidence of improved nuisance prediction under the evaluation procedure used; it does not automatically imply a better causal estimate or quantum advantage. The meaning of the reported DML coefficient and the validity of confidence intervals depend on the statistical assumptions and implementations in common.py and compare_qnn_pca.py.

Status: The selection and bootstrap orchestration have been reviewed from the two supplied scripts. The shared implementation and comparison script have not yet been inspected, so complete methodological validity is not established.