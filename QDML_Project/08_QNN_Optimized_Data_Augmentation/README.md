# Optimized QNN with fold-local data augmentation

This experiment compares the optimized QNN of experiment 05 with the same QNN trained on real and synthetic data. It estimates the edema effect with the original partially linear DML score. It reads the existing workbook and feature list from the sibling project and writes experiment outputs under this folder.

## Configuration and evaluation

- The original 183 patients and 12 ordered radiomic variables are used. D is 1 for Edema = Y. Y is 1 when the patient name ends in GSM. Both nuisance models use X only and predict binary probabilities, g(X) = P(Y=1|X) and m(X) = P(D=1|X).
- The QNN uses 12 Lightning qubits, Y-angle encoding, three StronglyEntanglingLayers, and Pauli-Z measurement on wire 0. It uses adjoint differentiation, the original seeded weight initialization, and 120 full-batch Adam steps at 0.05. The median imputer and MinMax scaling to [-pi, pi] are fitted on real training rows only. Both nuisance models use the original full training fold at baseline and all real plus synthetic rows when augmented.
- The plan contains 13 configurations. Baseline and BN at 25, 50, 75, 100, 150 and 200 percent are executable. The six SEQ configurations are reserved pending a precise method specification. Counts use the fold's real training rows, including bootstrap multiplicity, rounded to the nearest integer with ties rounded up. Override the ratios with `--ratios` and use a separate `--run` name for a different plan.
- All configurations share five treatment-stratified folds on the original data and the same QNN seed, 42. The selection criterion is lower mean fold outcome log loss than baseline, subject to mean treatment log loss no higher than baseline. Eligible candidates are ordered by outcome log loss, then treatment log loss, then augmentation percentage. Each method falls back to baseline if no ratio qualifies. The code evaluates all available ratios before selection and saves fold metrics and their sample standard deviations. This criterion is fixed before bootstrap.
- Bootstrap uses 1,000 original-style draws with replacement, seed equal to iteration number, and treatment-stratified grouped folds. Copies of a real patient stay together. The original three-iteration fold fingerprint is checked before computation. Results include both pooled held-out log losses, fold metrics, theta, standard error, interval, hashes and times. Synthetic rows never enter the score or its denominator. The baseline is computed once per iteration. A method that falls back to baseline does not run a duplicate baseline fit.

## Generators

**BN.** The fixed conditional-Gaussian network uses the statistical factorization P(D) P(Y|D) P(X|D,Y). It samples the discrete pair from its empirical training-fold joint distribution and samples continuous predictors from group-specific Gaussian means with a pooled covariance. The covariance is estimated in standardized coordinates with 20% shrinkage toward its diagonal. This is equivalent before clipping to a dense ordered linear-Gaussian predictor network with both discrete labels as parents. The implementation uses NumPy. It does not learn a causal graph. The conditional-Gaussian model is described by [Lauritzen (1992)](https://ics.uci.edu/~dechter/courses/ics-275b/spring-05/handouts/lauritzen_first_paper.pdf).

Synthetic feature values are clipped to each real training column's observed range. Diagnostics check finite values, binary labels and bounds, and report real and synthetic joint label counts, duplicate and exact training-row matches, clipped cells, standardized mean shifts and joint correlation shifts. Gaussianity, pooled covariance, sparse treatment/outcome cells and clipping can distort dependencies or yield unrealistic joint profiles. The diagnostic results should be reviewed when interpreting the comparison.

**SEQ.** No reference implementation or publication for the requested named algorithm was found in the project. Implementation needs the reference, variable order, conditional models for binary and continuous nodes, missing-value and sampling rules. The SEQ interface reports that blocker. The BN samples the joint distribution using its own specified network and is not a substitute for SEQ.

## Local checks

Use an existing environment with NumPy, pandas, SciPy, scikit-learn 1.8 or newer, openpyxl and Matplotlib. The tests use a stub for quantum training.

```bash
cd QDML_Project/08_QNN_Optimized_Data_Augmentation
python -B -m unittest discover -s tests -v
```

## Transfer and SLURM

From the local repository root in PowerShell, set the DICO SSH login and the absolute path to the existing remote QDML_Project. The sibling workbook and experiment 05 code must already exist there.

The loader accepts the nested checkout layout and a flat bootstrap directory with
`05_QNN_Optimized_Paired_Bootstrap/` and `data/` alongside experiment 08.
If the original files are elsewhere, export their absolute file paths before
running the tests or submitting jobs.

```bash
export QNN_AUG_ORIGINAL=/path/to/05_QNN_Optimized_Paired_Bootstrap.py
export QNN_AUG_WORKBOOK=/path/to/GBM_GBS.xlsx
```

The historical CSV check is skipped if the optional archive is absent. The
reference draw and fold fingerprints are checked independently.

```powershell
$dicoLogin = 'YOUR_USER@YOUR_DICO_SSH_HOST'
$dicoProject = '/absolute/path/to/QDML_Project'
scp -r .\QDML_Project\08_QNN_Optimized_Data_Augmentation "${dicoLogin}:${dicoProject}/"
```

On DICO, activate the experiment environment and check that its installed versions match the intended production environment. Use the same environment for all phases. Run these commands from the transferred folder.

```bash
source "$HOME/venv_qdml/bin/activate"
python -B -m unittest discover -s tests -v
mkdir -p logs
selection=$(sbatch --parsable job_select_augmentation.sh)
bootstrap=$(sbatch --parsable --dependency="afterok:${selection%%;*}" job_bootstrap_augmentation.sh)
sbatch --dependency="afterok:${bootstrap%%;*}" job_compare_augmentation.sh
```

The bootstrap array covers iterations 0–999 with at most eight simultaneous tasks. Run name arguments must match across phases. To resume, submit the same bootstrap script with the same run name after the prior jobs are inactive. The selection and bootstrap checkpoints are immutable. An incompatible or incomplete checkpoint stops the worker. On Linux, process-owned POSIX locks require shared-filesystem support for cross-node locking. Lock files can remain after interruptions. Completed JSON records are published atomically without replacing any existing record.

## Interpretation and optional decision model

Comparison exports a validation table, paired bootstrap estimates, percentile intervals, paired differences, and validation and bootstrap figures. The SEQ curves and estimates are unavailable pending its algorithm specification. The winning validation score may be optimistic because the same folds rank the candidates. The bootstrap holds the selected settings fixed, so its intervals omit variability from reselection. Predictive improvement and smaller estimator variability do not establish causal accuracy or identification. Interpretation of the edema effect still depends on the original causal assumptions, including overlap and adequate control for confounding.

A logistic-regression decision model would require multiple independent datasets or experiments with known augmentation outcomes and external validation. Candidate inputs would include cohort size, baseline log losses, class imbalance and predictor count. This project supplies no such training set, so no decision model is fitted.
