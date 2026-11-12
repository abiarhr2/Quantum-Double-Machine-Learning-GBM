import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import augmentation_bn
from augmentation_seq import generate as seq
import bootstrap_qnn_augmented
from bootstrap_qnn_augmented import evaluate_bootstrap, iteration_keys, validate_result
from config import ORIGINAL_DIR, ORIGINAL_NAME, PROJECT, original_source, planned_cases, synthetic_count, workbook_path
from data_utils import boot_indices, check_reference_folds, digest, load_data, splits
import qnn_dml
import select_augmentation
from select_augmentation import choose
from storage import cached, exclusive, publish


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.Y, cls.D, _ = load_data()

    def test_original_cohort_and_definitions(self):
        self.assertEqual(self.X.shape, (183, 12))
        self.assertEqual(np.bincount(self.Y).tolist(), [100, 83])
        self.assertEqual(np.bincount(self.D).tolist(), [22, 161])

    def test_checkout_and_dico_input_paths(self):
        for subdirectory in ('Projet_qdml_bootstrap', ''):
            with (self.subTest(subdirectory=subdirectory), TemporaryDirectory() as tmp,
                  patch.dict(os.environ, {'QNN_AUG_ORIGINAL': '', 'QNN_AUG_WORKBOOK': ''})):
                root = Path(tmp)
                source = root / subdirectory / ORIGINAL_DIR / ORIGINAL_NAME
                workbook = root / subdirectory / 'data' / 'GBM_GBS.xlsx'
                for path in (source, workbook):
                    path.parent.mkdir(parents=True)
                    path.touch()
                self.assertEqual(original_source(root), source.resolve())
                self.assertEqual(workbook_path(root), workbook.resolve())
                with patch.dict(os.environ, {'QNN_AUG_ORIGINAL': str(workbook)}):
                    self.assertEqual(original_source(root), workbook.resolve())

    def test_plan_and_fold_relative_rounding(self):
        self.assertEqual(len(planned_cases()), 13)
        self.assertEqual([synthetic_count(146, p) for p in (0, 25, 50, 75, 100, 150, 200)],
                         [0, 37, 73, 110, 146, 219, 292])
        with self.assertRaises(NotImplementedError):
            seq()

    def test_bn_deterministic_and_valid(self):
        X, Y, D = self.X[:140], self.Y[:140], self.D[:140]
        sample = augmentation_bn.generate(X, Y, D, 35, [42, 0, 1])
        second = augmentation_bn.generate(X, Y, D, 35, [42, 0, 1])
        for left, right in zip(sample, second):
            np.testing.assert_array_equal(left, right)
        audit = augmentation_bn.diagnostics(X, Y, D, *sample)
        self.assertEqual(sum(audit['synthetic_joint_counts']), 35)
        self.assertTrue(np.isfinite(audit['mean_absolute_joint_correlation_shift']))

    def test_fold_training_isolated_and_predictors_have_no_labels(self):
        tr, te = splits(self.X, self.D, 42)[0]
        Xtr, Ytr, Dtr = self.X[tr], self.Y[tr], self.D[tr]
        Xte = self.X[te] + 10_000
        n = synthetic_count(len(tr), 25)
        seen = []

        def generate(real, y, d, count, seed):
            np.testing.assert_array_equal(real, Xtr)
            np.testing.assert_array_equal(y, Ytr)
            np.testing.assert_array_equal(d, Dtr)
            self.assertEqual((count, seed), (n, [42, 0, 1]))
            return real[:n], np.ones(n, dtype=int), np.zeros(n, dtype=int), 0.0

        def fit(train, target, test, seed):
            self.assertEqual(train.shape, (len(tr) + n, 12))
            self.assertEqual(test.shape, (len(te), 12))
            self.assertEqual(seed, 42)
            self.assertTrue((test > np.pi).all())
            seen.append(target.copy())
            return np.full(len(te), 0.5)

        with patch.object(qnn_dml, 'generate', generate), patch.object(qnn_dml, 'quantum_nn_fit_predict', fit):
            g, m, info = qnn_dml.fit_fold(Xtr, Ytr, Dtr, Xte, 'BN', 25, 42, 0)
        self.assertEqual(info['n_synthetic'], n)
        self.assertEqual(len(g), len(te))
        self.assertEqual(len(m), len(te))
        np.testing.assert_array_equal(seen[0], np.r_[Ytr, np.ones(n)])
        np.testing.assert_array_equal(seen[1], np.r_[Dtr, np.zeros(n)])

    def test_bootstrap_indices_folds_and_historical_reference(self):
        check_reference_folds(self.X, self.D)
        hashes = {0: '50869cc64b3d', 1: '3192b7cf38fa', 17: 'f2c532e990bf',
                  100: 'f991475b002e', 349: '69bdc97bf502', 999: '42ab1dbc7278'}
        for b, expected in hashes.items():
            idx, folds, keys = iteration_keys(b, self.X, self.D, {}, planned_cases())
            self.assertEqual(len(folds), 5)
            self.assertEqual({k['idx_hash'] for k in keys}, {expected})
            self.assertEqual(len({k['fold_hash'] for k in keys}), 1)
            for tr, te in folds:
                self.assertFalse(np.intersect1d(idx[tr], idx[te]).size)

    def test_historical_csv_hashes_when_available(self):
        paths = {folder / 'resultats_bootstrap_QNN_OPT' for folder in
                 (original_source().parent, PROJECT, PROJECT / 'Projet_qdml_bootstrap')}
        files = sorted({file for path in paths for file in path.glob('boot_*.csv')})
        if not files:
            self.skipTest('Historical CSV archive is not installed')
        for path in files:
            with path.open(encoding='utf-8-sig', newline='') as handle:
                for row in csv.DictReader(handle):
                    self.assertEqual(digest(boot_indices(int(row['b']), len(self.D))), row['idx_hash'])

    def test_logloss_and_bootstrap_use_real_labels_only(self):
        probability = np.array([0.25, 0.75])
        metrics = qnn_dml.losses([0, 1], [1, 0], probability, probability)
        self.assertAlmostEqual(metrics['g_logloss'], -np.log(0.75))
        self.assertAlmostEqual(metrics['m_logloss'], -np.log(0.25))

        def fake_fold(Xtr, Ytr, Dtr, Xte, method, percent, seed, fold):
            return (np.full(len(Xte), 0.5), np.full(len(Xte), 0.5),
                    dict(n_synthetic=synthetic_count(len(Xtr), percent),
                         synthetic_diagnostics={}, training_s=0, elapsed_s=0))

        idx, folds, _ = iteration_keys(17, self.X, self.D, {}, [('BN', 25)])
        Y, D = self.Y[idx], self.D[idx]
        with patch('bootstrap_qnn_augmented.fit_fold', fake_fold):
            record = evaluate_bootstrap(self.X[idx], Y, D, folds, 'BN', 25, 17)
        validate_result(record, True)
        expected = np.dot(Y - 0.5, D - 0.5) / np.sum((D - 0.5) ** 2)
        self.assertAlmostEqual(record['theta'], expected)
        self.assertEqual(sum(r['n_real_test'] for r in record['folds']), 183)

    def test_selection_guard_and_baseline_fallback(self):
        def row(method, p, g, m):
            return dict(method=method, percent=p, g_logloss=g, m_logloss=m)
        rows = [row('baseline', 0, .6, .4), row('BN', 25, .51, .401),
                row('BN', 50, .55, .40), row('BN', 75, .59, .39)]
        self.assertEqual(choose(rows, 'BN'), 50)
        self.assertIsNone(choose(rows[:2], 'BN'))
        self.assertIsNone(choose(rows, 'SEQ'))

    def test_safe_resume_and_immutable_publication(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'record.json'
            lock = path.with_suffix('.lock')
            lock.touch()
            key = {'b': 0}
            validate = lambda result: result['theta']
            result = cached(path, key, lambda: {'theta': 0.1}, validate)
            self.assertEqual(result, {'theta': 0.1})
            before = path.read_bytes()
            with patch('storage.publish', side_effect=AssertionError('Recomputed')):
                self.assertEqual(cached(path, key, None, validate), result)
            with self.assertRaises(FileExistsError):
                publish(path, {})
            with self.assertRaises(ValueError):
                cached(path, {'b': 1}, None, validate)
            self.assertEqual(path.read_bytes(), before)

    def test_process_death_releases_lock(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            lock = directory / 'test.lock'
            script = ("from pathlib import Path; import sys, time; from storage import exclusive\n"
                      "with exclusive(Path(sys.argv[1])):\n"
                      "    print('locked', flush=True); time.sleep(30)\n")
            process = subprocess.Popen([sys.executable, '-B', '-c', script, str(lock)],
                                       stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), 'locked')
            finally:
                process.kill()
                process.wait(timeout=5)
                process.stdout.close()
            self.assertTrue(lock.exists())
            with exclusive(lock):
                self.assertTrue(lock.exists())

    def test_selection_to_paired_report_without_quantum_training(self):
        import compare_augmentation

        def fake_fold(Xtr, Ytr, Dtr, Xte, method, percent, seed, fold):
            g = np.full(len(Xte), 0.5)
            return g, g.copy(), dict(n_synthetic=synthetic_count(len(Xtr), percent),
                                     synthetic_diagnostics={} if percent else None,
                                     training_s=0, elapsed_s=0)

        def selection_losses(y, d, g, m):
            # Controlled scores give the pipeline fixture a known winner.
            return dict(g_logloss=float(g[0]), m_logloss=0.5)

        def selection_fold(Xtr, Ytr, Dtr, Xte, method, percent, seed, fold):
            g, m, timing = fake_fold(Xtr, Ytr, Dtr, Xte, method, percent, seed, fold)
            if method == 'BN':
                g.fill(0.4)
            return g, m, timing

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            for module, args in [(select_augmentation, ['--ratios', '25']),
                                 (bootstrap_qnn_augmented, ['--end', '2']),
                                 (compare_augmentation, ['--iterations', '2'])]:
                with (patch.object(module, 'run_directory', return_value=out),
                      patch.object(module, 'experiment_identity', return_value={'test': True}),
                      patch.object(sys, 'argv', ['test', *args]),
                      patch.object(select_augmentation, 'fit_fold', selection_fold),
                      patch.object(select_augmentation, 'losses', selection_losses),
                      patch.object(bootstrap_qnn_augmented, 'fit_fold', fake_fold),
                      patch('builtins.print')):
                    module.main()
            selected = json.loads((out / 'selection.json').read_text(encoding='utf-8'))
            self.assertEqual(selected['result']['selected'], {'SEQ': None, 'BN': 25})
            comparison = out / 'comparison_2'
            self.assertEqual(len(list(comparison.glob('*.png'))), 4)
            with (comparison / 'summary.csv').open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row['method'] for row in rows], ['baseline', 'BN'])
            self.assertAlmostEqual(float(rows[1]['paired_delta']), 0)


if __name__ == '__main__':
    unittest.main()
