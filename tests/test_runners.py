"""Runner plumbing and the encoding / evaluation helpers that need no training."""
import csv
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.result_status import dim_overrides, dims_for
from utils.runner import FailureLog, append_csv_rows, iter_work, parse_epochs


def _load_runner(name):
    path = os.path.join(ROOT, 'scripts', 'benchmarks', f'{name}.py')
    spec = importlib.util.spec_from_file_location(f'_runner_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTrainer:
    """encode(raw=True) -> per-timestep features [x, 2x] (or a pooled / shortened variant).

    Like the real single-encoder trainers it has no `use_target` argument;
    FakeDualTrainer has one and marks target outputs with a sign flip.
    """

    def __init__(self, mode='pointwise'):
        self.mode = mode
        self.calls = []

    def encode(self, data, raw=False):
        return self._features(data, raw)

    def _features(self, data, raw):
        self.calls.append((data.shape, raw))
        z = np.concatenate([data, 2 * data], axis=-1)       # (1, T, 2C)
        if self.mode == 'pooled':
            return z.mean(axis=1)                            # (1, 2C)
        if self.mode == 'half':
            return z[:, ::2]                                 # patch-like T' = T/2
        return z


class FakeDualTrainer(FakeTrainer):
    target_encoder = object()

    def encode(self, data, raw=False, use_target=False):
        z = self._features(data, raw)
        return -z if use_target else z


class TestEncodeSequence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ano = _load_runner('run_anomaly')

    def test_chunked_equals_direct_for_pointwise_encoder(self):
        x = np.random.RandomState(0).randn(12000).astype(np.float32)
        tr = FakeTrainer()
        z = self.ano.encode_sequence(tr, x)
        self.assertEqual(z.shape, (12000, 2))
        np.testing.assert_allclose(z[:, 0], x)
        self.assertGreater(len(tr.calls), 1)                 # really chunked

    def test_short_series_padding_is_stripped(self):
        x = np.arange(50, dtype=np.float32)
        z = self.ano.encode_sequence(FakeTrainer(), x)
        np.testing.assert_allclose(z[:, 0], x)

    def test_multivariate_input(self):
        x = np.random.RandomState(1).randn(300, 3).astype(np.float32)
        z = self.ano.encode_sequence(FakeTrainer(), x)
        np.testing.assert_allclose(z[:, :3], x)

    def test_shortened_output_is_interpolated_to_input_length(self):
        z = self.ano.encode_sequence(FakeTrainer('half'), np.zeros(400, np.float32))
        self.assertEqual(z.shape, (400, 2))

    def test_pooled_method_returns_vector(self):
        z = self.ano.encode_sequence(FakeTrainer('pooled'), np.zeros(9000, np.float32))
        self.assertEqual(z.ndim, 1)

    def test_use_target_reaches_the_target_encoder(self):
        x = np.arange(300, dtype=np.float32)
        z = self.ano.encode_sequence(FakeDualTrainer(), x, use_target=True)
        np.testing.assert_allclose(z[:, 0], -x)

    def test_row_labels_put_target_under_plain_name(self):
        dual = FakeDualTrainer()
        # target = plain name (the encoder the paper reports), online = *_online
        self.assertEqual(self.ano._row_labels('cf_jepa', dual), ('cf_jepa_online', 'cf_jepa'))
        self.assertEqual(self.ano._row_labels('ts2vec', FakeTrainer()), ('ts2vec', None))


class TestClassificationHeads(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cls = _load_runner('run_classification')

    def test_svm_kernel_choice(self):
        rng = np.random.RandomState(0)
        X = np.vstack([rng.randn(20, 4) - 2, rng.randn(20, 4) + 2])   # < 50: no grid search
        y = np.array([0] * 20 + [1] * 20)
        self.assertEqual(self.cls.fit_svm(X, y).kernel, 'rbf')
        self.assertEqual(self.cls.fit_svm(X, y, kernel='linear').kernel, 'linear')
        acc = self.cls.evaluate_classifier(X, y, X, y, 'svm_linear')
        self.assertEqual(acc, 1.0)

    def test_pooling_shapes(self):
        z = np.random.RandomState(0).randn(5, 40, 3).astype(np.float32)
        self.assertEqual(self.cls._apply_pooling(z, 'maxpool').shape, (5, 3))
        self.assertEqual(self.cls._apply_pooling(z, 'adaptive_avg').shape, (5, 8 * 3))
        np.testing.assert_allclose(self.cls._apply_pooling(z, 'maxpool'), z.max(axis=1))


class TestRunnerPlumbing(unittest.TestCase):

    def test_iter_work_single_process_yields_all_in_order(self):
        self.assertEqual(list(iter_work(['a', 'b'], 0, 1, 'unused')), [(0, 'a'), (1, 'b')])

    def test_failure_log_exits_nonzero_after_running_the_rest(self):
        log = FailureLog()
        ran = []

        def boom():
            raise RuntimeError('boom')

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as cm:
            log.run('first', boom)
            log.run('second', ran.append, 'second ran')
            log.finish()
        self.assertEqual(cm.exception.code, 1)
        self.assertEqual(ran, ['second ran'])
        self.assertEqual(log.labels, ['first'])

    def test_failure_log_clean_run_does_not_exit(self):
        log = FailureLog()
        log.run('ok', lambda: None)
        log.finish()                                          # no SystemExit

    def test_append_csv_rows_writes_header_once(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'sub', 'out.csv')
            append_csv_rows(path, ['a', 'b'], [{'a': 1, 'b': 2}])
            append_csv_rows(path, ['a', 'b'], [{'a': 3, 'b': 4}])
            append_csv_rows(path, ['a', 'b'], [])
            with open(path, newline='') as f:
                self.assertEqual(list(csv.reader(f)), [['a', 'b'], ['1', '2'], ['3', '4']])

    def test_dims_prefer_override(self):
        self.assertEqual(dims_for('cf_jepa'), (256, 320))
        self.assertEqual(dims_for('cf_jepa', 64), (64, 320))
        self.assertEqual(dim_overrides(None, 128), {'output_dims': 128})

    def test_parse_epochs_sorts_and_dedupes(self):
        self.assertEqual(parse_epochs('200,50, 100,50'), [50, 100, 200])


if __name__ == '__main__':
    unittest.main()
