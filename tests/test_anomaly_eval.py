import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.anomaly_eval import (
    eval_ad_result, eval_mv_protocols, eval_pa_f1, get_range_proba,
    score_gaussian, score_knn, suppress_within_delay, threshold_train4sigma,
)


class TestDelayRules(unittest.TestCase):

    def test_suppress_within_delay(self):
        pred = np.array([1, 1, 1, 0, 0, 0, 1, 1])
        out = suppress_within_delay(pred, delay=3)
        # the first `delay` steps are never suppressed (TS2Vec's rule, kept as is)
        np.testing.assert_array_equal(out, [1, 1, 1, 0, 0, 0, 1, 0])
        np.testing.assert_array_equal(pred, [1, 1, 1, 0, 0, 0, 1, 1])  # input untouched

    def test_range_counts_only_detections_within_delay(self):
        label = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 0])
        # first segment hit at its 2nd step (within delay 1), second hit too late
        pred = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0])
        out = get_range_proba(pred, label, delay=1)
        np.testing.assert_array_equal(out, [0, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    def test_eval_ad_result_fills_timestamp_gaps(self):
        ts = np.array([0, 1, 2, 4, 5])          # timestamp 3 missing
        label = np.array([0, 1, 1, 0, 0])
        pred = np.array([0, 1, 0, 0, 0])
        m = eval_ad_result([pred], [label], [ts], delay=7)
        self.assertEqual(m, {'f1': 1.0, 'precision': 1.0, 'recall': 1.0})

    def test_no_prediction_scores_zero(self):
        m = eval_ad_result([np.zeros(4, int)], [np.array([0, 1, 1, 0])],
                           [np.arange(4)], delay=7)
        self.assertEqual(m['f1'], 0.0)


class TestPointAdjust(unittest.TestCase):

    def test_one_hit_marks_whole_segment(self):
        label = np.array([0, 1, 1, 1, 0, 1, 1])
        pred = np.array([0, 0, 1, 0, 0, 0, 0])
        m = eval_pa_f1(pred, label)
        self.assertAlmostEqual(m['precision'], 1.0)
        self.assertAlmostEqual(m['recall'], 3 / 5)


class TestThresholdsAndProtocols(unittest.TestCase):

    def test_train4sigma(self):
        scores = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(threshold_train4sigma(scores), 2.0 + 4 * np.std(scores))

    def test_mv_protocol_rows(self):
        rng = np.random.RandomState(0)
        train = rng.rand(500)
        test = rng.rand(300)
        test[100:110] += 10
        label = np.zeros(300, int)
        label[100:110] = 1
        rows = eval_mv_protocols(train, test, label, anomaly_ratio=0.02, delay=7,
                                 extra_delays=(3, 7))
        names = [(p, d) for p, d, _ in rows]
        self.assertEqual(names, [
            ('delay_train4sigma', 7), ('delay_ratio', 7), ('point_train4sigma', 7),
            ('point_ratio', 7), ('pa_ratio', 7),
            ('delay_train4sigma', 3), ('delay_ratio', 3),   # extra delay 7 = primary, skipped
        ])
        self.assertEqual(dict((p, m['f1']) for p, d, m in rows if d == 7)['delay_train4sigma'], 1.0)


class TestScoring(unittest.TestCase):

    def test_outlier_scores_highest(self):
        rng = np.random.RandomState(0)
        train = rng.randn(200, 4)
        test = np.vstack([rng.randn(20, 4), np.full((1, 4), 8.0)])
        for fn in (score_knn, score_gaussian):
            train_scores, test_scores = fn(train, test)
            self.assertEqual(train_scores.shape, (200,))
            self.assertEqual(int(np.argmax(test_scores)), 20, fn.__name__)

    def test_pooled_representation_gives_placeholder(self):
        for fn in (score_knn, score_gaussian):
            tr, te = fn(np.zeros(8), np.zeros(8))
            self.assertEqual(tr.shape, (1,))


if __name__ == '__main__':
    unittest.main()
