import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.result_status import (
    ano_done, cls_completed_datasets, cls_done_pairs, default_seeds, fcst_done,
)


def _write(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class TestResultStatus(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv = os.path.join(self.tmp.name, 'r.csv')

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_seeds(self):
        self.assertEqual(default_seeds(3), [42, 43, 44])

    def test_missing_csv_means_nothing_done(self):
        self.assertFalse(fcst_done(self.csv, 'm', 'd', 42, 200, False, 64, 320))
        self.assertEqual(cls_completed_datasets(self.csv, 'm', 42, [50], 64, 320), set())

    def test_cls_needs_every_pooling_protocol_pair_at_every_checkpoint(self):
        base = dict(method='m', seed='42', hidden_dims='64', repr_dims='320')
        pairs = [(p, proto) for p in ('maxpool', 'adaptive_avg') for proto in ('svm', 'svm_linear')]

        def rows_for(ds, ep, use=pairs, **kw):
            return [dict(base, dataset=ds, epoch=str(ep), pooling=p, eval_protocol=proto, **kw)
                    for p, proto in use]

        rows = rows_for('A', 50) + rows_for('A', 100)
        rows += rows_for('B', 50) + rows_for('B', 100, use=pairs[:3])
        rows += rows_for('C', 50, hidden_dims='128') + rows_for('C', 100, hidden_dims='128')
        # four rows, but svm twice and no svm_linear: not complete for 'all'
        rows += rows_for('D', 50, use=[pairs[0], pairs[0], pairs[2], pairs[2]])
        rows += rows_for('D', 100, use=[pairs[0], pairs[0], pairs[2], pairs[2]])
        _write(self.csv, rows)
        self.assertEqual(cls_completed_datasets(self.csv, 'm', 42, [50, 100], 64, 320), {'A'})
        self.assertEqual(cls_completed_datasets(self.csv, 'm', 42, [50, 100], 64, 320,
                                                protocols=('svm',)), {'A', 'B', 'D'})
        done = cls_done_pairs(self.csv, 'm', 42, 64, 320)
        self.assertEqual(done[('B', 100)], set(pairs[:3]))

    def test_fcst_matches_mode_and_last_epoch(self):
        _write(self.csv, [dict(method='m', dataset='ETTh1', seed='42', epoch='200',
                               hidden_dims='64', repr_dims='320', univar='True')])
        self.assertTrue(fcst_done(self.csv, 'm', 'ETTh1', 42, 200, True, 64, 320))
        self.assertFalse(fcst_done(self.csv, 'm', 'ETTh1', 42, 200, False, 64, 320))
        self.assertFalse(fcst_done(self.csv, 'm', 'ETTh1', 42, 400, True, 64, 320))

    def test_rows_without_dims_columns_match_any_dims(self):
        _write(self.csv, [dict(method='m', dataset='kpi', seed='42', epoch='200', scoring='knn')])
        self.assertTrue(ano_done(self.csv, 'm', 'kpi', 42, 200, 'knn', 256, 320))

    def test_ano_both_needs_protocol_rows_for_both_scorings(self):
        base = dict(method='m', dataset='MSL', seed='42', epoch='200',
                    hidden_dims='64', repr_dims='320')
        _write(self.csv, [dict(base, scoring='knn', protocol='delay_train4sigma'),
                          dict(base, scoring='gaussian', protocol='')])
        self.assertTrue(ano_done(self.csv, 'm', 'MSL', 42, 200, 'knn', 64, 320))
        # the gaussian row predates the protocol column, so 'both' is not done
        self.assertFalse(ano_done(self.csv, 'm', 'MSL', 42, 200, 'both', 64, 320))


if __name__ == '__main__':
    unittest.main()
