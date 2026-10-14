"""Where results are written, and which runs are already in them.

Shared by the launchers in the repository root (run_*.py) and the benchmark
runners in scripts/benchmarks/. Kept free of torch so `--check` stays fast.
"""

import csv
import os

CLS_CSV = 'results/cls_results.csv'
FCST_CSV = 'results/fcst_results.csv'
ANO_CSV = 'results/ano_results.csv'        # univariate (KPI, Yahoo)
ANO_MV_CSV = 'results/ano_mv_results.csv'  # multivariate, one row per protocol

CLS_POOLINGS = ('maxpool', 'adaptive_avg')
CLS_PROTOCOLS = ('svm', 'svm_linear')


def default_seeds(n):
    return [42 + i for i in range(n)]


def dim_overrides(hidden_dims=None, output_dims=None):
    """CLI dim overrides as trainer kwargs (only the ones given)."""
    return {k: v for k, v in (('hidden_dims', hidden_dims), ('output_dims', output_dims))
            if v is not None}


def dims_for(method, hidden_dims=None, output_dims=None):
    """(hidden_dims, repr_dims) of a run as recorded in the CSV: override, else default."""
    from baselines import get_method_dims
    default_h, default_r = get_method_dims(method)
    return (default_h if hidden_dims is None else hidden_dims,
            default_r if output_dims is None else output_dims)


def _rows(eval_csv):
    if not os.path.exists(eval_csv):
        return
    with open(eval_csv, newline='') as f:
        yield from csv.DictReader(f)


def _same_run(row, method, seed, hidden_dims, repr_dims):
    # A row without the dims columns (written before they existed) matches any dims.
    return (row.get('method') == method and
            row.get('seed') == str(seed) and
            row.get('hidden_dims', str(hidden_dims)) == str(hidden_dims) and
            row.get('repr_dims', str(repr_dims)) == str(repr_dims))


def cls_done_pairs(eval_csv, method, seed, hidden_dims, repr_dims):
    """{(dataset, epoch): {(pooling, protocol), ...}} already in the CSV for one run."""
    done = {}
    for row in _rows(eval_csv):
        if _same_run(row, method, seed, hidden_dims, repr_dims):
            key = (row.get('dataset'), int(row['epoch']))
            done.setdefault(key, set()).add((row.get('pooling'), row.get('eval_protocol')))
    return done


def cls_completed_datasets(eval_csv, method, seed, checkpoint_epochs, hidden_dims,
                           repr_dims, protocols=CLS_PROTOCOLS):
    """Datasets with every (pooling, protocol) row at every checkpoint epoch."""
    need = {(p, proto) for p in CLS_POOLINGS for proto in protocols}
    done = cls_done_pairs(eval_csv, method, seed, hidden_dims, repr_dims)
    datasets = {ds for ds, _ in done}
    return {ds for ds in datasets
            if all(need <= done.get((ds, ep), set()) for ep in checkpoint_epochs)}


def fcst_done(eval_csv, method, dataset, seed, max_epoch, univar, hidden_dims, repr_dims):
    """True when the last checkpoint of this forecast run is in the CSV."""
    return any(_same_run(row, method, seed, hidden_dims, repr_dims) and
               row.get('dataset') == dataset and
               row.get('epoch') == str(max_epoch) and
               row.get('univar') == str(univar)
               for row in _rows(eval_csv))


def ano_done(eval_csv, method, dataset, seed, max_epoch, scoring, hidden_dims, repr_dims):
    """True when the last checkpoint of this anomaly run is in the CSV.

    `scoring='both'` (multivariate: one training scores with both) needs knn
    and gaussian rows carrying a `protocol` column, i.e. rows written by the
    current multivariate path.
    """
    need = {'knn', 'gaussian'} if scoring == 'both' else {scoring}
    found = {row['scoring'] for row in _rows(eval_csv)
             if (scoring != 'both' or row.get('protocol')) and
             _same_run(row, method, seed, hidden_dims, repr_dims) and
             row.get('dataset') == dataset and
             row.get('epoch') == str(max_epoch) and
             row.get('scoring') in need}
    return found == need
