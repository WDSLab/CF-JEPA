import os
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from utils.helpers import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler
from config import UCR_PATH, UEA_PATH, FORECAST_PATH, ANOMALY_PATH

def load_UCR(dataset):
    train_file = os.path.join(UCR_PATH, dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join(UCR_PATH, dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    # Open with explicit UTF-8 encoding to avoid Windows default-codec errors
    with open(os.path.join(UEA_PATH, dataset, f'{dataset}_TRAIN.arff'), 'r', encoding='utf-8') as f:
        train_data = loadarff(f)[0]
    with open(os.path.join(UEA_PATH, dataset, f'{dataset}_TEST.arff'), 'r', encoding='utf-8') as f:
        test_data = loadarff(f)[0]

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    
    
def _get_time_features(dt):
    """Extract time features (compatible across recent pandas/numpy versions)."""
    try:
        # Recent pandas: use isocalendar().week instead of weekofyear
        week_feature = dt.isocalendar().week.to_numpy()
    except AttributeError:
        # Older pandas: weekofyear
        week_feature = dt.weekofyear.to_numpy()
    
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        week_feature,
    ], axis=1).astype(np.float64)


def load_forecast_csv(name, univar=False):
    data = pd.read_csv(os.path.join(FORECAST_PATH, f'{name}.csv'), index_col='date', parse_dates=True)
    n_covariate_cols = 0

    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if name in ('ETTh1', 'ETTh2'):
        pred_lens = [24, 48, 168, 336, 720]
    elif name in ('ETTm1', 'ETTm2'):
        pred_lens = [24, 48, 96, 288, 672]
    else:
        pred_lens = [96, 192, 336, 720]

    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(os.path.join(ANOMALY_PATH, f'{name}.pkl'))
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data


# =============================================================================
# Multivariate anomaly (SMD, MSL, SMAP, PSM) — Anomaly-Transformer preproc
# =============================================================================

ANOMALY_MV_DATASETS = {'SMD', 'MSL', 'SMAP', 'PSM'}


def load_anomaly_mv(name):
    """Load a multivariate anomaly benchmark in Anomaly-Transformer layout.

    Returns:
        train_x:    (T_train, D) float array — normal-only training sequence
        test_x:     (T_test, D)  float array — test sequence with anomalies
        test_label: (T_test,)    binary int array — 1 = anomaly timestep

    Expected files under ANOMALY_PATH/{name}/:
        SMD:  SMD_train.npy / SMD_test.npy / SMD_test_label.npy
        MSL:  MSL_train.npy / MSL_test.npy / MSL_test_label.npy
        SMAP: SMAP_train.npy / SMAP_test.npy / SMAP_test_label.npy
        PSM:  train.csv / test.csv / test_label.csv
    """
    folder = os.path.join(ANOMALY_PATH, name)

    if name == 'PSM':
        train_df = pd.read_csv(os.path.join(folder, 'train.csv'))
        test_df = pd.read_csv(os.path.join(folder, 'test.csv'))
        label_df = pd.read_csv(os.path.join(folder, 'test_label.csv'))
        for df in (train_df, test_df, label_df):
            if 'timestamp_(min)' in df.columns:
                df.drop(columns=['timestamp_(min)'], inplace=True)
        train_x = np.nan_to_num(train_df.to_numpy().astype(np.float32))
        test_x = np.nan_to_num(test_df.to_numpy().astype(np.float32))
        test_label = label_df['label'].to_numpy().astype(np.int64) \
            if 'label' in label_df.columns else label_df.to_numpy().squeeze().astype(np.int64)
    else:
        train_x = np.load(os.path.join(folder, f'{name}_train.npy')).astype(np.float32)
        test_x = np.load(os.path.join(folder, f'{name}_test.npy')).astype(np.float32)
        test_label = np.load(os.path.join(folder, f'{name}_test_label.npy'))
        # Some MSL/SMAP labels are bool; collapse multi-dim labels (Anomaly-Transformer
        # files are already 1-D, but guard for variants).
        if test_label.ndim > 1:
            test_label = (test_label.sum(axis=tuple(range(1, test_label.ndim))) > 0)
        test_label = test_label.astype(np.int64)

    # Standardize train, apply same scaler to test (Anomaly-Transformer convention).
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x).astype(np.float32)
    test_x = scaler.transform(test_x).astype(np.float32)

    assert train_x.shape[1] == test_x.shape[1], \
        f'{name}: train D={train_x.shape[1]} != test D={test_x.shape[1]}'
    assert test_x.shape[0] == test_label.shape[0], \
        f'{name}: test T={test_x.shape[0]} != label T={test_label.shape[0]}'
    return train_x, test_x, test_label


def gen_ano_mv_train_data(train_x, window_size=100, stride=100):
    """Split a (T, D) multivariate train sequence into sliding windows.

    Returns (N_windows, window_size, D) suitable for trainer.fit().
    Default stride=window_size (non-overlapping) matches Anomaly-Transformer protocol.
    """
    T, D = train_x.shape
    if T < window_size:
        # Pad to window_size if needed
        pad = np.zeros((window_size - T, D), dtype=train_x.dtype)
        train_x = np.concatenate([train_x, pad], axis=0)
        T = window_size
    starts = np.arange(0, T - window_size + 1, stride)
    if len(starts) == 0 or starts[-1] + window_size < T:
        # Ensure we cover the tail
        starts = np.append(starts, T - window_size)
    windows = np.stack([train_x[s:s + window_size] for s in starts], axis=0)
    return windows.astype(np.float32)
