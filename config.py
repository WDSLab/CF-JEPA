import os

# ==============================================================================
# Data root paths
# Change this path when running on a different machine.
# ==============================================================================

DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')

# Classification
UCR_PATH = os.path.join(DATA_ROOT, 'UCR')
UEA_PATH = os.path.join(DATA_ROOT, 'UEA')

# Forecasting (ETTh1.csv, ETTm1.csv, electricity.csv, weather.csv, etc.)
FORECAST_PATH = os.path.join(DATA_ROOT, 'forecast')

# Anomaly detection
# Univariate:   kpi.pkl, yahoo.pkl
# Multivariate: SMD/, MSL/, SMAP/, PSM/   (Anomaly-Transformer preproc)
ANOMALY_PATH = os.path.join(DATA_ROOT, 'anomaly')
