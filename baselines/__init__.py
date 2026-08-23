# Baseline models for time series SSL
# - ts2vec: TS2Vec (AAAI 2022)
# - softclt: SoftCLT (ICLR 2024)
# - timesurl: TimesURL (ICML 2024)
# - trep: T-Rep (NeurIPS 2023)
# - autotcl: AutoTCL (ICLR 2024)
# - infots: InfoTS (ICML 2023)
# - cost: CoST (ICLR 2022)
# - simmtm: SimMTM (NeurIPS 2023)
# - ts_jepa: TS-JEPA (2024)
# - fei: FEI (2024)
# - cf_jepa: CF-JEPA (ours, paper main — 3-zone forward latent prediction + horizon annealing)
#                          Skips crop loop & multi-scale invariance (single-view).
#                         (TimeMAE style). Skips crop loop & multi-scale invariance.
#                              same building block as encoder backbone Transformer in 5D)


METHODS = [
    'ts2vec', 'softclt', 'timesurl', 'trep', 'autotcl',
    'infots', 'cost', 'simmtm', 'ts_jepa', 'fei',
]


# Default (hidden_dims, output_dims) per method
METHOD_DIMS = {
    'ts2vec': (64, 320),
    'softclt': (64, 320),
    'timesurl': (64, 320),
    'trep': (64, 128),
    'autotcl': (64, 320),
    'infots': (60, 320),
    'cost': (64, 320),
    'simmtm': (64, 128),
    'ts_jepa': (64, 384),
    'fei': (64, 1024),
    'cf_jepa': (256, 320),
}


def get_method_dims(method):
    """Return (hidden_dims, output_dims) for a method."""
    return METHOD_DIMS.get(method, (64, 320))


def get_trainer(method, input_dims, device='cuda', **kwargs):
    if method == 'ts2vec':
        from .ts2vec.trainer import TS2VecTrainer
        return TS2VecTrainer(input_dims=input_dims, device=device, **kwargs)
    elif method == 'softclt':
        from .softclt.trainer import SoftCLTTrainer
        return SoftCLTTrainer(input_dims=input_dims, device=device, **kwargs)
    elif method == 'timesurl':
        from .timesurl.trainer import TimesURLTrainer
        return TimesURLTrainer(input_dims=input_dims, device=device, **kwargs)
    elif method == 'trep':
        from .trep.trainer import TRepTrainer
        return TRepTrainer(input_dims=input_dims, device=device, **kwargs)
    elif method == 'autotcl':
        from .autotcl.trainer import AutoTCLTrainer
        return AutoTCLTrainer(input_dims=input_dims, device=device, **kwargs)
    elif method == 'infots':
        from .infots.trainer import InfoTSTrainer
        return InfoTSTrainer(input_dims=input_dims, device=device, **kwargs)
    elif method == 'cost':
        from .cost.trainer import CoSTTrainer
        return CoSTTrainer(input_dims=input_dims, device=device, **kwargs)
    elif method == 'simmtm':
        from .simmtm.trainer import SimMTMTrainer
        return SimMTMTrainer(input_dims=input_dims, device=device, **kwargs)
    elif method == 'ts_jepa':
        from .ts_jepa.trainer import TSJEPATrainer
        return TSJEPATrainer(input_dims=input_dims, device=device, **kwargs)
    elif method == 'fei':
        from .fei.trainer import FEITrainer
        return FEITrainer(input_dims=input_dims, device=device, **kwargs)
    elif method == 'cf_jepa':
        from .cf_jepa.trainer import CFJEPATrainer
        # hidden_dims (and depth) flow via kwargs so a caller can override the default.
        kwargs.setdefault('hidden_dims', 256)
        return CFJEPATrainer(input_dims=input_dims, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Available: {', '.join(METHODS)}")
