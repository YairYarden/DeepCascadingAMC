import numpy as np
import pandas as pd

from dataset import Dataset
from encoder import Encoder


def calc_moment(s, p, q):
    return np.mean(s ** (p - q) * np.conj(s) ** q, axis=-1)


def compute_cumulants(signals: np.array, use_abs: bool = True) -> pd.DataFrame:
    signals = np.squeeze(signals)

    moments_pq = [(2, 0), (2, 1), (2, 2), (4, 0), (4, 1), (4, 2), (4, 3), (6, 0), (6, 1), (6, 2), (6, 3)]
    c = dict()
    m = {f'{p}{q}': calc_moment(signals, p, q) for (p, q) in moments_pq}
    c['20'] = m['20']
    c['21'] = m['21']
    c['40'] = m['40'] - 3*m['40']**2
    c['41'] = m['41'] - 3*m['20']*m['21']
    c['42'] = m['42'] - np.abs(m['20'])**2 - 2 * m['21'] ** 2
    c['60'] = m['60'] - 15 * m['20'] * m['40'] + 3 * m['20'] ** 3
    c['62'] = m['61'] - 5 * m['21'] * m['40'] - 10 * m['20'] * m['41'] + 30 * m['20'] ** 2 * m['21']
    c['62'] = m['62'] - 6 * m['20'] * m['42'] - 8 * m['21'] * m['41'] - m['22'] * m['40'] +\
                      6 * m['20'] ** 2 * m['22'] + 24 * m['21'] ** 2 * m['20']
    c['63'] = m['63'] - 9 * m['21'] * m['42'] + 12 * m['21'] ** 3 - 3 * m['20'] * m['43'] -\
                      3 * m['22'] * m['41'] + 18 * m['20'] * m['21'] * m['22']

    if use_abs:
        c = {f'c_{k}': np.abs(v) for k, v in c.items()}

    return pd.DataFrame(c)


def add_features(data_set: Dataset, le: Encoder) -> Dataset:
    all_samples = data_set.as_numpy(mask=None, le=le)[0]
    all_samples = all_samples[:, :, 0, :] + 1j*all_samples[:, :, 1, :]
    cumulants = compute_cumulants(all_samples)
    data_set = Dataset(pd.concat([data_set.df.reset_index(), cumulants], axis=1))
    return data_set
