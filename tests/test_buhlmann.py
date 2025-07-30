import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from buhlmann import compute_buhlmann


def test_gradient_factor_progression():
    df = pd.DataFrame({'Time': [0, 10], 'Depth': [30, 0]})
    half_times = np.array([4.0])
    a = np.array([1.0])
    b = np.array([0.5])
    res = compute_buhlmann(df, half_times, a, b, gf_low=0.3, gf_high=1.0)
    p_amb = 1 + res.loc[1, 'Depth'] / 10
    m_raw = a[0] * p_amb + b[0]
    gf = res.loc[1, 'M_scaled_C1'] / m_raw
    assert np.isclose(gf, 1.0)


def test_ceiling_non_negative():
    df = pd.DataFrame({'Time': [0, 10], 'Depth': [0, 0]})
    half_times = np.array([4.0])
    a = np.array([1.0])
    b = np.array([0.5])
    res = compute_buhlmann(df, half_times, a, b)
    assert (res['Ceiling'] >= 0).all()
