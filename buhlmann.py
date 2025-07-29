import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

"""
Buhlmann decompression model ZH-L 16 implementation based on Subsurface XML data.
This code parses a Subsurface XML file, extracts dive data, and computes the Buhlmann decompression model.
It then plots the dive profile and the computed ceiling.
This code is intended for educational purposes and should not be used for actual diving planning.

The Buhlmann decompression model is a complex algorithm used in diving to calculate safe ascent profiles based on tissue saturation levels. 

Do not use this code for real diving activities without proper training and understanding of decompression theory.
This implementation is a simplified version and may not cover all aspects of the Buhlmann model.


Roger Skjetlein, 2025
"""


def parse_subsurface_xml(filepath: str) -> pd.DataFrame:
    tree = ET.parse(filepath)
    root = tree.getroot()
    samples = root.find('.//divecomputer').findall('sample')
    times, depths = [], []
    for s in samples:
        t_str = s.get('time').replace(' min', '')
        if ':' in t_str:
            mins, secs = map(float, t_str.split(':'))
            times.append(mins + secs/60)
        else:
            times.append(float(t_str))
        depths.append(float(s.get('depth').split()[0]))
    return pd.DataFrame({'Time': times, 'Depth': depths}).sort_values('Time').reset_index(drop=True)

def compute_buhlmann(df, half_times, a, b, f_n2=0.79, gf_low=0.3, gf_high=0.85):
    n = len(df)
    taus = half_times / np.log(2)
    tissues = np.zeros((n, len(half_times)))
    tissues[0] = f_n2
    M_raw = np.zeros_like(tissues)
    M_scaled = np.zeros_like(tissues)
    ceiling = np.zeros(n)
    for i in range(1, n):
        dt = df.loc[i, 'Time'] - df.loc[i-1, 'Time']
        p_amb_prev = 1 + df.loc[i-1, 'Depth']/10
        p_i = p_amb_prev * f_n2
        tissues[i] = tissues[i-1] + (p_i - tissues[i-1]) * (1 - np.exp(-dt/taus))
        p_amb = 1 + df.loc[i, 'Depth']/10
        M_raw[i] = a * p_amb + b
        progress = df.loc[i, 'Time'] / df['Time'].max()
        gf = gf_low + (gf_high - gf_low) * progress
        M_scaled[i] = M_raw[i] * gf
        p_ceiling = (M_scaled[i] - b) / a
        ceiling[i] = max(0.0, np.max(10 * (p_ceiling - 1)))
    out = df.copy()
    for j in range(len(half_times)):
        out[f'P_tissue_C{j+1}'] = tissues[:, j]
        out[f'M_scaled_C{j+1}'] = M_scaled[:, j]
    out['Ceiling'] = ceiling
    return out

if __name__ == "__main__":
    half_times = np.array([4,8,12.5,18.5,27,38.3,54.3,77,109,146,187,239,305,390,498,635])
    a = np.array([1.2599,1.0,0.8618,0.7562,0.6667,0.5933,0.5282,0.4701,
                  0.4187,0.3798,0.3497,0.3223,0.2971,0.2737,0.2523,0.2327])
    b = np.array([0.5050,0.6514,0.7222,0.7895,0.8590,0.9222,0.9625,1.0,
                  1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
    df = parse_subsurface_xml('test.xml')
    results = compute_buhlmann(df, half_times, a, b)
    # Plot
    import matplotlib.pyplot as plt
    plt.plot(results['Time'], results['Depth'], label='Dive')
    plt.plot(results['Time'], results['Ceiling'], '--', label='Ceiling')
    plt.gca().invert_yaxis(); plt.legend(); plt.show()


