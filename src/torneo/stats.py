# -*- coding: utf-8 -*-
"""
Pruebas exigidas por las bases (§5):
  * Diebold-Mariano (1995) con la corrección de muestra pequeña de Harvey-Leybourne-Newbold (1997) y varianza
    HAC (Newey-West). Se aplica (a) sobre la calidad de la predicción: pérdida de Brier del modelo vs la del
    pronóstico ingenuo (p = frecuencia histórica de alzas), y (b) sobre la pérdida económica: −P&L diario de la
    estrategia vs −P&L de cada referencia.
  * Bootstrap estacionario de Politis-Romano (1994) sobre la serie diaria de diferencias de P&L (bloques de largo
    aleatorio geométrico, media `bloque`), que respeta la autocorrelación que inducen las posiciones persistentes.
"""
import numpy as np
import pandas as pd
from scipy import stats as st


def _newey_west_var(d: np.ndarray, lags: int) -> float:
    T = len(d)
    dc = d - d.mean()
    g0 = np.dot(dc, dc) / T
    var = g0
    for k in range(1, lags + 1):
        gk = np.dot(dc[k:], dc[:-k]) / T
        var += 2 * (1 - k / (lags + 1)) * gk
    return var / T


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, h: int = 1, lags: int | None = None) -> dict:
    """H0: igual pérdida esperada. Estadístico negativo favorece a 'a' (menor pérdida)."""
    d = (loss_a - loss_b).dropna().values
    T = len(d)
    if lags is None:
        lags = max(h - 1, int(np.floor(4 * (T / 100) ** (2 / 9))))   # regla de Newey-West
    var = _newey_west_var(d, lags)
    dm = d.mean() / np.sqrt(var) if var > 0 else 0.0
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    pval = 2 * (1 - st.t.cdf(abs(hln), df=T - 1))
    return {"T": T, "media_dif": d.mean(), "DM": dm, "DM_HLN": hln, "p_valor": pval, "lags": lags}


def bootstrap_estacionario(x: np.ndarray, B: int = 10000, bloque: float = 10.0, seed: int = 42) -> np.ndarray:
    """Devuelve B sumas remuestreadas de la serie x (Politis-Romano, p = 1/bloque)."""
    rng = np.random.default_rng(seed)
    T = len(x)
    p = 1.0 / bloque
    out = np.empty(B)
    for b in range(B):
        idx = np.empty(T, dtype=int)
        idx[0] = rng.integers(T)
        for t in range(1, T):
            idx[t] = rng.integers(T) if rng.random() < p else (idx[t - 1] + 1) % T
        out[b] = x[idx].sum()
    return out


def ic_bootstrap(serie: pd.Series, B: int = 10000, bloque: float = 10.0, seed: int = 42) -> dict:
    """IC 95 % del resultado acumulado y probabilidad de que sea > 0 (serie = diferencia diaria de P&L)."""
    x = serie.dropna().values
    sums = bootstrap_estacionario(x, B, bloque, seed)
    return {"observado_%": x.sum() * 100, "ic95_%": (np.percentile(sums, 2.5) * 100, np.percentile(sums, 97.5) * 100),
            "p_mayor_que_0": (sums > 0).mean(), "B": B, "bloque_medio": bloque}


def brier(p: pd.Series, y: pd.Series) -> pd.Series:
    return (p - y) ** 2
