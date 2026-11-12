# -*- coding: utf-8 -*-
"""
Motor de evaluación. Replica la regla de las bases bajo las hipótesis documentadas en la auditoría (A.4):

    resultado_t+1 = pos_t · ΔS/S_t+1  −  pos_t · carry_t+1  −  c · |pos_t − pos_t−1|

  * pos_t: exposición en múltiplos del nocional de referencia, decidida con la fila t, vigente en t+1.
  * carry_t+1 = (TPM − SOFR)/360 · días corridos hasta la jornada t+1 (el largo paga, el corto recibe).
  * c: costo por unidad de cambio de exposición. El laboratorio indicó (11-11-2026) considerar 5 a 7 pb, que
    cubren comisión de la entidad y puntas de mercado; base 6 pb, sensibilidad 0/2/5/6/7/10.
  * P&L aditivo sobre nocional fijo. Se reporta como excedente sobre la tasa en pesos (la que gana "no operar");
    el total suma cash_next. El Sharpe se calcula sobre ese excedente: es el retorno ajustado por riesgo "mediante
    la tasa chilena" que sugirió el laboratorio.

Métricas (las que el laboratorio dijo que evaluará: drawdown, acierto direccional con y sin apalancamiento,
retorno ajustado por riesgo), más Sortino, Calmar y la razón ganancia/pérdida.

Referencias de las bases: comprar siempre (+1), no operar nunca (0, gana la tasa local), operar siempre con
exposición uno (dirección del modelo con tamaño 1). Cuando exista un código oficial del laboratorio, este
módulo debe reemplazarse o reconciliarse con él.
"""
import numpy as np
import pandas as pd

COSTO_BASE_PB = 6.0


def pnl(pos: pd.Series, F: pd.DataFrame, costo_bp: float = COSTO_BASE_PB) -> pd.DataFrame:
    df = pd.DataFrame(index=pos.index)
    df["pos"] = pos.values
    df["ret_next"] = F.loc[pos.index, "ret_next"]
    df["carry_next"] = F.loc[pos.index, "carry_next"].fillna(0.0)
    df["cash_next"] = F.loc[pos.index, "cash_next"].fillna(0.0)
    df["turnover"] = df["pos"].diff().abs().fillna(df["pos"].abs())
    df["bruto"] = df["pos"] * df["ret_next"]
    df["con_carry"] = df["bruto"] - df["pos"] * df["carry_next"]
    df["neto"] = df["con_carry"] - df["turnover"] * costo_bp / 1e4
    df["total"] = df["neto"] + df["cash_next"]
    return df.dropna(subset=["ret_next"])


def resumen(df: pd.DataFrame, direccion: pd.Series | None = None) -> dict:
    """
    direccion: señal direccional del modelo (+1/−1/0) para TODAS las jornadas, para medir el acierto sin
    apalancamiento (incluye las jornadas en que la estrategia se abstuvo). Si no se entrega, se usa el signo
    de la posición.
    """
    n = len(df)
    neto = df["neto"]
    sd = neto.std()
    acum = neto.cumsum()
    dd = (acum - acum.cummax()).min()
    anios = n / 252
    operado = df["pos"] != 0
    signo_pos = np.sign(df["pos"])
    signo_ret = np.sign(df["ret_next"])
    acierto_operado = float((signo_pos[operado] == signo_ret[operado]).mean()) if operado.any() else np.nan
    dirr = np.sign(direccion.loc[df.index]) if direccion is not None else signo_pos
    con_dir = dirr != 0
    acierto_total = float((dirr[con_dir] == signo_ret[con_dir]).mean()) if con_dir.any() else np.nan
    # acierto ponderado por exposición ("con apalancamiento"): jornadas operadas ponderadas por |pos|
    w = df["pos"].abs()
    acierto_apal = float(((signo_pos == signo_ret) * w).sum() / w.sum()) if w.sum() > 0 else np.nan
    gan = neto[operado & (neto > 0)]
    per = neto[operado & (neto < 0)]
    downside = np.sqrt(np.mean(np.minimum(neto, 0.0) ** 2))   # desviación a la baja (objetivo 0) sobre todas las jornadas
    # operaciones: cada cambio de exposición (entrada, salida o cambio de tamaño) es una operación; un episodio es
    # un período continuo con posición distinta de cero (entrada → salida)
    cambios = df["turnover"] > 0
    entradas = operado & ~operado.shift(1, fill_value=False)
    return {
        "jornadas": n,
        "bruto_%": df["bruto"].sum() * 100,
        "con_carry_%": df["con_carry"].sum() * 100,
        "costos_%": (df["con_carry"] - df["neto"]).sum() * 100,
        "neto_%": neto.sum() * 100,
        "total_%": df["total"].sum() * 100,
        "neto_anual_%": neto.sum() / anios * 100 if anios > 0 else np.nan,
        "vol_anual_%": sd * np.sqrt(252) * 100,
        "sharpe_neto_anual": (neto.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        "sortino_anual": (neto.mean() / downside * np.sqrt(252)) if downside and downside > 0 else np.nan,
        "max_drawdown_%": dd * 100,
        "calmar": (neto.sum() / anios / abs(dd)) if dd < 0 and anios > 0 else np.nan,
        "acierto_direccional_operado": acierto_operado,
        "acierto_direccional_total": acierto_total,
        "acierto_ponderado_exposicion": acierto_apal,
        "ganancia_media_pb": gan.mean() * 1e4 if len(gan) else np.nan,
        "perdida_media_pb": per.mean() * 1e4 if len(per) else np.nan,
        "razon_ganancia_perdida": (gan.mean() / abs(per.mean())) if len(gan) and len(per) else np.nan,
        "n_operaciones": int(cambios.sum()),
        "n_episodios": int(entradas.sum()),
        "jornadas_operadas": int(operado.sum()),
        "turnover_unidades": df["turnover"].sum(),
        "dias_fuera_%": (df["pos"] == 0).mean() * 100,
        "dias_±3_%": (df["pos"].abs() == 3).mean() * 100,
    }


def referencias(F: pd.DataFrame, idx: pd.Index, p: pd.Series | None, costo_bp: float = COSTO_BASE_PB) -> dict:
    """Las tres referencias de las bases sobre el mismo tramo y con la misma regla de P&L."""
    refs = {
        "comprar_siempre": pd.Series(1.0, index=idx),
        "no_operar": pd.Series(0.0, index=idx),
    }
    if p is not None:
        refs["intensidad_constante"] = pd.Series(np.nan_to_num(np.sign(2 * p.loc[idx] - 1)), index=idx)
    return {k: pnl(v, F, costo_bp) for k, v in refs.items()}


def tabla_costos(pos: pd.Series, F: pd.DataFrame, costos=(0, 2, 5, 6, 7, 10)) -> pd.DataFrame:
    filas = []
    for c in costos:
        r = resumen(pnl(pos, F, c))
        filas.append({"costo_pb": c, "neto_%": round(r["neto_%"], 2), "neto_anual_%": round(r["neto_anual_%"], 2),
                      "sharpe": round(r["sharpe_neto_anual"], 2), "max_dd_%": round(r["max_drawdown_%"], 2)})
    return pd.DataFrame(filas)
