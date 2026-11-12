# -*- coding: utf-8 -*-
"""
Tests de integridad temporal. Ejecutar:  python -m pytest tests -q   (o  python tests/test_no_leakage.py <ruta base>)

1. Las posiciones de la estrategia hasta una fecha T no cambian si se alteran los retornos POSTERIORES a T
   (prueba de "futuro barajado": si cambiaran, la regla estaría mirando el futuro).
2. Las posiciones están siempre en {-3, -1, 0, 1, 3} (bases §2).
3. El P&L de la jornada t+1 usa la posición decidida en t: desplazar la posición un día cambia el resultado
   (si no lo hiciera, el motor estaría pagando la posición con el retorno del mismo día).
4. Las features no dependen de filas futuras: recortar la base en T no cambia la fila T.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from torneo.calendario import estrategia_calendario  # noqa: E402
from torneo.data import PX, cargar_base, construir_features, construir_jornadas, dataset  # noqa: E402
from torneo.evaluate import pnl  # noqa: E402

RUTA = os.environ.get("BASE_USDCLP", sys.argv[1] if len(sys.argv) > 1 else "data/base_usdclp.xlsx")
T = "2024-06-28"


def _cargar():
    return dataset(RUTA, inicio="2009-01-01")


def test_futuro_barajado_no_cambia_posiciones():
    j, F = _cargar()
    pos_a, _ = estrategia_calendario(F, "2014-01-02")
    F2 = F.copy()
    futuro = F2.index > pd.Timestamp(T)
    rng = np.random.default_rng(0)
    F2.loc[futuro, "ret_next"] = rng.permutation(F2.loc[futuro, "ret_next"].values)
    F2.loc[futuro, "rv20"] = F2.loc[futuro, "rv20"].values[::-1]
    pos_b, _ = estrategia_calendario(F2, "2014-01-02")
    hasta = pos_a.index <= pd.Timestamp(T)
    assert (pos_a[hasta].values == pos_b[hasta].values).all(), "las posiciones hasta T dependen de datos posteriores a T"


def test_posiciones_permitidas():
    j, F = _cargar()
    pos, _ = estrategia_calendario(F, "2014-01-02")
    assert set(np.unique(pos.values)) <= {-3.0, -1.0, 0.0, 1.0, 3.0}


def test_pnl_paga_con_retorno_siguiente():
    j, F = _cargar()
    pos, _ = estrategia_calendario(F, "2014-01-02")
    m = (pos.index >= "2018-01-01") & F["ret_next"].notna()
    a = pnl(pos[m], F, 2.0)["bruto"]
    b = pnl(pos[m].shift(1).fillna(0), F, 2.0)["bruto"]
    assert not np.allclose(a.values, b.values), "el P&L no distingue la posición de t de la de t-1"
    assert np.allclose(a.values, (pos[m] * F.loc[pos[m].index, "ret_next"]).values, equal_nan=True)


def test_features_no_usan_futuro():
    base = cargar_base(RUTA)
    F_full = construir_features(construir_jornadas(base, "2009-01-01"))
    F_cut = construir_features(construir_jornadas(base[base.index <= pd.Timestamp(T)], "2009-01-01"))
    cols = [c for c in F_cut.columns if c not in ("ret_next", "dias_next", "carry_next", "cash_next", "y", "y_reg", "ret_next5", "y5",
                                                  "prox_lunes", "prox_viernes", "prox_mar_jue")]
    fila_full = F_full.loc[pd.Timestamp(T), cols].astype(float)
    fila_cut = F_cut.loc[pd.Timestamp(T), cols].astype(float)
    assert np.allclose(fila_full.values, fila_cut.values, equal_nan=True), "alguna feature de T usa información posterior a T"


if __name__ == "__main__":
    for f in (test_futuro_barajado_no_cambia_posiciones, test_posiciones_permitidas, test_pnl_paga_con_retorno_siguiente, test_features_no_usan_futuro):
        f()
        print("OK", f.__name__)
