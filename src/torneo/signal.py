# -*- coding: utf-8 -*-
"""
Señal del día (entregable 1 de las bases): dirección estimada, probabilidad y posición sugerida para la
jornada siguiente, usando SOLO filas ≤ hoy. Es la misma regla que evalúa el backtest (torneo.calendario), de
modo que lo que se mide fuera de muestra es exactamente lo que opera en vivo.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .calendario import estrategia_calendario
from .data import PX, construir_features, construir_jornadas


FERIADOS_CL = ["2026-12-08", "2026-12-25", "2027-01-01"]   # feriados chilenos próximos (sin jornada USD/CLP); actualizar cada año


def generar_senal(df_base: pd.DataFrame, hasta: str | None = None, k1: float = 0.0, k3: float = 0.25,
                  costo_bp: float = 6.0, inicio_regla: str = "2014-01-02", inicio_datos: str = "2009-01-01",
                  feriados=FERIADOS_CL) -> dict:
    """
    df_base: hoja 'datos' indexada por fecha (si trae filas posteriores a `hasta`, se recortan).
    Devuelve la señal para la jornada siguiente a la última jornada USD/CLP disponible.
    """
    if hasta is not None:
        df_base = df_base[df_base.index <= pd.Timestamp(hasta)]
    j = construir_jornadas(df_base, inicio_datos)
    F = construir_features(j)
    # las próximas jornadas aún no existen en la base: se agregan como filas vacías (solo calendario) para que la regla
    # pueda clasificar la jornada objetivo y evaluar la racha completa (p. ej. viernes + lunes)
    hoy = F.index[-1]
    fer = {pd.Timestamp(x) for x in (feriados or [])}
    F = _extender_calendario(F, fer, n_min=4)
    prox = F.index[F.index.get_loc(hoy) + 1]
    pos, det = estrategia_calendario(F, inicio_regla, k1, k3, costo_bp)
    d = det.loc[hoy]
    sigma_cat = d["sd_cat_pb"] if pd.notna(d["sd_cat_pb"]) else d["sigma_d_pb"]
    p_up = float(norm.cdf((d["media_bruta_pb"] if pd.notna(d["media_bruta_pb"]) else 0.0) / sigma_cat)) if sigma_cat else 0.5
    posicion = int(pos.loc[hoy])
    return {
        "fecha_informacion": hoy.date().isoformat(),
        "jornada_objetivo": prox.date().isoformat(),
        "categoria_jornada_objetivo": d["categoria_prox"],
        "cierre_usdclp": float(j.loc[hoy, PX]),
        "direccion": "alza" if p_up > 0.5 else ("baja" if p_up < 0.5 else "neutral"),
        "prob_alza": round(p_up, 3),
        "ventaja_esperada_pb": round(float(d["mu_pb"]) if pd.notna(d["mu_pb"]) else 0.0, 2),
        "t_categoria": round(float(d["t_cat"]), 2) if pd.notna(d["t_cat"]) else None,
        "sharpe_esperado_diario": round(float(d["sharpe_esperado"]), 3) if pd.notna(d["sharpe_esperado"]) else 0.0,
        "posicion_sugerida": posicion,
        "lectura": _lectura(d, posicion),
    }


def calendario_esperado(df_base: pd.DataFrame, desde: str, hasta: str, k1: float = 0.0, k3: float = 0.25,
                        costo_bp: float = 6.0, inicio_regla: str = "2014-01-02", inicio_datos: str = "2009-01-01",
                        feriados=FERIADOS_CL) -> pd.DataFrame:
    """
    Proyección de la posición que la regla emitiría para cada jornada objetivo entre `desde` y `hasta` (p. ej. el
    período de monitoreo 2026-11-13 → 2026-12-01), usando la tabla de ventajas vigente al final de la base y el
    calendario de jornadas futuras (días hábiles menos feriados chilenos). Sirve para que el monitor del laboratorio
    contraste su simulación. La posición vinculante de cada día es la que emite generar_senal() con la base
    actualizada: la tabla se re-estima cada 21 jornadas, de modo que la proyección solo cambiaría si una categoría
    cruzara su umbral de activación (|t| de 2 a 1), algo que con cientos de observaciones por categoría exigiría un
    cambio extremo en pocas semanas.
    Columnas: fecha_informacion (cierre con que se decide), jornada_objetivo (jornada en que rige la posición),
    categoria, ventaja_esperada_pb, posicion.
    """
    j = construir_jornadas(df_base, inicio_datos)
    F = construir_features(j)
    fer = {pd.Timestamp(x) for x in (feriados or [])}
    F = _extender_calendario(F, fer, n_min=4, hasta=pd.Timestamp(hasta))
    pos, det = estrategia_calendario(F, inicio_regla, k1, k3, costo_bp)
    prox = pd.Series(F.index, index=F.index).shift(-1)
    out = pd.DataFrame({"fecha_informacion": F.index, "jornada_objetivo": prox.values,
                        "dia": prox.dt.day_name().str[:3].values, "categoria": det["categoria_prox"].values,
                        "ventaja_esperada_pb": det["mu_pb"].round(1).values, "posicion": pos.astype(int).values})
    m = (out["jornada_objetivo"] >= pd.Timestamp(desde)) & (out["jornada_objetivo"] <= pd.Timestamp(hasta))
    return out[m].reset_index(drop=True)


def _extender_calendario(F: pd.DataFrame, feriados=frozenset(), n_min: int = 4, hasta: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Agrega al final de F filas vacías (solo calendario) para las próximas jornadas USD/CLP: al menos `n_min` y, si se
    indica `hasta`, todas las jornadas hasta esa fecha más tres adicionales (la clasificación de una jornada necesita
    conocer las dos siguientes). Las filas nuevas no tienen retornos: la regla solo usa su categoría de calendario.
    """
    anterior = F.index[-1]
    nuevas = []
    f_ = _proxima_jornada(anterior, feriados)
    extra_tras_hasta = 0
    while len(nuevas) < n_min or (hasta is not None and extra_tras_hasta < 3):
        nuevas.append(f_)
        if hasta is not None and f_ > hasta:
            extra_tras_hasta += 1
        f_ = _proxima_jornada(f_, feriados)
    ext = pd.DataFrame(np.nan, index=pd.DatetimeIndex(nuevas, name=F.index.name), columns=F.columns)
    F = pd.concat([F, ext])
    for a, b in zip([anterior] + nuevas[:-1], nuevas):
        F.loc[a, "dias_next"] = (b - a).days
    return F


def _proxima_jornada(fecha: pd.Timestamp, feriados=frozenset()) -> pd.Timestamp:
    """Siguiente jornada USD/CLP: día hábil (lunes a viernes) que no sea feriado chileno."""
    f = fecha + pd.Timedelta(days=1)
    while f.dayofweek >= 5 or f in feriados:
        f += pd.Timedelta(days=1)
    return f


def _lectura(d, posicion: int) -> str:
    cat = {"lunes": "lunes (primera jornada tras el fin de semana)", "pre_finde": "última jornada antes de un fin de semana o feriado",
           "post_feriado": "jornada posterior a un feriado", "interior": "jornada interior de la semana"}.get(d["categoria_prox"], d["categoria_prox"])
    if posicion == 0:
        return (f"Sin posición: la jornada objetivo es {cat}; la ventaja histórica de esa categoría no supera el umbral "
                f"estadístico (|t| > 2) o la ventaja acumulada de la racha de jornadas con ventaja no cubre 1,25 veces el costo de ida y vuelta (12 pb). Quedarse en pesos ganando la tasa local.")
    lado = "compra" if posicion > 0 else "venta en corto"
    return (f"{lado.capitalize()} de dólar con exposición {abs(posicion)}: la jornada objetivo es {cat}, categoría con ventaja "
            f"histórica de {d['media_bruta_pb']:.1f} pb/día (t = {d['t_cat']:.1f}) y Sharpe diario esperado {d['sharpe_esperado']:.2f}.")
