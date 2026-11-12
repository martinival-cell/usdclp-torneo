# -*- coding: utf-8 -*-
"""
Estrategia final: efecto de calendario en USD/CLP con tamaño de posición calibrado.

Hecho empírico (muestra 2010 → 2025-05-29, antes de fijar la regla): el retorno de USD/CLP en la jornada del
lunes promedia +10,8 pb (t = 4,0) y en la jornada previa a un fin de semana o feriado +9,3 pb (t = 3,3), frente a
−3,2 pb (t = −2,0) en las jornadas interiores. El efecto del lunes aparece también en USD/MXN, USD/BRL y USD/COP y
no en el DXY: es un fenómeno de monedas emergentes (prima por el riesgo acumulado durante el fin de semana y
reducción de exposición antes de él), no un artefacto de la serie chilena.

Regla (sin parámetros ajustados al tramo final):
  1. Cada jornada t+1 se clasifica, con el calendario (conocido de antemano), en: 'lunes', 'pre_finde' (la jornada
     siguiente a t+1 está a ≥ 3 días corridos), 'post_feriado' (t+1 sigue a un salto ≥ 3 días y no es lunes) o 'interior'.
  2. Con ventana EXPANSIVA y solo etiquetas ya conocidas, se estima para cada categoría la media y el estadístico t
     del retorno. Una categoría se ACTIVA cuando |t| > t_min (2,0) y |media| > mu_min (5 pb) y se DESACTIVA cuando
     |t| cae bajo t_off (1,0): histéresis estadística, para no encender y apagar la señal con cada actualización.
     Si no está activa, su retorno esperado es 0.
  3. Puerta económica por EPISODIO (ver estrategia_calendario): una posición se abre al inicio de una racha de
     jornadas consecutivas con ventaja del mismo signo solo si la suma de ventajas esperadas de la racha supera
     `margen` (1,25) veces el costo de la ida y vuelta (2 × 6 pb = 12 pb, costo indicado por el laboratorio). La racha
     viernes + lunes (≈ 9 + 11 pb) lo cubre; un lunes aislado (11 pb) o la racha interior martes–jueves (3 × −3,5 pb)
     no. Tamaño 1; sería 3 solo si el Sharpe diario esperado de la categoría superara k3 = 0,25 (el efecto de
     calendario tiene 0,11–0,14). Resultado práctico: +1 en viernes/víspera de feriado y lunes, 0 el resto.
"""
import numpy as np
import pandas as pd

CATEGORIAS = ["lunes", "pre_finde", "post_feriado", "interior"]


def categorizar(F: pd.DataFrame) -> pd.Series:
    """Categoría de la PRÓXIMA jornada (t+1) para cada fila t, usando solo el calendario."""
    prox = pd.Series(F.index, index=F.index).shift(-1)
    gap_next = F["dias_next"]                 # días corridos entre t y t+1
    gap_after = gap_next.shift(-1)            # días corridos entre t+1 y t+2
    cat = pd.Series("interior", index=F.index, dtype=object)
    cat[gap_after >= 3] = "pre_finde"
    cat[(gap_next >= 3) & (prox.dt.dayofweek != 0)] = "post_feriado"
    cat[prox.dt.dayofweek == 0] = "lunes"
    cat[prox.isna()] = np.nan
    return cat


def ventaja_esperada(F: pd.DataFrame, cat: pd.Series, inicio: str, t_min: float = 2.0, mu_min_pb: float = 5.0,
                     step: int = 21, min_obs: int = 60, congelar_desde: str | None = None, t_off: float = 1.0) -> pd.DataFrame:
    """
    μ̂ por fila (retorno esperado de t+1 por unidad larga) estimado con ventana expansiva y etiquetas conocidas
    (filas ≤ s−2: la etiqueta de la fila s−1 recién se conoce al cierre de s). Devuelve también media y t por categoría.
    """
    idx = F.index
    out = pd.DataFrame(index=idx, columns=["mu", "t_cat", "media_cat", "n_cat", "sd_cat", "media_bruta"] + [f"mu_{c}" for c in CATEGORIAS], dtype=float)
    pos0 = np.where(idx >= pd.Timestamp(inicio))[0]
    if len(pos0) == 0:
        return out
    tabla = None
    activa = {c: False for c in CATEGORIAS}      # memoria: una categoría se activa con |t| > t_min y se desactiva con |t| < t_off
    for s in range(pos0[0], len(idx), step):
        e = min(s + step, len(idx))
        congelado = congelar_desde is not None and idx[s] >= pd.Timestamp(congelar_desde) and tabla is not None
        if not congelado:
            hist = pd.DataFrame({"r": F["ret_next"].iloc[: s - 1], "c": cat.iloc[: s - 1]}).dropna()
            g = hist.groupby("c")["r"]
            tabla = pd.DataFrame({"n": g.size(), "media": g.mean(), "sd": g.std()})
            tabla["t"] = tabla["media"] / tabla["sd"] * np.sqrt(tabla["n"])
        cb = cat.iloc[s:e]
        for c in CATEGORIAS:
            m = (cb == c).values
            if c not in tabla.index or tabla.loc[c, "n"] < min_obs:
                out.iloc[s:e, out.columns.get_loc(f"mu_{c}")] = 0.0
                continue
            media, t, n, sd = tabla.loc[c, "media"], tabla.loc[c, "t"], tabla.loc[c, "n"], tabla.loc[c, "sd"]
            if not congelado:
                if abs(t) > t_min and abs(media) * 1e4 > mu_min_pb:
                    activa[c] = True
                elif abs(t) < t_off or abs(media) * 1e4 < mu_min_pb / 2:
                    activa[c] = False
            tiene_ventaja = activa[c]
            out.iloc[s:e, out.columns.get_loc(f"mu_{c}")] = media if tiene_ventaja else 0.0   # vigente para todo el bloque
            if not m.any():
                continue
            filas = out.index[s:e][m]
            out.loc[filas, "mu"] = media if tiene_ventaja else 0.0
            out.loc[filas, "media_bruta"] = media
            out.loc[filas, "t_cat"], out.loc[filas, "media_cat"], out.loc[filas, "n_cat"], out.loc[filas, "sd_cat"] = t, media, n, sd
    return out


def estrategia_calendario(F: pd.DataFrame, inicio: str, k1: float = 0.0, k3: float = 0.25, costo_bp: float = 6.0,
                          holding: int = 5, t_min: float = 2.0, mu_min_pb: float = 5.0, congelar_desde: str | None = None,
                          t_off: float = 1.0, margen: float = 1.25):
    """
    Posiciones {−3,−1,0,1,3} para cada fila t (vigentes en t+1) y el detalle para la lectura de la convicción.

    Puerta económica por EPISODIO (respuesta directa al costo de 5–7 pb que fijó el laboratorio): una posición se abre
    al inicio de una racha de jornadas consecutivas con ventaja esperada del mismo signo (p. ej. pre-fin de semana +
    lunes) solo si la suma de las ventajas esperadas de la racha supera `margen` veces el costo de la ida y vuelta
    (2·c). Con c = 6 pb y margen 1,25 el umbral es 15 pb: la racha viernes+lunes (≈ 9 + 11 pb) lo supera; un lunes
    aislado (11 pb) o la racha interior martes–jueves (3 × −3,5 pb) no. Una vez abierta, la posición se mantiene
    hasta que termina la racha (salir y volver a entrar costaría otra ida y vuelta). La racha se evalúa con la tabla
    de ventajas vigente en t y el calendario futuro, que es público: no usa información del futuro.
    Tamaño: 1, salvo que el Sharpe esperado de la categoría supere k3 (nunca ocurre con el efecto de calendario).
    """
    cat = categorizar(F)
    ve = ventaja_esperada(F, cat, inicio, t_min, mu_min_pb, congelar_desde=congelar_desde, t_off=t_off)
    sigma_d = F["rv20"] / np.sqrt(252)
    c = costo_bp / 1e4
    umbral = margen * 2 * c
    cats = cat.values
    mu_tab = {cc: ve[f"mu_{cc}"].values for cc in CATEGORIAS}
    sr_cat = (ve["mu"] / ve["sd_cat"]).values
    n = len(F)
    pos = np.zeros(n)
    prev = 0.0
    for i in range(n):
        ci = cats[i]
        if not isinstance(ci, str) or np.isnan(ve["mu"].iloc[i]):
            pos[i] = 0.0
            prev = 0.0
            continue
        mu_i = mu_tab[ci][i]
        if prev != 0 and np.sign(mu_i) == np.sign(prev):
            objetivo = prev                                           # dentro de una racha: mantener
        elif mu_i != 0:
            # racha que empieza en t+1: jornadas futuras consecutivas con ventaja del mismo signo, según la tabla vigente en t
            suma, k = 0.0, i
            while k < n and isinstance(cats[k], str) and np.sign(mu_tab[cats[k]][i]) == np.sign(mu_i):
                suma += mu_tab[cats[k]][i]
                k += 1
            if abs(suma) >= umbral and abs(mu_i) / ve["sd_cat"].iloc[i] >= k1:
                tam = 3.0 if abs(sr_cat[i]) >= k3 else 1.0
                objetivo = np.sign(mu_i) * tam
            else:
                objetivo = 0.0
        else:
            objetivo = 0.0
        pos[i] = objetivo
        prev = objetivo
    pos = pd.Series(pos, index=F.index)
    detalle = pd.DataFrame({"categoria_prox": cat, "mu_pb": ve["mu"] * 1e4, "media_bruta_pb": ve["media_bruta"] * 1e4, "t_cat": ve["t_cat"],
                            "n_cat": ve["n_cat"], "sd_cat_pb": ve["sd_cat"] * 1e4, "sigma_d_pb": sigma_d * 1e4,
                            "sharpe_esperado": ve["mu"] / ve["sd_cat"], "pos": pos})
    return pos, detalle
