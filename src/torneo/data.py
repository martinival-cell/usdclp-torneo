# -*- coding: utf-8 -*-
"""
Carga de la base del torneo y construcción del dataset sobre el calendario de jornadas USD/CLP.

Convención (diccionario del kick-off): "los valores de cierre de una misma fecha están todos disponibles
al momento de decidir". Por lo tanto, la fila t (cierres fechados t) decide la posición de la jornada t+1,
que se paga con el retorno de t+1. Aquí NO se rezagan las features: se construyen con información ≤ t y
el objetivo mira a t+1.

Tratamiento de huecos (pregunta 5 del kick-off):
  * "mercado cerrado" (feriado de la plaza de la serie): se arrastra el último cierre (ffill) sobre el
    calendario completo y luego se seleccionan solo las jornadas con cierre USD/CLP. Así el retorno de una
    serie entre dos jornadas USD/CLP consecutivas es el retorno realmente acumulado entre esos dos cierres.
  * "serie inexistente" o descontinuada: no se usa (IPSA, GTCLP*, US0001M, flujos BCCh).
  * "frecuencia mensual" (macro_cl): no se usa en esta versión; si se usa, solo por FECHA_RELEASE.
"""
import numpy as np
import pandas as pd

PX = "USDCLP Curncy|PX_LAST"
PX_OPEN, PX_HIGH, PX_LOW = "USDCLP Curncy|PX_OPEN", "USDCLP Curncy|PX_HIGH", "USDCLP Curncy|PX_LOW"

SERIES = {
    "cobre": "LMCADS03 Comdty|PX_LAST",
    "hg": "HG1 Comdty|PX_LAST",
    "dxy": "DXY Curncy|PX_LAST",
    "bbdxy": "BBDXY Index|PX_LAST",
    "mxn": "USDMXN Curncy|PX_LAST",
    "brl": "USDBRL Curncy|PX_LAST",
    "cop": "USDCOP Curncy|PX_LAST",
    "zar": "USDZAR Curncy|PX_LAST",
    "cnh": "USDCNH Curncy|PX_LAST",
    "spx": "SPX Index|PX_LAST",
    "mxef": "MXEF Index|PX_LAST",
    "vix": "VIX Index|PX_LAST",
    "move": "MOVE Index|PX_LAST",
    "xau": "XAU Curncy|PX_LAST",
    "vclp1m": "USDCLPV1M Curncy|PX_LAST",
    "vclp3m": "USDCLPV3M Curncy|PX_LAST",
    "vxyem": "JPMVXYEM Index|PX_LAST",
    "us2y": "USGG2YR Index|PX_LAST",
    "us10y": "USGG10YR Index|PX_LAST",
    "chswp1": "CHSWP1 Curncy|PX_LAST",
    "chswp2": "CHSWP2 Curncy|PX_LAST",
    "ndf1m": "CHNI1M Curncy|PX_LAST",
}
RATES = {"tpm": "CHOVCHOV Index|PX_LAST", "sofr": "SOFRRATE Index|PX_LAST", "ff": "FEDL01 Index|PX_LAST"}


def cargar_base(path: str) -> pd.DataFrame:
    """Lee la hoja 'datos' (formato ancho) y devuelve un DataFrame indexado por fecha."""
    df = pd.read_excel(path, sheet_name="datos")
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.sort_values("fecha").set_index("fecha")


def construir_jornadas(df: pd.DataFrame, inicio: str = "2009-01-01") -> pd.DataFrame:
    """Calendario de jornadas USD/CLP con las series de mercado arrastradas (causa: mercado cerrado)."""
    cols = [PX, PX_OPEN, PX_HIGH, PX_LOW] + list(SERIES.values()) + list(RATES.values())
    ff = df[cols].ffill()
    j = ff[df[PX].notna()].copy()
    j = j[j.index >= inicio]
    j["dias"] = j.index.to_series().diff().dt.days.fillna(1.0)   # días corridos desde la jornada anterior
    # Tasa en dólares: SOFR desde que existe (2018-04); Fed funds effective antes (diccionario: "sirve para empalmar")
    usa_sofr = (j.index >= "2018-04-02") & j[RATES["sofr"]].notna()
    j["usd_rate"] = np.where(usa_sofr, j[RATES["sofr"]], j[RATES["ff"]])
    j["clp_rate"] = j[RATES["tpm"]]
    return j


def construir_features(j: pd.DataFrame) -> pd.DataFrame:
    """Features con información ≤ t, objetivo y retorno de t+1, carry y tasa local devengados hasta t+1."""
    f = pd.DataFrame(index=j.index)
    px = j[PX]
    f["r_usd"] = px.pct_change()
    for k in ["cobre", "dxy", "mxn", "brl", "cop", "zar", "cnh", "spx", "xau", "mxef", "bbdxy"]:
        f[f"r_{k}"] = j[SERIES[k]].pct_change()
    for w in (5, 20, 60):
        f[f"r_usd_{w}"] = px.pct_change(w)
    for k, w in [("cobre", 5), ("cobre", 20), ("dxy", 5), ("dxy", 20), ("mxn", 5), ("spx", 5), ("brl", 5)]:
        f[f"r_{k}_{w}"] = j[SERIES[k]].pct_change(w)
    f["dlvix"] = np.log(j[SERIES["vix"]]).diff()
    f["vix"] = j[SERIES["vix"]] / 100
    f["dlmove"] = np.log(j[SERIES["move"]]).diff()
    f["vclp1m"] = j[SERIES["vclp1m"]] / 100
    f["dvclp1m"] = j[SERIES["vclp1m"]].diff() / 100
    f["dvxyem"] = j[SERIES["vxyem"]].diff() / 100
    f["rv20"] = f["r_usd"].rolling(20).std() * np.sqrt(252)
    f["rv5"] = f["r_usd"].rolling(5).std() * np.sqrt(252)
    f["dus2y"] = j[SERIES["us2y"]].diff() / 100
    f["dus2y_5"] = j[SERIES["us2y"]].diff(5) / 100
    f["dus10y"] = j[SERIES["us10y"]].diff() / 100
    f["dswp1"] = j[SERIES["chswp1"]].diff() / 100
    f["dswp2_5"] = j[SERIES["chswp2"]].diff(5) / 100
    f["dif_tasas"] = (j["clp_rate"] - j["usd_rate"]) / 100
    f["basis_ndf"] = (j[SERIES["ndf1m"]] - j["clp_rate"]) / 100
    # intradía de la propia jornada t (cierre vs apertura, ubicación del cierre en el rango, amplitud del rango)
    o, h, l = j[PX_OPEN], j[PX_HIGH], j[PX_LOW]
    f["intra_co"] = px / o - 1
    rng = (h - l)
    f["clv"] = np.where(rng > 0, (px - l) / rng.replace(0, np.nan) - 0.5, 0.0)
    f["rango"] = rng / px
    f["clv_1"] = pd.Series(f["clv"], index=f.index).shift(1)
    # calendario de la PRÓXIMA jornada (conocido de antemano: el calendario de feriados es público). Hipótesis
    # económica: las monedas emergentes se deprecian frente al dólar en el salto de fin de semana (lunes) y, en
    # el caso chileno desde 2020, en la jornada del viernes (reducción de riesgo hacia el fin de semana).
    prox = pd.Series(f.index, index=f.index).shift(-1)
    f["prox_lunes"] = (prox.dt.dayofweek == 0).astype(float)
    f["prox_viernes"] = (prox.dt.dayofweek == 4).astype(float)
    f["prox_mar_jue"] = prox.dt.dayofweek.isin([1, 2, 3]).astype(float)
    f.loc[prox.isna(), ["prox_lunes", "prox_viernes", "prox_mar_jue"]] = np.nan
    # --- lo que ocurre en t+1 (nunca entra como feature) ---
    f["ret_next"] = f["r_usd"].shift(-1)
    f["dias_next"] = j["dias"].shift(-1)
    f["carry_next"] = (j["clp_rate"] - j["usd_rate"]) / 100 / 360 * f["dias_next"]   # por unidad de exposición
    f["cash_next"] = j["clp_rate"] / 100 / 360 * f["dias_next"]                      # tasa local devengada
    f["y"] = np.where(f["ret_next"].isna(), np.nan, (f["ret_next"] > 0).astype(float))
    f["y_reg"] = f["ret_next"] / (f["rv20"] / np.sqrt(252))          # retorno siguiente en unidades de σ diaria (para regresión)
    # horizonte 5: retorno acumulado de t+1..t+5 (hipótesis de señal lenta: momentum / traspaso del cobre)
    f["ret_next5"] = px.shift(-5) / px - 1
    f["y5"] = np.where(f["ret_next5"].isna(), np.nan, (f["ret_next5"] > 0).astype(float))
    return f


FEATURES_BASE = [
    "r_usd", "r_cobre", "r_dxy", "r_mxn", "r_brl", "r_cop", "r_zar", "r_cnh", "r_spx", "r_xau", "r_mxef",
    "r_usd_5", "r_usd_20", "r_usd_60", "r_cobre_5", "r_cobre_20", "r_dxy_5", "r_dxy_20", "r_mxn_5", "r_spx_5", "r_brl_5",
    "dlvix", "vix", "dlmove", "vclp1m", "dvclp1m", "dvxyem", "rv20", "rv5",
    "dus2y", "dus2y_5", "dus10y", "dswp1", "dswp2_5", "dif_tasas", "basis_ndf",
]

FEATURES_INTRA = [  # macro chico + información intradía de la jornada t
    "r_usd", "r_cobre", "r_dxy", "r_mxn", "r_brl", "r_spx", "dlvix", "r_cobre_5", "r_usd_20", "dus2y", "dif_tasas", "vclp1m", "rv20",
    "intra_co", "clv", "rango", "clv_1",
]

FEATURES_CAL = ["prox_lunes", "prox_viernes"]                       # solo calendario
FEATURES_MACRO_CAL = [  # macro chico + calendario de la próxima jornada
    "r_usd", "r_cobre", "r_dxy", "r_mxn", "r_brl", "r_spx", "dlvix", "r_cobre_5", "r_usd_20", "dus2y", "dif_tasas", "vclp1m", "rv20",
    "prox_lunes", "prox_viernes",
]

FEATURES_MACRO = [  # subconjunto "económico" chico, para comparar
    "r_usd", "r_cobre", "r_dxy", "r_mxn", "r_brl", "r_spx", "dlvix", "r_cobre_5", "r_usd_20", "dus2y", "dif_tasas", "vclp1m", "rv20",
]


def dataset(path: str, inicio: str = "2009-01-01"):
    """Atajo: base -> jornadas -> features."""
    j = construir_jornadas(cargar_base(path), inicio)
    return j, construir_features(j)
