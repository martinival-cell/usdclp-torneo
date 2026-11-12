# -*- coding: utf-8 -*-
"""
Modelo direccional y validación walk-forward.

* Walk-forward con ventana expansiva: para el bloque de jornadas [s, s+step) el modelo se ajusta solo con
  filas k ≤ s-1-embargo (sus etiquetas, que miran a k+1, ya se conocen al cierre de s). Así cada
  probabilidad fuera de muestra se produce con información estrictamente anterior.
* Modelo base: regresión logística regularizada (L2) sobre features estandarizadas; alternativa: boosting
  con regularización fuerte. La elección entre ambos se hace por Brier/log-loss en el período de validación,
  NO por retorno, para no sobreajustar el backtest (López de Prado, AFML cap. 11).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def crear_modelo(tipo: str = "logit", C: float = 0.05, seed: int = 42):
    if tipo == "logit":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=C, max_iter=2000))   # penalización L2 por defecto
    if tipo == "ridge":   # regresión sobre el retorno estandarizado: aprende magnitud, no solo signo
        from sklearn.linear_model import Ridge
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0 / max(C, 1e-6)))
    if tipo == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_depth=3, min_samples_leaf=60,
                                              l2_regularization=1.0, random_state=seed)
    raise ValueError(tipo)


def walk_forward(F: pd.DataFrame, features: list, inicio_eval: str, step: int = 21, embargo: int = 1,
                 tipo: str = "logit", C: float = 0.05, min_train: int = 750, congelar_desde: str | None = None,
                 objetivo: str = "y"):
    """
    Devuelve una Serie con la probabilidad fuera de muestra P(objetivo = 1) para cada jornada ≥ inicio_eval.
    objetivo: "y" (dirección de t+1) o "y5" (dirección del retorno acumulado t+1..t+5); con horizonte h el
    embargo debe ser ≥ h, porque las etiquetas de las últimas h filas aún no se conocen (purga, AFML cap. 7).
    congelar_desde: si se indica, desde esa fecha el modelo NO se reajusta (lectura estricta de las bases:
    "el tramo final que el modelo nunca vio"); el último ajuste usa solo datos anteriores a esa fecha.
    """
    X, y = F[features], F[objetivo]
    idx = F.index
    p = pd.Series(np.nan, index=idx, name="p_up")
    pos_eval = np.where(idx >= pd.Timestamp(inicio_eval))[0]
    if len(pos_eval) == 0:
        return p
    modelo = None
    for s in range(pos_eval[0], len(idx), step):
        e = min(s + step, len(idx))
        congelado = congelar_desde is not None and idx[s] >= pd.Timestamp(congelar_desde) and modelo is not None
        if not congelado:
            corte = s - embargo
            tr = y.iloc[:corte].notna()
            Xtr, ytr = X.iloc[:corte][tr], y.iloc[:corte][tr]
            if len(Xtr) < min_train:
                continue
            modelo = crear_modelo(tipo, C).fit(Xtr, ytr)
        if hasattr(modelo, "predict_proba"):
            p.iloc[s:e] = modelo.predict_proba(X.iloc[s:e])[:, 1]
        else:   # regresor: devuelve μ̂/σ (Sharpe diario esperado); se convierte a pseudo-probabilidad Φ(μ̂/σ)
            from scipy.stats import norm
            p.iloc[s:e] = norm.cdf(modelo.predict(X.iloc[s:e]))
    return p


def recalibrar_rodante(p_raw: pd.Series, y: pd.Series, h: int = 1, step: int = 21, min_obs: int = 500,
                       congelar_desde: str | None = None) -> pd.Series:
    """
    Calibración de Platt rodante: p_cal = sigmoid(a + b·logit(p_raw)), con (a, b) ajustados SOLO sobre
    probabilidades fuera de muestra anteriores cuyas etiquetas ya se conocen (filas ≤ s−1−h). Corrige la
    sobre/infra-confianza del modelo sin usar el futuro; las bases premian la calibración por sobre la precisión.
    Hasta acumular `min_obs` observaciones devuelve NaN (ese tramo no se usa para seleccionar ni evaluar).
    """
    from sklearn.linear_model import LogisticRegression

    idx = p_raw.index
    z = np.log(np.clip(p_raw, 1e-4, 1 - 1e-4) / (1 - np.clip(p_raw, 1e-4, 1 - 1e-4)))
    out = pd.Series(np.nan, index=idx, name="p_cal")
    validos = np.where(p_raw.notna())[0]
    if len(validos) == 0:
        return out
    cal = None
    for s in range(validos[0], len(idx), step):
        e = min(s + step, len(idx))
        congelado = congelar_desde is not None and idx[s] >= pd.Timestamp(congelar_desde) and cal is not None
        if not congelado:
            corte = s - h
            m = p_raw.iloc[:corte].notna() & y.iloc[:corte].notna()
            if m.sum() < min_obs:
                continue
            cal = LogisticRegression(C=1e6, max_iter=1000).fit(z.iloc[:corte][m].values.reshape(-1, 1), y.iloc[:corte][m])
        zb = z.iloc[s:e]
        ok = zb.notna()
        out.iloc[s:e] = np.where(ok, cal.predict_proba(zb.fillna(0).values.reshape(-1, 1))[:, 1], np.nan)
    return out


def ajustar_final(F: pd.DataFrame, features: list, hasta: str | None = None, tipo: str = "logit", C: float = 0.05):
    """Ajusta el modelo con todas las filas con etiqueta conocida hasta `hasta` (para la señal del día)."""
    X, y = F[features], F["y"]
    m = y.notna() if hasta is None else (y.notna() & (F.index <= pd.Timestamp(hasta)))
    return crear_modelo(tipo, C).fit(X[m], y[m])
