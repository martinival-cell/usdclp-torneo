# -*- coding: utf-8 -*-
"""
De probabilidad a posición {-3, -1, 0, 1, 3}.

Regla de valor esperado neto (la que premian las bases: "discriminar cuándo tomar posición y cuándo abstenerse"):

    EV(s) = s · μ̂ − s · carry − c · |s − s_prev| − λ · s² · σ̂²

    μ̂     = (2p̂ − 1) · E|r|      retorno esperado por unidad larga; E|r| ≈ σ̂·sqrt(2/π) con σ̂ = vol realizada 20d diaria
    carry = carry devengado por unidad hasta la próxima jornada (el largo paga, el corto recibe)
    c     = costo por unidad de cambio de exposición (hipótesis base 2 pb)
    λ     = aversión al riesgo; único parámetro libre, se fija en validación

Se elige el s de la grilla que maximiza EV. El término de costo produce histéresis: no se cambia de posición
por una diferencia de convicción menor que el costo de hacerlo. Con p̂ ≈ 0,5 la regla devuelve 0.
"""
import numpy as np
import pandas as pd

GRILLA = np.array([-3, -1, 0, 1, 3])


def posiciones_ev(p: pd.Series, sigma_d: pd.Series, carry: pd.Series, lam: float, costo_bp: float = 2.0,
                  grilla=GRILLA, horizonte: int = 1) -> pd.Series:
    """
    Posición para la jornada siguiente a cada fecha del índice (secuencial, por la histéresis).
    horizonte h: si p es la probabilidad de alza del retorno acumulado a h jornadas, la ventaja esperada por
    jornada es (2p−1)·E|r_h|/h con E|r_h| ≈ σ_d·sqrt(h)·sqrt(2/π); el riesgo sigue siendo diario (σ_d²).
    """
    c = costo_bp / 1e4
    mu = (2 * p.values - 1) * sigma_d.values * np.sqrt(horizonte) * np.sqrt(2 / np.pi) / horizonte
    var = sigma_d.values ** 2
    car = np.nan_to_num(carry.values, nan=0.0)
    out = np.zeros(len(p))
    prev = 0.0
    for i in range(len(p)):
        if np.isnan(mu[i]) or np.isnan(var[i]):
            out[i] = 0.0
            prev = 0.0
            continue
        ev = grilla * mu[i] - grilla * car[i] - c * np.abs(grilla - prev) - lam * grilla ** 2 * var[i]
        s = grilla[int(np.argmax(ev))]
        out[i] = s
        prev = s
    return pd.Series(out, index=p.index, name="pos")


def posiciones_kappa(p: pd.Series, sigma_d: pd.Series, carry: pd.Series, k1: float = 0.10, k3: float = 0.25,
                     costo_bp: float = 2.0, horizonte: int = 1, holding: int = 5, grilla=GRILLA,
                     sigma_sr: pd.Series | None = None) -> pd.Series:
    """
    Regla de "Sharpe esperado mínimo" (parámetros de preferencia, no ajustados a los datos):
      SR_long  = (μ̂ − carry)/σ_sr ,  SR_short = (−μ̂ + carry)/σ_sr      (Sharpe diario esperado de la apuesta)
      |posición| = 3 si SR ≥ k3 ; 1 si SR ≥ k1 ; 0 en otro caso.
    σ_sr es la desviación con que se mide la ventaja (por defecto la vol realizada 20d; para efectos de calendario,
    la desviación histórica de la propia categoría, para no apalancar en períodos de calma sin evidencia).
    Histéresis por costos SOLO entre tamaños del mismo signo: pasar de ±1 a ±3 (o al revés) exige que la mejora
    esperada en `holding` jornadas supere el costo del cambio. Si la ventaja desaparece (objetivo 0) se cierra
    siempre: mantener riesgo sin ventaja esperada no se justifica. Con probabilidades ≈ 0,5 la regla devuelve 0.
    """
    c = costo_bp / 1e4
    mu = (2 * p.values - 1) * sigma_d.values * np.sqrt(horizonte) * np.sqrt(2 / np.pi) / horizonte
    sig = sigma_d.values if sigma_sr is None else sigma_sr.values
    car = np.nan_to_num(carry.values, nan=0.0)
    out = np.zeros(len(p))
    prev = 0.0
    for i in range(len(p)):
        if np.isnan(mu[i]) or np.isnan(sig[i]) or sig[i] <= 0:
            out[i] = 0.0
            prev = 0.0
            continue
        sr_long = (mu[i] - car[i]) / sig[i]
        sr_short = (-mu[i] + car[i]) / sig[i]
        if sr_long >= k3:
            objetivo = 3
        elif sr_long >= k1:
            objetivo = 1
        elif sr_short >= k3:
            objetivo = -3
        elif sr_short >= k1:
            objetivo = -1
        else:
            objetivo = 0
        if objetivo != 0 and prev != 0 and np.sign(objetivo) == np.sign(prev) and objetivo != prev:
            ganancia = holding * (objetivo - prev) * (mu[i] - car[i])      # mejora esperada de P&L en el horizonte de tenencia
            if ganancia <= c * abs(objetivo - prev):
                objetivo = prev                                           # no paga el costo de cambiar de tamaño
        out[i] = objetivo
        prev = objetivo
    return pd.Series(out, index=p.index, name="pos")


def posiciones_umbral(p: pd.Series, a: float, b: float) -> pd.Series:
    """Regla simple por bandas sobre |2p−1| fijadas en entrenamiento: 0 si < a, ±1 si en [a,b), ±3 si ≥ b."""
    edge = 2 * p - 1
    s = np.where(edge.abs() >= b, 3, np.where(edge.abs() >= a, 1, 0)) * np.sign(edge)
    return pd.Series(np.nan_to_num(s), index=p.index, name="pos")


def intensidad_constante(p: pd.Series) -> pd.Series:
    """Referencia de las bases 'operar siempre con exposición uno': la dirección del propio modelo, tamaño 1."""
    s = np.sign(2 * p - 1)
    return pd.Series(np.nan_to_num(s), index=p.index, name="pos_ref1")
