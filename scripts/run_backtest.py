# -*- coding: utf-8 -*-
"""
Backtest honesto de extremo a extremo.

Uso:  python scripts/run_backtest.py <ruta base_usdclp.xlsx> [carpeta_salida]

Protocolo (declarado antes de mirar el tramo final):
  1. Probabilidades fuera de muestra por walk-forward desde 2018-01-02 (ventana expansiva, reajuste cada 21
     jornadas, purga = horizonte de la etiqueta) para las configuraciones declaradas abajo. Hipótesis:
     dirección a 1 jornada (clasificación y regresión del retorno estandarizado), con y sin información
     intradía de la jornada t, y dirección del retorno acumulado a 5 jornadas (señal lenta).
  2. Calibración de Platt rodante con etiquetas ya conocidas (las bases premian la calibración).
  3. Selección de la configuración por HABILIDAD PROBABILÍSTICA en VALIDACIÓN (hasta 2025-05-29): Brier del
     modelo calibrado menos Brier del pronóstico ingenuo, con test de Diebold-Mariano. No por retorno.
  4. Sizing por regla de Sharpe esperado mínimo con parámetros de PREFERENCIA fijados de antemano (k1 = 0,10 para
     ±1, k3 = 0,25 para ±3, histéresis por costos con tenencia esperada de 5 jornadas). Se muestra sensibilidad.
  5. El TRAMO FINAL (2025-05-30 → 2026-10-06) se evalúa con la configuración ya fijada, en dos lecturas: modelo y
     calibración congelados al 2025-05-29 (lectura estricta de las bases) y walk-forward (operación en vivo).
  6. Diebold-Mariano (HLN) y bootstrap estacionario contra las tres referencias; sensibilidad a costos.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from torneo.data import FEATURES_BASE, FEATURES_CAL, FEATURES_INTRA, FEATURES_MACRO, FEATURES_MACRO_CAL, dataset  # noqa: E402
from torneo.evaluate import pnl, referencias, resumen, tabla_costos  # noqa: E402
from torneo.model import recalibrar_rodante, walk_forward  # noqa: E402
from torneo.sizing import posiciones_kappa  # noqa: E402
from torneo.stats import brier, diebold_mariano, ic_bootstrap  # noqa: E402

from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402

RUTA = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(OUT, exist_ok=True)

INICIO_EVAL, FIN_VAL, INICIO_TEST = "2018-01-02", "2025-05-29", "2025-05-30"
COSTO_BASE = 6.0   # laboratorio: 5–7 pb por unidad
K1, K3, HOLDING = 0.10, 0.25, 5
SENSIBILIDAD_K = [(0.05, 0.15), (0.10, 0.25), (0.15, 0.35)]
CONFIGS = {
    "h1_logit_macro": dict(tipo="logit", features=FEATURES_MACRO, C=0.1, objetivo="y", calib="y", h=1),
    "h1_logit_intra": dict(tipo="logit", features=FEATURES_INTRA, C=0.1, objetivo="y", calib="y", h=1),
    "h1_logit_base": dict(tipo="logit", features=FEATURES_BASE, C=0.02, objetivo="y", calib="y", h=1),
    "h1_hgb_base": dict(tipo="hgb", features=FEATURES_BASE, C=None, objetivo="y", calib="y", h=1),
    "h1_ridge_macro": dict(tipo="ridge", features=FEATURES_MACRO, C=0.01, objetivo="y_reg", calib="y", h=1),
    "h1_ridge_cal": dict(tipo="ridge", features=FEATURES_CAL, C=1.0, objetivo="y_reg", calib="y", h=1),
    "h1_ridge_macro_cal": dict(tipo="ridge", features=FEATURES_MACRO_CAL, C=0.01, objetivo="y_reg", calib="y", h=1),
    "h1_logit_macro_cal": dict(tipo="logit", features=FEATURES_MACRO_CAL, C=0.1, objetivo="y", calib="y", h=1),
    "h1_ridge_intra": dict(tipo="ridge", features=FEATURES_INTRA, C=0.01, objetivo="y_reg", calib="y", h=1),
    "h5_logit_base": dict(tipo="logit", features=FEATURES_BASE, C=0.02, objetivo="y5", calib="y5", h=5),
    "h5_hgb_base": dict(tipo="hgb", features=FEATURES_BASE, C=None, objetivo="y5", calib="y5", h=5),
}

j, F = dataset(RUTA, inicio="2009-01-01")
F["sigma_d"] = F["rv20"] / np.sqrt(252)
F["p_naive_y"] = F["y"].shift(1).expanding(250).mean()
F["p_naive_y5"] = F["y5"].shift(5).expanding(250).mean()
val = (F.index >= INICIO_EVAL) & (F.index <= FIN_VAL)
tst = (F.index >= INICIO_TEST)
print(f"Jornadas: {len(F)} ({F.index[0].date()} → {F.index[-1].date()}) | validación {int((val & F['y'].notna()).sum())} | tramo final {int((tst & F['y'].notna()).sum())}")


def calidad(p, m, objetivo):
    yv, pv, pn = F.loc[m, objetivo], p[m], F.loc[m, f"p_naive_{objetivo}"]
    dm = diebold_mariano(brier(pv, yv), brier(pn, yv))
    return {"n": int(m.sum()), "brier": brier_score_loss(yv, pv), "brier_naive": brier_score_loss(yv, pn),
            "habilidad": brier_score_loss(yv, pv) - brier_score_loss(yv, pn), "logloss": log_loss(yv, pv),
            "auc": roc_auc_score(yv, pv), "acc": float(((pv > .5) == (yv == 1)).mean()),
            "DM_HLN_vs_naive": dm["DM_HLN"], "p_valor": dm["p_valor"], "sd_p": float(pv.std())}


def posiciones(p, cfgx, m, k1=K1, k3=K3):
    return posiciones_kappa(p[m], F.loc[m, "sigma_d"], F.loc[m, "carry_next"], k1, k3, COSTO_BASE, cfgx["h"], HOLDING)


# ---------------- 1-3. Walk-forward, calibración rodante, selección ----------------
probs_raw, probs, filas = {}, {}, []
for nombre, cfg in CONFIGS.items():
    p_raw = walk_forward(F, cfg["features"], INICIO_EVAL, step=21, embargo=cfg["h"], tipo=cfg["tipo"],
                         C=cfg["C"] or 0.05, objetivo=cfg["objetivo"])
    p = recalibrar_rodante(p_raw, F[cfg["calib"]], h=cfg["h"], step=21, min_obs=500)
    probs_raw[nombre], probs[nombre] = p_raw, p
    m = val & p.notna() & F[cfg["calib"]].notna()
    q, q_raw = calidad(p, m, cfg["calib"]), calidad(p_raw, m, cfg["calib"])
    mv = val & p.notna() & F["y"].notna()
    r = resumen(pnl(posiciones(p, cfg, mv), F, COSTO_BASE))
    filas.append({"config": nombre, **q, "brier_sin_calibrar": q_raw["brier"], "auc_sin_calibrar": q_raw["auc"],
                  "neto_val_%": round(r["neto_%"], 2), "sharpe_val": round(r["sharpe_neto_anual"], 2),
                  "fuera_val_%": round(r["dias_fuera_%"], 1), "turnover_val": r["turnover_unidades"]})
tabla_val = pd.DataFrame(filas).sort_values("habilidad")
tabla_val.to_csv(os.path.join(OUT, "validacion_configs.csv"), index=False)
print("\n=== VALIDACIÓN (calibración rodante): habilidad = Brier − Brier ingenuo (negativo = mejor); sizing k1=0.10/k3=0.25 ===")
print(tabla_val.round(4).to_string(index=False))
elegida = tabla_val.iloc[0]["config"]
cfg = CONFIGS[elegida]
p_sel = probs[elegida]
mejor_p = float(tabla_val.iloc[0]["p_valor"])
print(f"\nConfiguración elegida en validación: {elegida} (objetivo {cfg['objetivo']}, horizonte {cfg['h']}); "
      f"DM vs ingenuo p = {mejor_p:.3f} → {'habilidad probabilística demostrable' if mejor_p < 0.10 else 'SIN habilidad probabilística demostrable'}")

# sensibilidad del sizing en validación (parámetros de preferencia)
mv = val & p_sel.notna() & F["y"].notna()
filas = []
for k1, k3 in SENSIBILIDAD_K:
    r = resumen(pnl(posiciones(p_sel, cfg, mv, k1, k3), F, COSTO_BASE))
    filas.append({"k1": k1, "k3": k3, **{k: round(v, 3) for k, v in r.items()}})
tabla_k = pd.DataFrame(filas)
tabla_k.to_csv(os.path.join(OUT, "validacion_sizing.csv"), index=False)
print("\n=== VALIDACIÓN: sensibilidad de la regla de sizing (costo 2 pb) ===")
print(tabla_k.to_string(index=False))

pos_val = posiciones(p_sel, cfg, mv)
res_val = {"estrategia": resumen(pnl(pos_val, F, COSTO_BASE))}
for k, v in referencias(F, pos_val.index, p_sel, COSTO_BASE).items():
    res_val[k] = resumen(v)
print("\n=== VALIDACIÓN: estrategia vs referencias (neto 2 pb, excedente sobre tasa local) ===")
print(pd.DataFrame(res_val).T.round(2).to_string())

# ---------------- 4-6. Tramo final ----------------
resultados = {"config": elegida, "k1": K1, "k3": K3, "holding": HOLDING, "costo_base_pb": COSTO_BASE,
              "p_valor_habilidad_validacion": mejor_p, "validacion": res_val,
              "tabla_validacion": tabla_val.round(5).to_dict(orient="records"), "tabla_sizing": tabla_k.to_dict(orient="records")}


def evaluar_tramo(p, cfgx):
    m = tst & p.notna() & F["y"].notna()
    pos = posiciones(p, cfgx, m)
    est = pnl(pos, F, COSTO_BASE)
    refs = referencias(F, pos.index, p, COSTO_BASE)
    mq = tst & p.notna() & F[cfgx["calib"]].notna()
    bloque = {"resumen": {"estrategia": resumen(est), **{k: resumen(v) for k, v in refs.items()}},
              "calidad": calidad(p, mq, cfgx["calib"]), "DM_economico": {}, "bootstrap": {},
              "sensibilidad_costos": tabla_costos(pos, F).to_dict(orient="records"),
              "distribucion_posiciones": {str(int(k)): int(v) for k, v in pos.value_counts().sort_index().items()}}
    for k, v in refs.items():
        bloque["DM_economico"][k] = diebold_mariano(-est["neto"], -v["neto"])
        bloque["bootstrap"][k] = ic_bootstrap(est["neto"] - v["neto"], B=5000, bloque=10)
    return bloque, est, pos, refs


p_frozen_raw = walk_forward(F, cfg["features"], INICIO_EVAL, step=21, embargo=cfg["h"], tipo=cfg["tipo"],
                            C=cfg["C"] or 0.05, objetivo=cfg["objetivo"], congelar_desde=INICIO_TEST)
p_frozen = recalibrar_rodante(p_frozen_raw, F[cfg["calib"]], h=cfg["h"], step=21, min_obs=500, congelar_desde=INICIO_TEST)
for lectura, p in [("congelado", p_frozen), ("walk_forward", p_sel)]:
    bloque, est, pos, refs = evaluar_tramo(p, cfg)
    resultados[f"final_{lectura}"] = bloque
    est.assign(p_up=p[est.index].values).to_csv(os.path.join(OUT, f"diario_final_{lectura}.csv"))
    print(f"\n=== TRAMO FINAL {INICIO_TEST} → {F.index[-1].date()} | {elegida} | lectura: {lectura} ===")
    print(pd.DataFrame(bloque["resumen"]).T.round(2).to_string())
    print("calidad:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in bloque["calidad"].items()})
    for k in refs:
        d, b = bloque["DM_economico"][k], bloque["bootstrap"][k]
        print(f"  vs {k:22s} DM_HLN {d['DM_HLN']:+.2f} (p={d['p_valor']:.3f}) | dif neta {b['observado_%']:+.2f}% IC95 [{b['ic95_%'][0]:+.2f}, {b['ic95_%'][1]:+.2f}] P(>0)={b['p_mayor_que_0']:.2f}")
    print("sensibilidad costos:", bloque["sensibilidad_costos"])
    print("posiciones:", bloque["distribucion_posiciones"])

# dispersión informativa: todas las configuraciones en el tramo final (walk-forward, misma regla de sizing)
filas = []
for nombre, p in probs.items():
    c2 = CONFIGS[nombre]
    m = tst & p.notna() & F["y"].notna()
    r = resumen(pnl(posiciones(p, c2, m), F, COSTO_BASE))
    filas.append({"config": nombre, "neto_%": round(r["neto_%"], 2), "sharpe": round(r["sharpe_neto_anual"], 2),
                  "dias_fuera_%": round(r["dias_fuera_%"], 1), "turnover": r["turnover_unidades"]})
tabla_disp = pd.DataFrame(filas)
resultados["dispersion_final_todas_configs"] = tabla_disp.to_dict(orient="records")
print("\n=== Dispersión (informativo): todas las configuraciones en el tramo final, misma regla de sizing ===")
print(tabla_disp.to_string(index=False))

# ---------------- Gráficos ----------------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def grafico(est, pos, refs, titulo, archivo):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(est.index, est["neto"].cumsum() * 100, lw=2.2, color="C3", label=f"Estrategia (neto {COSTO_BASE:.0f} pb)")
    ax[0].plot(est.index, est["con_carry"].cumsum() * 100, lw=1, ls=":", color="C3", label="Estrategia antes de costos")
    for k, v in refs.items():
        ax[0].plot(v.index, v["neto"].cumsum() * 100, lw=1.2, alpha=.85, label=k.replace("_", " "))
    ax[0].axhline(0, color="k", lw=.5); ax[0].set_ylabel("Excedente acumulado sobre tasa local (% nocional)")
    ax[0].set_title(titulo); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].step(pos.index, pos.values, where="post", lw=1, color="C3")
    ax[1].set_yticks([-3, -1, 0, 1, 3]); ax[1].set_ylabel("Exposición"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, archivo), dpi=130); plt.close(fig)


for lectura, p in [("congelado", p_frozen), ("walk_forward", p_sel)]:
    bloque, est, pos, refs = evaluar_tramo(p, cfg)
    grafico(est, pos, refs, f"Tramo final fuera de muestra {INICIO_TEST} → {F.index[-1].date()} — {elegida}, lectura {lectura}",
            f"curva_final_{lectura}.png")
grafico(pnl(pos_val, F, COSTO_BASE), pos_val, referencias(F, pos_val.index, p_sel, COSTO_BASE),
        f"Validación walk-forward → {FIN_VAL} — {elegida}", "curva_validacion.png")

mq = val & p_sel.notna() & F[cfg["calib"]].notna()
bins = pd.qcut(p_sel[mq], 10, duplicates="drop")
cal = pd.DataFrame({"p": p_sel[mq], "y": F.loc[mq, cfg["calib"]]}).groupby(bins, observed=True).mean()
raw = probs_raw[elegida]
bins_r = pd.qcut(raw[mq], 10, duplicates="drop")
cal_r = pd.DataFrame({"p": raw[mq], "y": F.loc[mq, cfg["calib"]]}).groupby(bins_r, observed=True).mean()
fig, ax = plt.subplots(figsize=(5.5, 5))
lo, hi = min(cal["p"].min(), cal_r["p"].min()) - .02, max(cal["p"].max(), cal_r["p"].max()) + .02
ax.plot([lo, hi], [lo, hi], "k--", lw=.8, label="calibración perfecta")
ax.plot(cal_r["p"], cal_r["y"], "s-", alpha=.6, label="sin calibrar")
ax.plot(cal["p"], cal["y"], "o-", label="calibrado (Platt rodante)")
ax.set_xlabel("probabilidad predicha (deciles)"); ax.set_ylabel("frecuencia real de alza")
ax.set_title(f"Confiabilidad en validación — {elegida}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "confiabilidad_validacion.png"), dpi=130); plt.close(fig)

with open(os.path.join(OUT, "backtest_resumen.json"), "w") as fh:
    json.dump(resultados, fh, indent=2, default=float)
pd.DataFrame({k: v for k, v in probs.items()} | {f"{k}_raw": v for k, v in probs_raw.items()} | {"p_congelado_elegida": p_frozen}).to_csv(os.path.join(OUT, "probabilidades_oos.csv"))
print(f"\nSalidas en {OUT}")
