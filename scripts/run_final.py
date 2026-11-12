# -*- coding: utf-8 -*-
"""
Evaluación de la ESTRATEGIA FINAL (efecto de calendario con tamaño calibrado) con la misma disciplina del backtest:
validación 2014 → 2025-05-29 (la regla adaptativa necesita historia), tramo final 2025-05-30 → 2026-10-06 en dos
lecturas (congelada al 2025-05-29 y walk-forward), referencias de las bases, Diebold-Mariano, bootstrap
estacionario, sensibilidad a costos y a los parámetros de preferencia, y la lectura de la convicción.

Uso: python scripts/run_final.py <ruta base_usdclp.xlsx> [carpeta_salida]
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from torneo.calendario import estrategia_calendario  # noqa: E402
from torneo.data import dataset  # noqa: E402
from torneo.evaluate import pnl, referencias, resumen, tabla_costos  # noqa: E402
from torneo.stats import diebold_mariano, ic_bootstrap  # noqa: E402

RUTA = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(OUT, exist_ok=True)
INICIO_REGLA, INICIO_VAL, FIN_VAL, INICIO_TEST = "2014-01-02", "2014-01-02", "2025-05-29", "2025-05-30"
COSTO = 6.0          # laboratorio (11-11-2026): 5–7 pb por unidad
K1, K3 = 0.0, 0.25   # la puerta económica es por episodio (margen × ida y vuelta); k3 = 0,25 nunca se alcanza

j, F = dataset(RUTA, inicio="2009-01-01")
res = {"regla": "calendario adaptativo (lunes / pre fin de semana) con sizing kappa", "k1": K1, "k3": K3, "costo_pb": COSTO}


def bloque(pos, p_dir, a, b, etiqueta):
    m = (pos.index >= a) & (pos.index <= b) & F["ret_next"].notna()
    est = pnl(pos[m], F, COSTO)
    refs = referencias(F, est.index, p_dir, COSTO)
    direccion = np.sign(2 * p_dir - 1)        # dirección del modelo sin filtro, para el acierto "sin apalancamiento"
    out = {"resumen": {"estrategia": resumen(est, direccion), **{k: resumen(v, direccion) for k, v in refs.items()}}, "DM": {}, "bootstrap": {},
           "sensibilidad_costos": tabla_costos(pos[m], F).to_dict(orient="records"),
           "posiciones": {str(int(k)): int(v) for k, v in pos[m].value_counts().sort_index().items()}}
    for k, v in refs.items():
        out["DM"][k] = diebold_mariano(-est["neto"], -v["neto"])
        out["bootstrap"][k] = ic_bootstrap(est["neto"] - v["neto"], B=5000, bloque=10)
    print(f"\n=== {etiqueta}: {a} → {b} (n={int(m.sum())}) ===")
    print(pd.DataFrame(out["resumen"]).T.round(2).to_string())
    for k in refs:
        d, bs = out["DM"][k], out["bootstrap"][k]
        print(f"  vs {k:22s} DM_HLN {d['DM_HLN']:+.2f} (p={d['p_valor']:.3f}) | dif neta {bs['observado_%']:+.2f}% IC95 [{bs['ic95_%'][0]:+.2f}, {bs['ic95_%'][1]:+.2f}] P(>0)={bs['p_mayor_que_0']:.2f}")
    print("  costos (pb, neto %, sharpe):", [(c["costo_pb"], c["neto_%"], c["sharpe"]) for c in out["sensibilidad_costos"]])
    print("  posiciones:", out["posiciones"])
    return out, est, refs


# ---- walk-forward (regla adaptativa, se re-estima cada 21 jornadas) ----
pos_wf, det_wf = estrategia_calendario(F, INICIO_REGLA, K1, K3, COSTO)
p_dir_wf = (0.5 + np.sign(det_wf["media_bruta_pb"].fillna(0)) * 0.1)   # ref. intensidad constante: dirección del modelo SIN filtro, tamaño 1
res["validacion"], est_v, refs_v = bloque(pos_wf, p_dir_wf, INICIO_VAL, FIN_VAL, "VALIDACIÓN walk-forward")
res["final_walk_forward"], est_f, refs_f = bloque(pos_wf, p_dir_wf, INICIO_TEST, "2030-01-01", "TRAMO FINAL walk-forward")

# ---- lectura congelada al 2025-05-29 ----
pos_cg, det_cg = estrategia_calendario(F, INICIO_REGLA, K1, K3, COSTO, congelar_desde=INICIO_TEST)
p_dir_cg = (0.5 + np.sign(det_cg["media_bruta_pb"].fillna(0)) * 0.1)
res["final_congelado"], est_c, refs_c = bloque(pos_cg, p_dir_cg, INICIO_TEST, "2030-01-01", "TRAMO FINAL congelado")

# ---- sensibilidad a los parámetros de preferencia (validación y final) ----
filas = []
for k1, k3 in [(0.0, 0.25)]:
    for t_min in (1.5, 2.0, 2.5):
      for margen in (1.0, 1.25, 1.5):
        p_, _ = estrategia_calendario(F, INICIO_REGLA, k1, k3, COSTO, t_min=t_min, margen=margen)
        for a, b, nom in [(INICIO_VAL, FIN_VAL, "val"), (INICIO_TEST, "2030-01-01", "final")]:
            m = (p_.index >= a) & (p_.index <= b) & F["ret_next"].notna()
            r = resumen(pnl(p_[m], F, COSTO))
            filas.append({"margen": margen, "t_min": t_min, "tramo": nom, "neto_%": round(r["neto_%"], 2), "sharpe": round(r["sharpe_neto_anual"], 2),
                          "dd_%": round(r["max_drawdown_%"], 2), "fuera_%": round(r["dias_fuera_%"], 1), "turnover": r["turnover_unidades"]})
sens = pd.DataFrame(filas)
res["sensibilidad_parametros"] = sens.to_dict(orient="records")
print("\n=== Sensibilidad a parámetros de preferencia ===")
print(sens.pivot_table(index=["margen", "t_min"], columns="tramo", values=["neto_%", "sharpe", "turnover"]).round(2).to_string())

# ---- lectura de la convicción (entregable 3): por categoría, en el tramo final ----
m = (det_wf.index >= INICIO_TEST) & F["ret_next"].notna()
lec = det_wf[m].assign(ret_next_pb=F.loc[m, "ret_next"] * 1e4)
tabla_lec = lec.groupby("categoria_prox").agg(jornadas=("pos", "size"), pos_media=("pos", "mean"), mu_esperado_pb=("mu_pb", "mean"),
                                              ret_real_pb=("ret_next_pb", "mean"), sharpe_esp=("sharpe_esperado", "mean")).round(2)
print("\n=== Lectura de la convicción (tramo final): qué hizo la regla por tipo de jornada ===")
print(tabla_lec.to_string())
res["lectura_conviccion_final"] = tabla_lec.reset_index().to_dict(orient="records")
det_wf.loc[m].assign(ret_next_pb=F.loc[m, "ret_next"] * 1e4).to_csv(os.path.join(OUT, "final_detalle_diario.csv"))
est_f.to_csv(os.path.join(OUT, "final_pnl_diario.csv"))

# ---- gráficos ----
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def grafico(est, pos, refs, titulo, archivo):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(est.index, est["neto"].cumsum() * 100, lw=2.2, color="C3", label=f"Estrategia (neto {COSTO:.0f} pb)")
    ax[0].plot(est.index, est["con_carry"].cumsum() * 100, lw=1, ls=":", color="C3", label="Estrategia antes de costos")
    for k, v in refs.items():
        ax[0].plot(v.index, v["neto"].cumsum() * 100, lw=1.2, alpha=.85, label=k.replace("_", " "))
    ax[0].axhline(0, color="k", lw=.5); ax[0].set_ylabel("Excedente acumulado sobre tasa local (% nocional)")
    ax[0].set_title(titulo); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].step(pos.index, pos.values, where="post", lw=1, color="C3")
    ax[1].set_yticks([-3, -1, 0, 1, 3]); ax[1].set_ylabel("Exposición"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, archivo), dpi=130); plt.close(fig)


grafico(est_f, pos_wf.loc[est_f.index], refs_f, f"Estrategia final — tramo fuera de muestra {INICIO_TEST} → {F.index[-1].date()}", "final_curva_tramo_final.png")
grafico(est_v, pos_wf.loc[est_v.index], refs_v, f"Estrategia final — validación walk-forward {INICIO_VAL} → {FIN_VAL}", "final_curva_validacion.png")

# evidencia del efecto: media por categoría y sub-período (gráfico de barras)
from torneo.calendario import categorizar  # noqa: E402
cat = categorizar(F)
dfc = pd.DataFrame({"r": F["ret_next"], "c": cat}).dropna()
periodos = [("2010-01-01", "2014-12-31"), ("2015-01-01", "2019-12-31"), ("2020-01-01", "2025-05-29"), ("2025-05-30", "2030-01-01")]
fig, ax = plt.subplots(figsize=(10, 4.5))
w = 0.2
for i, (a, b) in enumerate(periodos):
    s = dfc[(dfc.index >= a) & (dfc.index <= b)]
    g = s.groupby("c")["r"]
    medias = (g.mean() * 1e4).reindex(["lunes", "pre_finde", "interior", "post_feriado"])
    errs = (g.std() / np.sqrt(g.size()) * 1e4).reindex(medias.index)
    ax.bar(np.arange(4) + (i - 1.5) * w, medias.values, w, yerr=errs.values, capsize=2, label=f"{a[:4]}–{b[:4] if b < '2030' else '2026-10'}")
ax.axhline(0, color="k", lw=.6); ax.set_xticks(range(4)); ax.set_xticklabels(["lunes", "pre fin de semana", "interior", "post feriado"])
ax.set_ylabel("retorno medio USD/CLP de la jornada (pb) ± 1 e.e."); ax.set_title("Efecto de calendario por sub-período"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "final_evidencia_calendario.png"), dpi=130); plt.close(fig)

with open(os.path.join(OUT, "final_resumen.json"), "w") as fh:
    json.dump(res, fh, indent=2, default=float)
print(f"\nSalidas en {OUT}")
