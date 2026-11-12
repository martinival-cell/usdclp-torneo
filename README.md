# USD/CLP — Torneo Data Science en Finanzas (FEN · Banco Itaú 2026)

Estrategia direccional del USD/CLP a una jornada con **tamaño de posición calibrado** (−3, −1, 0, 1, 3), evaluada
frente a las tres referencias de las bases, neta de carry y costos, con pruebas de Diebold-Mariano y bootstrap.
Repositorio del participante **Martín Valenzuela**. Licencia MIT.

**Qué hace la estrategia, en una frase:** compra dólar con exposición 1 en las rachas de jornadas con ventaja histórica
—la última jornada antes de un fin de semana o feriado y el lunes siguiente— solo cuando la ventaja esperada acumulada de
la racha cubre con margen el costo de la ida y vuelta (12 pb con los 6 pb por unidad que indicó el laboratorio), y se
queda en pesos, ganando la tasa local, el resto del tiempo. La regla se re-estima cada 21 jornadas con ventana expansiva
y solo con información ya conocida; nunca alcanza ±3 porque la evidencia no lo justifica. Las variables de calendario
son variables adicionales documentadas (notebook §5, informe §3).

## Entrega (jueves 12-11-2026): dónde está cada cosa

| Pedido por el laboratorio | Dónde |
|---|---|
| README con la ejecución paso a paso | Este archivo, sección siguiente |
| Notebook con el resumen del backtesting, gráficos, retornos y estadísticas de trading | [`estrategia_usdclp.ipynb`](estrategia_usdclp.ipynb) (ejecutado: las salidas están guardadas; se recalcula con *Restart & Run All*) |
| Supuestos clave y modelos utilizados (qué hace cada uno y cómo se combinan) | Notebook §2–§3; [`reports/INFORME.md`](reports/INFORME.md) §2–§4 |
| Mecanismo de ejecución (al cierre, qué gatilla cada operación) | Notebook §9; informe §7; sección "Mecanismo de ejecución" de este README |
| Estrategia automatizada (señal del día) | `python scripts/predict_next.py data/base_usdclp.xlsx` |

## Cómo ejecutar el proyecto paso a paso

Requisitos: **Python 3.10 o superior** y git. Probado con Python 3.13, pandas 3.0, scikit-learn 1.9, scipy 1.18,
matplotlib 3.11 (versiones anteriores recientes también funcionan; `requirements.txt` fija mínimos).

**1. Clonar el repositorio**

```bash
git clone https://github.com/<usuario>/usdclp-torneo.git
cd usdclp-torneo
```

**2. Crear un entorno virtual e instalar dependencias**

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(≈ 2 minutos; instala pandas, numpy, scipy, scikit-learn, matplotlib, openpyxl y Jupyter.)

**3. Datos**

La base del torneo `base_usdclp.xlsx` (extracción Bloomberg del laboratorio, 2026-10-07, hoja `datos`) va en
`data/base_usdclp.xlsx`. Está incluida en el repositorio para que pueda ejecutarse sin pasos adicionales; para usar una
versión más reciente basta reemplazar el archivo (mismo formato ancho, columna `fecha`).

**4. Ejecutar el notebook** (recomendado: contiene todo lo pedido en la entrega)

```bash
jupyter notebook estrategia_usdclp.ipynb
```

y en el menú *Kernel ▸ Restart & Run All* (≈ 1–2 minutos). Sin interfaz gráfica, el equivalente por línea de comandos es
`jupyter nbconvert --to notebook --execute --inplace estrategia_usdclp.ipynb`.

**5. Scripts equivalentes** (producen las mismas cifras, gráficos y un JSON con todos los resultados en `reports/`)

```bash
python scripts/run_final.py data/base_usdclp.xlsx reports        # estrategia final, referencias, DM, bootstrap, gráficos (≈ 1 min)
python scripts/run_backtest.py data/base_usdclp.xlsx reports     # protocolo de los 11 modelos de ML, resultado negativo (≈ 1,5 min)
python scripts/predict_next.py data/base_usdclp.xlsx             # señal del día: posición para la jornada siguiente (JSON)
python scripts/predict_next.py data/base_usdclp.xlsx 2026-10-01  # señal como si la base terminara ese día (auditable)
python scripts/predict_next.py data/base_usdclp.xlsx --calendario 2026-11-13 2026-12-01   # proyección del período de monitoreo
python tests/test_no_leakage.py data/base_usdclp.xlsx            # 4 tests de integridad temporal (≈ 30 s)
```

Resultado esperado de las comprobaciones rápidas: `predict_next.py ... 2026-10-01` → posición **+1** (viernes 02-10, inicio
de la racha viernes + lunes); `... 2026-10-05` → posición **0** (martes interior); los cuatro tests imprimen OK.

## Resultados (excedente sobre la tasa en pesos, neto de carry y 6 pb por unidad de cambio de exposición)

| Tramo | Estrategia | Comprar siempre (+1) | No operar | Intensidad constante (±1 siempre) |
|---|---|---|---|---|
| Validación walk-forward 2014-01 → 2025-05 (2.878 jornadas) | **+29,0 %** · Sharpe 0,45 · DD −13,9 % · acierto operado 59 % · 311 operaciones · fuera 89 % | +38,9 % · 0,27 · −33,4 % | 0 % | +45,6 % · 0,32 · −43,5 % · 1.211 operaciones |
| Validación, período activo 2022-05 → 2025-05 (774) | **+26,8 %** · Sharpe 0,81 · DD −13,9 % · acierto 59 % · fuera 60 % | +1,2 % · 0,03 · −33,4 % | 0 % | +51,0 % · 1,05 · −18,7 % |
| Tramo final fuera de muestra 2025-05-30 → 2026-10-06 (341) | **+2,6 %** · Sharpe 0,25 · DD −5,8 % · acierto operado 54 % · 138 operaciones · fuera 60 % | +2,8 % · 0,18 · −14,3 % | 0 % | +3,6 % · 0,23 · −13,7 % · 144 operaciones |

La regla no operó entre 2014 y 2021: con 6 pb un lunes aislado no cubre el umbral de 15 pb por episodio y la jornada
previa al fin de semana solo cruzó los umbrales estadísticos de la ventana expansiva en 2022; por eso se reporta también
el período activo. Ninguna diferencia es estadísticamente significativa con 341 jornadas (Diebold-Mariano p > 0,6;
bootstrap estacionario frente a no operar: IC95 [−9,6 %, +14,6 %], P(>0) = 0,66); en validación, frente a no operar
p = 0,115. Lo consistente es el perfil de riesgo: retorno similar al de comprar siempre con menos de la mitad de la caída
máxima y en el mercado el 40 % de las jornadas. Sensibilidad a costos en el tramo final: 5 pb +3,9 %, 7 pb +1,2 %,
10 pb −3,0 %.

## Por qué una regla de calendario y no un modelo de machine learning

Probamos 11 configuraciones (regresión logística regularizada, boosting, regresión ridge sobre el retorno
estandarizado; con variables macro, intradía y de calendario; horizontes de 1 y 5 jornadas), todas con walk-forward,
purga y calibración de Platt rodante. **Ninguna mostró habilidad probabilística demostrable** frente al pronóstico
ingenuo (Brier ≈ 0,250; AUC 0,47–0,55; Diebold-Mariano p > 0,10). El único efecto robusto en 2010 → 2025-05 es el de
calendario: el del lunes aparece también en USD/MXN, USD/BRL y USD/COP (no en el DXY), el de la jornada previa al fin de
semana es propio del peso chileno desde 2020. Documentar el resultado negativo es parte del entregable.

## Mecanismo de ejecución (replicable por el monitor del laboratorio)

- **Al cierre.** Con la fila t de la base (último cierre `USDCLP Curncy PX_LAST` y los demás cierres fechados t) se decide
  la exposición que rige durante toda la jornada t+1; el resultado de t+1 se mide de cierre a cierre. No se usa la
  apertura ni el dólar observado.
- **Gatillo.** Categoría de calendario de la jornada objetivo (lunes / pre fin de semana / post feriado / interior) y
  tabla de ventajas por categoría estimada con ventana expansiva (re-estimada cada 21 jornadas; activa si |t| > 2 y
  |media| > 5 pb, inactiva si |t| < 1). Se abre +1 al inicio de una racha de jornadas con ventaja si la suma de ventajas
  esperadas ≥ 1,25 × 12 pb = 15 pb; se mantiene durante la racha; se cierra en la primera jornada sin ventaja. Tamaño 1
  (sería 3 solo con Sharpe diario esperado ≥ 0,25, que nunca ocurre).
- **En el período de monitoreo (13-nov → 1-dic, sin feriados chilenos):** +1 en las jornadas de viernes y lunes (compra al
  cierre del jueves, venta al cierre del lunes), 0 de martes a jueves. La posición del viernes 13-nov se decide con el
  cierre del jueves 12-nov. Proyección completa con `predict_next.py --calendario`; la posición vinculante es la señal
  diaria.
- **Contabilización.** 6 pb por unidad de cambio de exposición en la jornada en que rige la nueva exposición; carry
  `pos × (TPM − SOFR)/360 × días corridos`; en feriados chilenos la posición se mantiene y las tasas se devengan por días
  corridos.

## Estructura

```
estrategia_usdclp.ipynb   notebook de entrega: supuestos, modelos, evidencia, backtest, pruebas, lectura de la convicción, mecanismo
src/torneo/data.py        base → calendario de jornadas USD/CLP → features con información ≤ t, objetivo en t+1
src/torneo/calendario.py  estrategia final: categorías de calendario, ventaja esperada expansiva con histéresis, puerta por episodio
src/torneo/sizing.py      reglas de tamaño por valor esperado / Sharpe mínimo (usadas por los modelos ML)
src/torneo/model.py       modelos ML, walk-forward con purga, calibración de Platt rodante (resultado negativo documentado)
src/torneo/evaluate.py    motor de P&L (exposición × ΔS/S − carry − costos), referencias, métricas, sensibilidad a costos
src/torneo/stats.py       Diebold-Mariano (HLN, HAC) y bootstrap estacionario (Politis-Romano)
src/torneo/signal.py      generar_senal(): señal del día; calendario_esperado(): proyección de un período
scripts/run_final.py      evaluación de la estrategia final, referencias, pruebas, gráficos, lectura de la convicción
scripts/run_backtest.py   protocolo completo de los modelos ML (selección en validación, tramo final una vez)
scripts/predict_next.py   señal del día para operar en vivo (y proyección de un período con --calendario)
tests/test_no_leakage.py  futuro barajado, posiciones permitidas, alineación del P&L, features sin futuro
reports/INFORME.md        informe; reports/bitacora_vivo.md: registro del monitoreo; gráficos, CSV y JSON con todas las cifras
data/base_usdclp.xlsx     base del torneo (hoja "datos")
```

## Hipótesis de contabilización (a reconciliar con el código de evaluación oficial)

Fila t decide la posición de t+1 (convención del diccionario del kick-off). P&L aditivo sobre nocional fijo:
`pos·ΔS/S − pos·carry − c·|Δpos|`, con carry = (TPM − SOFR)/360 por días corridos (Fed funds antes de 2018-04),
c = 6 pb por unidad (el laboratorio indicó 5–7 pb; sensibilidad 0–10) y la tasa local devengada como total de la
referencia "no operar". El Sharpe se calcula sobre el excedente respecto de la TPM. Calendario de jornadas USD/CLP; en
feriados chilenos no se opera y la posición se mantiene. Todo está aislado en `src/torneo/evaluate.py` y `src/torneo/data.py`.

## Operación en vivo (13-nov → 1-dic)

Cada jornada, al cierre: actualizar `data/base_usdclp.xlsx` con la versión del laboratorio y ejecutar
`python scripts/predict_next.py data/base_usdclp.xlsx`. La regla se re-estima sola con ventana expansiva. Los feriados
chilenos próximos están en `torneo.signal.FERIADOS_CL` (no hay feriados entre el 13-nov y el 1-dic de 2026; el 26-nov es
feriado en EE.UU. pero el USD/CLP cotiza). Registrar cada señal emitida en `reports/bitacora_vivo.md`: es la lectura de
la convicción en vivo para la final del 2 de diciembre.
