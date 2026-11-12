# Informe — Modelo direccional USD/CLP con tamaño de posición calibrado

Torneo Data Science en Finanzas, FEN · Banco Itaú 2026. Versión 11-11-2026 (entrega 12-11). Participante: Martín Valenzuela.
El notebook `estrategia_usdclp.ipynb` (raíz del repositorio) reproduce todas las cifras de este informe desde la base y
contiene el resumen del backtesting, los gráficos, las estadísticas de trading y el mecanismo de ejecución.

## 1. Resumen ejecutivo

- **Qué predecimos.** La dirección del USD/CLP en la jornada siguiente, con toda la información de cierre de la jornada
  actual (convención del laboratorio: la fila t decide la posición de t+1).
- **Qué encontramos.** Con 11 configuraciones de aprendizaje automático validadas por walk-forward (2018 → 2025-05),
  ninguna muestra habilidad probabilística distinguible del azar (Brier ≈ 0,250, AUC 0,50–0,55, Diebold-Mariano p > 0,10).
  El único patrón robusto es de **calendario**: el USD/CLP sube en promedio +10,8 pb (t = 4,0) en la jornada del lunes y
  +9,3 pb (t = 3,3) en la última jornada antes de un fin de semana o feriado, frente a −3,2 pb (t = −2,0) en las
  jornadas interiores (muestra 2010 → 2025-05-29). El efecto del lunes aparece también en USD/MXN, USD/BRL y USD/COP y no
  en el índice DXY: es una prima de riesgo de fin de semana propia de las monedas emergentes, no un artefacto de la serie.
- **Qué hace la estrategia.** Compra dólar con exposición **1** al inicio de una racha de jornadas con ventaja (típicamente
  la jornada previa al fin de semana más el lunes) solo si la ventaja acumulada de la racha cubre con margen el costo de
  la ida y vuelta (12 pb con el costo de 6 pb que indicó el laboratorio); mantiene la posición hasta que la racha termina
  y se queda en pesos el resto del tiempo. Nunca usa ±3: la evidencia no lo justifica. La regla se re-estima cada 21
  jornadas con ventana expansiva y se desactiva sola si el efecto pierde significancia.
- **Resultado fuera de muestra (2025-05-30 → 2026-10-06, 341 jornadas), neto de carry y de 6 pb por unidad:**
  **+2,6 %** del nocional sobre la tasa en pesos (total 8,9 %), Sharpe 0,25, caída máxima −5,8 %, acierto direccional
  en las jornadas operadas 54 %, en el mercado el 40 % de las jornadas. Referencias con la misma regla: comprar siempre
  +2,8 % (Sharpe 0,18, caída −14,3 %); no operar 0 %; intensidad constante +3,6 % (Sharpe 0,23, caída −13,7 %, rotación
  doble). Validación walk-forward 2014 → 2025-05 (2.878 jornadas): +29,0 %, Sharpe 0,45, caída −13,9 %, acierto operado 59 %.
  **Advertencia de lectura:** la regla no operó entre 2014 y 2021 (con 6 pb un lunes aislado no cubre el umbral de 15 pb
  por episodio y la jornada previa al fin de semana solo cruzó los umbrales estadísticos de la ventana expansiva en
  2022); en su período activo, 2022-05-12 → 2025-05-29 (774 jornadas), rindió +26,8 % con Sharpe 0,81 y caída −13,9 %,
  frente a +1,2 %, 0,03 y −33,4 % de comprar siempre.
- **Significancia.** Ninguna diferencia frente a las referencias es estadísticamente significativa con 341 jornadas
  (Diebold-Mariano p > 0,6; bootstrap estacionario frente a no operar: IC95 [−9,6 %, +14,6 %], P(>0) = 0,66). En
  validación, frente a no operar p = 0,115 (P(>0) = 0,95). No vendemos el tramo final como prueba; vendemos el método y
  el perfil de riesgo: la mitad de la caída máxima de comprar siempre con un retorno similar.

## 2. Datos y tratamiento (respuestas a las seis preguntas del kick-off)

1. **Dólar observado:** el valor publicado en t corresponde a la jornada t−1 (corr(ret_obs[t+1], ret_spot[t]) = 0,74;
   |obs[t+1] − (máx+mín)/2 de t| mediana 0,59 CLP). No se usa como precio de t.
2. **Ventana:** calendario de jornadas USD/CLP desde 2009 para construir features; la regla se evalúa desde 2014, cuando
   acumula historia suficiente. Las series se usan en retornos entre cierres consecutivos de USD/CLP.
3. **Intradía:** en 2020+ el rango tiene contenido (1–5 días/año con máximo = mínimo = cierre); se probaron features
   intradía (cierre/apertura, ubicación del cierre en el rango, amplitud) sin ganancia predictiva.
4. **Quiebres:** WTI con precio negativo (2020-04-20) excluido de transformaciones; `GTCLP5Y/10Y` (tasa en columna de
   precio, genéricos con saltos, rancios), `US0001M` (descontinuado 2024-09) e `IPSA` (rancio) no se usan.
5. **Huecos:** tres causas, tres tratamientos. Mercado cerrado → se arrastra el último cierre sobre el calendario completo y
   luego se toman solo las jornadas USD/CLP (así el retorno de una serie entre dos jornadas es el realmente acumulado).
   Serie inexistente o descontinuada → no se usa. Frecuencia mensual → no se usa en esta versión (solo por FECHA_RELEASE).
   Feriados chilenos: no hay jornada, no se opera, la posición se mantiene y la tasa se devenga por días corridos.
6. **Disponibilidad:** cierres de la fila t disponibles para decidir t+1 (diccionario). SOFR y Fed funds se publican la
   mañana siguiente, antes de la jornada t+1; solo entran en el cálculo del carry.

## 3. Variables adicionales documentadas (bases §3; confirmado por el laboratorio el 11-11-2026)

Además de las series de la base se usan dos variables construidas únicamente con el calendario de la propia base
(las fechas de las jornadas USD/CLP y los saltos entre ellas), por lo que son públicas y conocidas de antemano:

| Variable | Definición | Fuente | Disponibilidad |
|---|---|---|---|
| `prox_lunes` / categoría *lunes* | la jornada t+1 es lunes (primera jornada tras el fin de semana) | fechas de `USDCLP Curncy` en la base | conocida en t (calendario público) |
| categoría *pre fin de semana* | la jornada posterior a t+1 está a ≥ 3 días corridos (viernes, o víspera de feriado) | fechas de `USDCLP Curncy` en la base y calendario de feriados chilenos | conocida en t |
| categoría *post feriado* / *interior* | complemento de las anteriores | ídem | conocida en t |

Hipótesis económica: prima por el riesgo acumulado durante el fin de semana (información que llega con el mercado local
cerrado) y reducción de exposición en monedas emergentes antes de él. Validación cruzada: el efecto del lunes tiene
t = 3,2–3,8 en USD/CLP, USD/MXN, USD/BRL y USD/COP y t = −0,1 en el DXY (2010 → 2026). No se usan datos externos.

## 4. Método

### 4.1 Modelos de aprendizaje automático (resultado negativo, documentado)

Walk-forward con ventana expansiva, reajuste cada 21 jornadas, purga igual al horizonte de la etiqueta y calibración de
Platt rodante con etiquetas ya conocidas. Selección por habilidad probabilística (Brier del modelo − Brier del pronóstico
ingenuo) en validación 2018 → 2025-05-29; el tramo final no participa en la selección.

| Configuración | Brier calibrado | Brier ingenuo | AUC | DM vs ingenuo (p) |
|---|---|---|---|---|
| ridge macro + calendario (h=1) | 0,2483 | 0,2501 | 0,549 | 0,11 |
| logit macro + calendario (h=1) | 0,2490 | 0,2501 | 0,535 | 0,25 |
| ridge solo calendario (h=1) | 0,2492 | 0,2501 | 0,536 | 0,34 |
| ridge macro / ridge macro + intradía (h=1) | 0,2496–0,2497 | 0,2501 | 0,52 | 0,60–0,67 |
| HGB base, logit base, logit macro, logit macro + intradía (h=1) | 0,2501–0,2502 | 0,2501 | 0,50–0,51 | > 0,8 |
| HGB base, logit base (h=5) | 0,2523–0,2532 | 0,2503 | 0,47–0,51 | > 0,3 |

Conclusión: la única información con contenido predictivo medible es la de calendario; el resto de las variables macro
(cobre, DXY, monedas de la región, S&P, VIX, tasas, vol implícita) explica el movimiento contemporáneo (correlaciones de
±0,3–0,4) pero no el de la jornada siguiente (correlaciones de ±0,05 o menos).

### 4.2 Estrategia final: efecto de calendario con tamaño calibrado y puerta económica por episodio

1. Cada jornada t+1 se clasifica con el calendario en *lunes*, *pre fin de semana*, *post feriado* o *interior*.
2. Con ventana expansiva y solo etiquetas conocidas, se estima la media y el estadístico t del retorno de cada categoría.
   Una categoría se activa con |t| > 2 y |media| > 5 pb, y se desactiva si |t| cae bajo 1 (histéresis estadística).
3. **Puerta económica por episodio.** Una posición se abre al inicio de una racha de jornadas consecutivas con ventaja del
   mismo signo solo si la suma de las ventajas esperadas de la racha supera 1,25 veces el costo de la ida y vuelta
   (2 × 6 pb = 12 pb → umbral 15 pb). La racha viernes + lunes (≈ 9 + 11 pb) lo supera; un lunes aislado (11 pb) o la
   racha interior martes–jueves (3 × −3,5 pb) no. Una vez abierta, la posición se mantiene hasta que termina la racha.
4. Tamaño 1; sería 3 solo si el Sharpe diario esperado de la categoría superara 0,25 (el efecto de calendario tiene
   0,11–0,14). La categoría interior (Sharpe ≈ 0,04) queda fuera: la estrategia se abstiene.

Lectura de la convicción en el tramo final: 69 jornadas de lunes con posición +1 (ventaja esperada +9,9 pb, retorno real
medio −1,7 pb); 70 jornadas pre fin de semana, 68 con +1 (esperado +8,6 pb, real +19,2 pb); 200 jornadas interiores y 2
post feriado en 0 (real −4,2 pb). El efecto del lunes no se manifestó en el tramo final y el de pre fin de semana lo superó
con creces: el resultado del tramo depende de un solo leg, lo que es la principal fragilidad de la estrategia.

### 4.3 Motor de evaluación y métricas

`resultado_{t+1} = pos_t · ΔS/S_{t+1} − pos_t · carry_{t+1} − c · |pos_t − pos_{t−1}|`, aditivo sobre nocional fijo;
carry = (TPM − SOFR)/360 por días corridos entre jornadas (Fed funds antes de 2018-04); **c = 6 pb por unidad de cambio
de exposición** (rango 5–7 indicado por el laboratorio; sensibilidad 0–10); la referencia "no operar" rinde la tasa en
pesos (6,4 % en el tramo final). Retorno ajustado por riesgo "mediante la tasa chilena": el Sharpe se calcula sobre el
excedente respecto de la TPM devengada. Métricas reportadas: retorno neto y total, volatilidad anualizada, Sharpe,
Sortino (desviación a la baja con objetivo 0), caída máxima, Calmar, acierto direccional en jornadas operadas (sin
apalancamiento), acierto ponderado por exposición (con apalancamiento), acierto direccional total de la señal sin filtro,
ganancia y pérdida medias, número de operaciones (cambios de exposición) y de episodios (entrada → salida), rotación,
fracción de jornadas fuera del mercado. Pruebas:
Diebold-Mariano con corrección de Harvey-Leybourne-Newbold y varianza Newey-West; bootstrap estacionario de
Politis-Romano (bloques de media 10 jornadas, 5.000 réplicas).

## 5. Resultados (excedente sobre la tasa en pesos, neto de carry y 6 pb por unidad)

| Tramo | Estrategia | Comprar siempre | No operar | Intensidad constante |
|---|---|---|---|---|
| Validación 2014-01 → 2025-05 (2.878) | +29,0 % · vol 5,7 % · Sh 0,45 · DD −13,9 % · acierto operado 59 % · 311 operaciones (156 episodios) · fuera 89 % | +38,9 % · 0,27 · −33,4 % · 50 % | 0 % | +45,6 % · 0,32 · −43,5 % · 53 % · 1.211 operaciones |
| Validación, período activo 2022-05-12 → 2025-05-29 (774) | +26,8 % · vol 10,8 % · Sh 0,81 · DD −13,9 % · acierto operado 59 % · 307 operaciones · fuera 60 % | +1,2 % · 0,03 · −33,4 % | 0 % | +51,0 % · 1,05 · −18,7 % · 322 operaciones |
| Tramo final 2025-05-30 → 2026-10-06 (341) | +2,6 % · vol 7,5 % · Sh 0,25 · DD −5,8 % · acierto operado 54 % · 138 operaciones (69 episodios) · fuera 60 % | +2,8 % · 0,18 · −14,3 % · 50 % | 0 % | +3,6 % · 0,23 · −13,7 % · 54 % · 144 operaciones |

Sensibilidad a costos (tramo final, neto): 0 pb +10,8 %; 2 pb +8,1 %; 5 pb +3,9 %; 6 pb +2,6 %; 7 pb +1,2 %; 10 pb −3,0 %.
Validación: 5 pb +32,1 %; 6 pb +29,0 %; 7 pb +25,9 %; 10 pb +16,6 %. Sensibilidad al margen de la puerta económica
(validación / final): 1,0 → +29,0 % / +2,6 %; 1,25 → +29,0 % / +2,6 %; 1,5 → −3,5 % / +2,6 % (con margen 1,5 el umbral,
18 pb, iguala la ventaja esperada de la racha viernes+lunes, 18–20 pb, y la regla casi deja de operar). Lectura congelada
al 2025-05-29 y walk-forward coinciden en el tramo final. En el período activo la intensidad constante rinde más porque
vende dólar martes a jueves (jornadas que en 2020 → 2025 promediaron −7 pb) con 24 pb de costos semanales en vez de 12 y
mayor caída; la regla no toma esa pata porque la jornada interior no supera los umbrales declarados sobre la muestra
completa (−3 pb, t ≈ −2) y una racha de tres interiores (≈ 12 pb con la media expansiva) no cubre 15 pb.

Figuras: `final_curva_tramo_final.png` (curva neta y exposición, formato de la Figura 1 de las bases),
`final_curva_validacion.png`, `final_evidencia_calendario.png` (medias por categoría y sub-período).

## 6. Riesgos y límites (lo que diremos antes de que lo pregunten)

- El efecto de calendario está documentado en varias monedas emergentes y es económicamente interpretable, pero no hay
  garantía de persistencia: el leg del lunes (el histórico de las monedas emergentes) no apareció en el tramo final
  (−1,7 pb), y el leg de la jornada previa al fin de semana, que sostiene el resultado fuera de muestra (+19 pb), es
  específico del peso chileno y reciente (2020 en adelante; en USD/MXN es negativo). Por eso el tamaño es 1 y la regla
  se desactiva sola si el estadístico t cae bajo 1; la ventana expansiva reacciona con lentitud, en ambos sentidos.
- Con costos realistas (5–7 pb) la ventaja es delgada: ~20 pb esperados por episodio viernes+lunes contra 12 pb de ida y
  vuelta. La estrategia deja de ganar sobre ~8 pb en el tramo final y sobre ~15 pb en validación.
- 341 jornadas no bastan para significancia; el monitoreo en vivo (≈ 13 jornadas, unos 5 episodios) tampoco: la varianza
  de dos semanas es ~20 veces el retorno esperado.
- La referencia de intensidad constante rinde algo más porque también vende dólar en las jornadas interiores; nuestra
  regla no lo hace porque esa racha (3 × −3,5 pb) no cubre el costo de la ida y vuelta. Es una decisión explícita.
- Los parámetros de contabilización no fijados por el laboratorio (fórmula exacta del carry, serie de precio del P&L,
  calendario de feriados, definición de las referencias) son hipótesis; el motor está aislado en `evaluate.py`.
- Si el laboratorio paga el P&L con el dólar observado en vez del cierre, el leg del viernes debe revisarse: en el
  observado el efecto del lunes se mantiene (desplazado al martes de publicación) pero el del viernes se debilita.

## 7. Mecanismo de ejecución (para el monitor del laboratorio, 13-nov → 1-dic)

- **Momento y precio.** La estrategia opera **al cierre**: con la fila t de la base (último cierre `USDCLP Curncy PX_LAST`
  y el resto de los cierres fechados t) se decide la exposición que rige durante toda la jornada t+1, y el resultado de
  t+1 se mide de cierre a cierre, `pos_t × (S_{t+1}/S_t − 1)`. No se usa la apertura ni el dólar observado (el observado
  publicado con fecha t+1 corresponde al promedio de la jornada t).
- **Gatillo.** Solo el calendario de la jornada objetivo y la tabla de ventajas por categoría estimada con información
  pasada (§4.2): se abre +1 al inicio de una racha de jornadas con ventaja (viernes o víspera de feriado seguido de lunes)
  si la suma de ventajas esperadas de la racha es ≥ 15 pb; se mantiene mientras dura la racha; se cierra en la primera
  jornada sin ventaja. Sin stops, sin órdenes intradía, sin otros gatillos.
- **En el período de monitoreo** (sin feriados chilenos): **+1 en las jornadas de viernes y lunes** (compra al cierre del
  jueves, venta al cierre del lunes) y **0 de martes a jueves**. La posición del viernes 13-nov se decide con el cierre
  del jueves 12-nov. Proyección completa: `python scripts/predict_next.py data/base_usdclp.xlsx --calendario 2026-11-13
  2026-12-01`. La tabla de ventajas se re-estima cada 21 jornadas; la posición vinculante es la que emite
  `python scripts/predict_next.py data/base_usdclp.xlsx` cada día con la base actualizada, y se registra en
  `reports/bitacora_vivo.md`.
- **Contabilización.** 6 pb por unidad de cambio de exposición, cargados en la jornada en que rige la nueva exposición
  (12 pb por episodio); carry `pos × (TPM − SOFR)/360 × días corridos` entre t y t+1; en feriados chilenos no hay jornada,
  la posición se mantiene y las tasas se devengan por días corridos.
