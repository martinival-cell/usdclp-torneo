# -*- coding: utf-8 -*-
"""
Señal del día: posición sugerida para la próxima jornada USD/CLP con la información disponible hasta la última
fila de la base (o hasta la fecha indicada).

Uso:
  python scripts/predict_next.py <ruta base_usdclp.xlsx>                 # señal para la jornada siguiente a la última fila
  python scripts/predict_next.py <ruta base_usdclp.xlsx> AAAA-MM-DD      # como si la base terminara en esa fecha
  python scripts/predict_next.py <ruta base_usdclp.xlsx> --calendario 2026-11-13 2026-12-01
                                                                         # proyección de posiciones para un período (monitoreo)

Operación en vivo (13-nov → 1-dic): cada día, al cierre, se reemplaza la base por la versión actualizada del
laboratorio y se ejecuta este script. Los feriados chilenos próximos están en torneo.signal.FERIADOS_CL (no hay
feriados entre el 13-nov y el 1-dic de 2026; el 26-nov es feriado en EE.UU. pero el USD/CLP cotiza).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from torneo.data import cargar_base  # noqa: E402
from torneo.signal import calendario_esperado, generar_senal  # noqa: E402

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)
ruta = sys.argv[1]
base = cargar_base(ruta)
if "--calendario" in sys.argv:
    k = sys.argv.index("--calendario")
    desde, hasta = sys.argv[k + 1], sys.argv[k + 2]
    tabla = calendario_esperado(base, desde, hasta)
    print(f"Proyección de posiciones (tabla de ventajas vigente al {base.index[-1].date()}; la posición vinculante es la señal diaria):")
    print(tabla.to_string(index=False))
else:
    hasta = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(generar_senal(base, hasta=hasta), indent=2, ensure_ascii=False))
