# Estado vivo del proyecto

**Actualizado:** 2026-10-03  
**Checkout:** `main` alineado con `origin/main` al iniciar la revisión.  
**Modo de trabajo de este ciclo:** se implementó y ejecutó el notebook 03 hasta las métricas predictivas de validation. No se consultó test ni se calcularon costos.

## PROJECT_STATE

### Objetivo

Comparar reglas de decisión para bloquear transacciones sospechosas del conjunto IEEE-CIS Fraud Detection, priorizando costo económico fuera de muestra sobre una métrica de clasificación aislada.

### Pregunta principal

¿La optimización de la decisión basada en costos mejora el ahorro respecto a un umbral convencional y cómo varía el resultado con distintos costos administrativos de revisión?

### Contexto, fase y avance

- **Observado:** proyecto académico de Ingeniería de Sistemas; documentación y código cubren EDA y partición temporal.
- **Inferido:** EDA y partición se reproducen en este checkout; baseline y sensibilidad hipotética se ejecutaron solo sobre validation. El protocolo económico real y la evaluación de test siguen pendientes.
- **Fase actual:** EDA/partición ejecutadas; notebook de modelado implementado; métricas predictivas de validation calculadas; pendientes los costos aprobados para la etapa económica.
- **Nivel de avance:** etapa de preparación experimental; no se asigna porcentaje porque no hay un plan de entregables cuantificado.

### Arquitectura y flujo

`train_transaction.csv` + `train_identity.csv` → validación y `left join` por `TransactionID` → EDA → partición temporal de 60%/80% con gaps → manifiesto Parquet → preprocesamiento train-only → XGBoost baseline → calibración Platt temprana en validation → evaluación predictiva posterior; costos/test quedan condicionados a aprobación y protocolo congelado.

Los datos se esperan en `data/raw/`; el manifiesto se espera en `data/interim/`. El código reutilizable está en `src/fraud_cost/`; la narración experimental está en `notebooks/`; los informes en `docs/`; los resultados tabulares y figuras en `results/`.

### Datos disponibles

- **Observado:** están disponibles localmente `data/raw/train_transaction.csv` (683.351.067 bytes) y `data/raw/train_identity.csv` (26.529.680 bytes). Git los ignora según la política del proyecto.
- **Observado:** existe `data/interim/split_manifest.parquet` (5.288.710 bytes), regenerado por el notebook y también ignorado por Git.
- **Observado:** están versionadas seis tablas de EDA, tres de partición y cuatro figuras.
- **Dato requerido:** procedencia/hash/versión de los CSV y definición del protocolo de costos para modelado.
- **Entorno ejecutado:** `.venv` usa Python 3.13.2 y contiene las dependencias necesarias. `py -0p` no reporta instalaciones registradas; se inició directamente el ejecutable del entorno. El notebook 02 declara el kernel `ieee-cis-pi2`, que no está registrado; ambos notebooks se ejecutaron con el kernel `python3` del mismo entorno, sin editar los `.ipynb`.

### Código relevante

- `src/fraud_cost/data.py`: carga los dos CSV, valida campos e IDs, realiza `left join` uno-a-uno y resume faltantes.
- `src/fraud_cost/split.py`: asigna, resume y valida los bloques temporales train/gap_1/validation/gap_2/test.
- `src/fraud_cost/paths.py`: centraliza rutas del repositorio.
- `notebooks/01_exploracion_datos.ipynb`: EDA y exportación de resultados.
- `notebooks/02_particion_temporal.ipynb`: compara gaps de 1, 3 y 7 días y guarda el manifiesto.
- `notebooks/03_modelado_y_decisiones_costos.ipynb`: baseline XGBoost, calibración temporal, métricas predictivas y evaluación parametrizada con gates para costos/test.
- `notebooks/04_sensibilidad_hipotetica_validation.ipynb`: sensibilidad hipotética reproducible con scores guardados de validation; no consulta test.
- `scripts/setup.ps1` y `scripts/download_data.ps1`: preparación del entorno y descarga desde Kaggle.
- `scripts/escenarios_hipoteticos_validation.py`: compara umbral fijo y BMR binario en scores guardados de validation con supuestos etiquetados como hipotéticos; no carga test.

### Trabajo completado

- EDA documentado: objetivo, montos, tiempo, ausencias, categorías e identidad.
- Partición temporal definida con 60% para train, límite al 80% para validation y gaps de 7 días.
- Informes y tablas coinciden aritméticamente en conteos, sumas y tasas comprobables entre artefactos.
- Ambos notebooks se ejecutaron exitosamente en orden con los CSV locales; tablas y figuras se regeneraron.
- Se validó que el manifiesto tenga 590.540 filas e IDs únicos, que cubra los cinco bloques y que el orden temporal sea estricto. Las sumas de filas y fraudes coinciden con las tablas.
- Los artefactos regenerados no producen diferencias rastreables en Git frente a los resultados versionados; CSV y Parquet siguen ignorados.
- Agentes Profiler, Auditor y Validador revisaron previamente la situación; un Validador adicional contrastó las métricas del ciclo ejecutado.
- Se creó el notebook 03 con preprocesamiento ajustado solo en train, predictor fijo, calibración Platt en ventana temprana de validation, métricas en la ventana posterior, BMR binario parametrizado, revisión opcional, ahorro frente al umbral fijo y test cerrado por defecto.
- Se creó y ejecutó el notebook 04 para la sensibilidad hipotética; su esquema es válido y sus 24 filas (12 escenarios × 2 políticas) coinciden con el CSV producido por el script.
- Se actualizó el README para incluir los notebooks 03/04 y enlazar el protocolo y análisis.
- Se analizaron 12 escenarios hipotéticos BMR en validation (`λ=0,25/0,50/1,00`; `B=0,25/0,50/1/2 × mediana 65` unidades; `C_block_fraud=0`; sin revisión). BMR redujo el costo proxy entre 14,49% y 52,89% frente al corte 0,5 dentro de esta muestra. Ver `docs/04_sensibilidad_hipotetica_validation.md` y `results/tables/escenarios_hipoteticos_validation.csv`.

### Trabajo parcial

- La descarga está disponible vía Kaggle, pero no se necesitó porque los CSV ya estaban cargados. El setup prescribe Python 3.11; la corrida local usó Python 3.13.2, permitido por el mínimo del `pyproject.toml`, pero no por el comando actual de `setup.ps1`.
- Existe un protocolo de costos/evaluación propuesto en `docs/03_protocolo_costos_y_evaluacion.md`; falta aprobar sus parámetros económicos. El notebook implementa BMR binario y deja revisión optativa con parámetros aparte.
- Los resultados se reprodujeron con los CSV locales, pero no se registró hash o versión de estos archivos.
- El notebook 03 se ejecutó hasta las métricas predictivas; `VALIDATION_CALIBRATION_FRACTION=0.50` y los hiperparámetros XGBoost siguen siendo decisiones de baseline no optimizadas ni comparadas.

### Trabajo pendiente

1. Registrar procedencia/hash de los CSV y decidir el entorno Python oficialmente soportado.
2. Acordar `λ`, costo de bloqueo de legítimas, costo residual de fraude bloqueado y unidad compatible con `TransactionAmt`; revisión puede permanecer deshabilitada.
3. Revisar si los rangos hipotéticos sirven como escenarios exploratorios; sustituirlos por costos aprobados/sustentados y ejecutar una comparación económica separada únicamente en validation.
4. Acordar por separado costo/calidad/capacidad de revisión si se desea habilitarla.
5. Congelar configuración aprobada y reservar test para una sola evaluación final; completar trazabilidad y pruebas citadas.

### Hipótesis abiertas

- Una evaluación cronológica representa mejor el uso futuro que una partición aleatoria, dada la variación temporal de prevalencia observada.
- El monto transaccional puede aproximar exposición, pero no equivale a pérdida bancaria neta.
- La calibración probabilística es necesaria para interpretar Bayes Minimum Risk bajo la matriz de costos elegida.

Estas hipótesis no se consideran demostradas para test/despliegue; existe un baseline evaluado solo en la ventana posterior de validation.

### Experimentos realizados y resultados

- **Ejecutado y observado:** EDA reporta 590.540 filas y 20.663 fraudes (3,499%); cobertura de identidad 24,42%; periodo de aproximadamente 182 días; 414 columnas con faltantes y 12 con más de 90% ausentes.
- **Ejecutado y observado:** medianas de monto 68,50 para legítimas y 75 para fraudulentas en la escala reportada; fraude representa 3,87% del monto total; el 1% de fraudes de mayor monto concentra 10,63% del monto fraudulento. La etiqueta USD de los informes aún requiere confirmación en metadatos/fuente.
- **Ejecutado y observado:** partición de 7 días reporta 380.815 train, 84.093 validation, 84.233 test y 41.399 filas excluidas en gaps; validation tiene 3.075 fraudes y test 2.960.
- **Validado:** manifiesto con 590.540 filas e IDs únicos; bloques completos y en orden estricto. Las tablas regeneradas coinciden con las cifras guardadas.
- **Observado en validation posterior (42.047 filas, 1.517 fraudes):** AP 0,4671; recall fijo 0,5 = 0,2657; precisión = 0,7589; Brier = 0,02518; log-loss = 0,10184; tasa de legítimas bloqueadas = 0,003158.
- **Resultado económico solo hipotético:** BMR redujo el costo proxy entre 14,49% y 52,89% en los 12 escenarios descritos en `docs/04_sensibilidad_hipotetica_validation.md`. No se consideran ahorros reales; parámetros no aprobados. No se evaluaron revisión/árbol ni se consultó test.

### Evidencia clasificada

- **Observado:** el repositorio contiene `data/README.md`, dos informes, tres notebooks, tres módulos Python y tablas/figuras versionadas.
- **Observado:** el informe temporal cita `tests/test_split.py`, pero no existe carpeta `tests/` en el checkout; `pyproject.toml` configura pytest para dicha carpeta.
- **Observado:** el README enlaza el protocolo económico y el notebook 03; los parámetros todavía requieren aprobación del equipo.
- **Observado:** el script de descarga espera dos CSV, pero un mensaje habla de cinco; el setup invoca Python 3.11 pese a anunciar 3.11 o 3.12.
- **Observado:** las salidas guardadas de notebooks incluyen rutas locales de otra ubicación/equipo (`C:\Users\Milton Cuervo\Documents\...`).
- **Inferido:** los resultados actuales son reproducibles con los archivos locales; la procedencia/versionado de los archivos todavía no está documentada.
- **Supuesto:** los archivos locales son los train CSV oficiales de IEEE-CIS, según sus nombres y las comprobaciones realizadas.
- **Dato requerido:** hash/procedencia de los archivos, matriz de costos, costo de revisión y protocolo de calibración.

### Problemas, riesgos y bloqueos

- **Bloqueo principal actual:** solo se tienen escenarios de costos hipotéticos; para conclusiones económicas hacen falta valores sustentados/aprobados. Test requiere además congelar el protocolo.
- El setup del README requiere Python 3.11 y el script invoca `py -3.11`; esta corrida usó Python 3.13.2 del `.venv` existente. Conviene alinear el setup para reproducir en otros equipos.
- El notebook EDA sobrescribe las tablas/figuras versionadas; el notebook de partición también escribe tablas/figura y el manifiesto Parquet. El primer notebook carga y une el dataset ancho completo, con uso de memoria potencialmente alto. Revisar el diff antes de aceptar regeneraciones.
- La ausencia de versión/hash impide asociar los resultados a una copia determinada de datos.
- `validate_temporal_split()` no verifica que los índices de `frame` y `split` coincidan; esto puede volver frágil la validación ante entradas externas desalineadas.
- La propuesta incluye matriz simbólica; el costo de bloquear legítimas, costos/eficacia de revisión, factor de pérdida y unidad monetaria aún no están confirmados.
- Referencias de documentación y pruebas están desalineadas con el árbol del repositorio.

### Decisiones tomadas

- Mantener `left join` para conservar las transacciones etiquetadas.
- Usar evaluación temporal, 60%/80% y gap de 7 días según la documentación existente.
- Reservar test para evaluación económica final y ajustar transformaciones únicamente con train.
- Usar `docs/03_protocolo_costos_y_evaluacion.md` como protocolo; no reportar ahorros numéricos hasta aprobar y completar los parámetros.

### Agentes

Los roles disponibles en el ciclo son: Orquestador, Explorador, Profiler de Datos, Auditor de Código, Analista/Investigador, Experimentador, Validador, Documentador y Coordinador de Resultados. La ejecución actual usó revisión focalizada de Profiler, Auditor y Validador; los agentes son invocados por tarea y no se asume que permanezcan activos entre ciclos.

## Siguiente paso

- **Acción:** revisar los escenarios hipotéticos documentados y, cuando existan, reemplazarlos por `λ`, costo de bloqueo de legítimas, residual de fraude bloqueado y unidad sustentados/aprobados.
- **Agente responsable:** equipo del proyecto para validar supuestos; Experimentador/Validador para comparar resultados.
- **Por qué ahora:** la sensibilidad muestra que las decisiones cambian con la relación entre costos; no establece ahorros reales.
- **Entrada necesaria:** fuentes/costos reales o aprobación formal de un rango hipotético para el informe académico.
- **Resultado esperado:** comparación económica de validation con parámetros trazables, todavía sin revisión si sus costos operativos no están disponibles.
- **Criterio de éxito:** supuestos y unidad documentados; mantener `RUN_FINAL_TEST=False` hasta congelar la política.
- **Decisión que permitirá tomar:** elegir escenarios operativamente defendibles antes de la evaluación final reservada en test.

## Historial de acciones

- **2026-10-03 — Comprender:** inventario de carpetas, documentos, código, configuración, notebooks y artefactos; sin ejecutar código.
- **2026-10-03 — Validar:** agentes Profiler, Auditor y Validador comprobaron existencia de datos, implementación disponible y coherencia de artefactos; reportaron el bloqueo de reproducibilidad y discrepancias documentales.
- **2026-10-03 — Actualizar:** se crea este estado vivo para orientar el siguiente ciclo sin repetir la exploración.
- **2026-10-03 — Intento de ejecución según README:** se confirmó ausencia de Python 3.11 local, entorno virtual, dependencias, CSV originales y autenticación de Kaggle. Se consultó el Python 3.12 empaquetado de Codex y se comprobó que no dispone de las librerías necesarias. No se ejecutaron descargas, instalaciones ni notebooks; no se alteraron los resultados existentes.
- **2026-10-03 — Ejecución según README tras carga de datos:** se ejecutaron el EDA y la partición con `.venv` Python 3.13.2 y kernel `python3`; se regeneraron artefactos y manifiesto. Se validaron cobertura, unicidad y orden estricto. No hubo diferencias rastreables en tablas/figuras versionadas. No se ejecutaron pruebas unitarias ni modelos.
- **2026-10-03 — Diseño metodológico:** se añadió propuesta paramétrica de matriz de costos, BMR, comparación de estrategias, métricas y protocolo temporal en `docs/03_protocolo_costos_y_evaluacion.md`; el README ahora enlaza el documento. La revisión independiente confirmó fórmulas bajo los supuestos declarados y pidió ajustes para compatibilidad de calibración, revisión, árbol y unidades, que fueron incorporados. Los valores económicos siguen pendientes de aprobación humana.
- **2026-10-03 — Implementación del notebook 03:** creado `notebooks/03_modelado_y_decisiones_costos.ipynb` con baseline XGBoost, calibración en ventanas temporales de validation, preprocesamiento ajustado en train, métricas predictivas, políticas/costos parametrizados y gates para costos/test. README y protocolo enlazan el notebook.
- **2026-10-03 — Corrida del usuario revisada:** observadas métricas predictivas de validation posterior y artefactos guardados; no se evaluó revisión ni test.
- **2026-10-03 — Sensibilidad hipotética:** se agregó y ejecutó `scripts/escenarios_hipoteticos_validation.py` y el notebook 04 sobre el Parquet de scores existente. Evaluaron 12 escenarios de BMR binario y umbral fijo, guardaron tabla y gráfico; la salida del notebook coincide con la del script. Se mantuvieron sin cambios la aprobación económica y el gate de test.
