# Optimización basada en costos para el bloqueo de transacciones sospechosas

Proyecto Integrador II — Ingeniería de Sistemas, Universidad de Antioquia, 2026-2.

El proyecto compara tres políticas bajo una misma matriz: umbral convencional, Bayes Minimum Risk (BMR) y un árbol sensible al costo. Las dos primeras usan probabilidades calibradas de XGBoost; el árbol aprende acciones con atributos y costos de train. Contamos con una línea base, selección temporal, reajuste/calibración, comparación económica exploratoria, diagnóstico probabilístico/temporal y un experimento Platt/isotónica. Isotónica permanece como candidata y no sustituye los resultados anteriores. Los costos de negocio y el calibrador final no están aprobados. El test temporal no se ha puntuado; por ello no existe todavía una conclusión económica final sobre ese bloque.

Aplicamos la matriz del anteproyecto: intervenir (bloquear y revisar) cuesta `Ca` tanto en fraude como en operaciones legítimas; aprobar fraude cuesta su monto y aprobar legítimas cuesta cero. BMR interviene cuando `p_i × TransactionAmt_i > Ca`. Calculamos ahorro frente al menor costo entre aprobar todo e intervenir todo y, por separado, mejora frente a la política fija.

## Pregunta de investigación

¿En qué medida la optimización de la regla de decisión basada en costos permite mejorar el ahorro financiero frente a un umbral fijo convencional para el bloqueo de transacciones sospechosas en el conjunto de datos IEEE-CIS, y cómo varía este beneficio ante diferentes costos administrativos de revisión?

## Estructura

```text
.
├── data/
│   ├── raw/          # Archivos originales, inmutables y no versionados
│   ├── interim/      # Datos intermedios reproducibles
│   └── processed/    # Particiones listas para modelar
├── notebooks/        # Relato experimental breve y numerado
├── src/fraud_cost/   # Lógica reutilizable y comprobable
├── docs/             # Decisiones, protocolo y preparación de sustentaciones
├── results/
│   ├── figures/      # Figuras finales exportadas
│   ├── tables/       # Métricas y tablas finales
│   └── models/       # Modelos y calibradores serializados
└── requirements.txt
```

Los CSV originales y artefactos intermedios/modelos se mantienen localmente e ignorados por Git. Las tablas y figuras seleccionadas de `results/` sí están versionadas como evidencia de las corridas documentadas; al regenerarlas, revisar el diff antes de confirmarlo.

## Preparación reproducible

1. Desde la raíz, ejecutar `scripts/setup.ps1`, que crea `.venv` con Python 3.12 o, si no está disponible, Python 3.11; instala dependencias, el paquete local en modo editable y el kernel `ieee-cis-pi2`. Comprueba si un entorno existente funciona antes de reutilizarlo. Si no funciona, lo conserva en una carpeta de respaldo y crea otro; no borra el anterior. Puede indicarse un intérprete explícito con `-PythonExecutable`.
2. Si se requiere reinstalar manualmente, usar el intérprete del entorno:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m pip install -e .
   ```

3. Descargar desde Kaggle los archivos de la competencia IEEE-CIS Fraud Detection y ubicarlos sin modificar en `data/raw/`:

   - `train_transaction.csv`
   - `train_identity.csv`

4. No escribir credenciales en notebooks. Configurar la API de Kaggle fuera del repositorio siguiendo su documentación oficial o descargar los datos manualmente.
5. Registrar hash SHA-256 de los CSV, versiones de dependencias y commit con `.\.venv\Scripts\python.exe scripts\data_provenance.py`. El manifiesto local se escribe en `data/interim/data_provenance.json` y no se versiona.
6. Abrir Jupyter desde la raíz y seleccionar el kernel `ieee-cis-pi2`:

   ```powershell
   .\.venv\Scripts\python.exe -m jupyter lab
   ```

   Ejecutar los notebooks en orden. El 05 necesita el manifiesto generado por el 02, pero no reutiliza las probabilidades ni el calibrador del 03. El 06 necesita la selección del 05 y verifica sus hashes antes de entrenar. El 07 verifica fuentes y artefactos del 06, reutiliza su preprocesamiento y ajusta solo los árboles por Ca. El 08 verifica fuentes y artefactos del 06/07 y reutiliza sus scores/acciones; no entrena ni carga modelos, y no puntúa otros bloques. No basta con las tablas agregadas para regenerarlo: necesita los dos Parquet locales de esas etapas. El 09 verifica la selección/reajuste, carga el predictor y los scores del 06 y ajusta únicamente un candidato isotónico con validación temprana; no reajusta XGBoost/Platt ni sobrescribe sus artefactos. Requiere los datos originales y artefactos locales del 02/05/06, no solo las tablas agregadas.
7. Verificar la lógica reutilizable con `.\.venv\Scripts\python.exe -m pytest -q`. Las versiones utilizadas quedan registradas en `results/tables/seleccion_temporal_config.json`, `results/tables/reajuste_config.json`, `results/tables/comparacion_economica_config.json`, `results/tables/diagnostico_estabilidad_config.json` y `results/tables/comparacion_calibradores_config.json`; `requirements.txt` declara las dependencias, no constituye un lockfile exacto.


## Secuencia experimental

| Notebook | Propósito | Salida esperada |
|---|---|---|
| `01_exploracion_datos.ipynb` | Validar estructura, objetivo, monto, tiempo y ausencia | Evidencia para decisiones de preparación |
| `02_particion_temporal.ipynb` | Congelar train/validación/test y gaps | Manifiesto de particiones |
| `03_modelado_y_decisiones_costos.ipynb` | Baseline calibrado en validation y comparación parametrizada de políticas | Métricas predictivas; comparación con la matriz original; escenarios finales de Ca pendientes |
| `04_sensibilidad_hipotetica_validation.ipynb` | Sensibilidad exploratoria de Ca sobre probabilidades guardadas | Tabla/gráfica hipotéticas; usa `src/fraud_cost/scenarios.py`; mantiene los controles de aprobación y reserva de test |
| `05_preparacion_y_seleccion_temporal.ipynb` | Comparar atributos y cuatro configuraciones mediante dos folds internos de train | Métricas por ventana, diagnóstico de variables, figura y configuración seleccionada; no puntúa validación externa ni test |
| `06_reajuste_y_calibracion.ipynb` | Reajustar la selección con todo train, calibrar en validación temprana y diagnosticar en posterior | Métricas raw/calibradas, curva y grupos de calibración, predictor completo y scores nuevos; sin evaluación económica ni test |
| `07_comparacion_economica_validation.ipynb` | Comparar política fija, BMR y árbol ponderado por Ca ilustrativo | Costos, referencias, compromisos de acción y sensibilidad a calibración; escenarios no aprobados y test sin puntuar |
| `08_diagnostico_y_estabilidad.ipynb` | Examinar calibración por monto/riesgo, frontera BMR y cuatro subperiodos posteriores | Cuatro tablas, dos figuras y configuración; sin nuevos ajustes, selección de calibrador ni evaluación de test |
| `09_comparacion_calibradores.ipynb` | Contrastar raw, Platt e isotónica manteniendo el XGBoost del 06 y fundamentar Ca con literatura | Cuatro tablas, dos figuras, configuración y candidato isotónico local; sin selección automática, reemplazo de la comparación principal ni evaluación de test |

Los notebooks presentan el flujo experimental. La carga/unión, la partición, los costos, la ponderación del árbol y la evaluación compartida de escenarios viven en `src/fraud_cost/`; el script y el notebook 04 llaman a la misma función de escenarios. `features.py` construye atributos disponibles antes de cada operación y el preprocesamiento por fold; `selection.py` registra la selección dentro de train; `refit.py` reajusta y calibra. `economics.py` verifica esos artefactos y compara acciones mediante las mismas funciones de costos, sin duplicar la matriz. `diagnostics.py` describe probabilidades y acciones guardadas por grupos y subperiodos. `calibration.py` contrasta calibradores sobre el mismo predictor/población y conserva el candidato vinculado al modelo original. No mantenemos una segunda implementación de esa lógica dentro de las celdas.

Antes de evaluar test necesitamos acordar escenarios finales de Ca y unidades, discutir calibración y congelar el protocolo. Los resultados 03/04 se conservan como línea base histórica. El 04 sigue utilizando `validation_scores.parquet`; el 07 usa `validation_scores_selected.parquet` y `selected_xgboost_platt.joblib` del 06. Árboles y acciones del 07 se guardan en `results/models/cost_trees_hypothetical.joblib` y `data/interim/decisiones_economicas_validation.parquet`. Todos esos artefactos por operación/modelo son locales e ignorados por Git; tablas agregadas, figura y configuración sí se versionan.

El 09 guarda `results/models/candidate_isotonic_calibrator.joblib`, también local e ignorado, sin sustituir el predictor Platt. El candidato recibe probabilidades raw, no logit, y no comparte directamente la interfaz del calibrador del 06. Sus Ca=1; 2,5; 5; 10 se inspiran numéricamente en literatura de otros datos/empresas; 20; 50; 100 amplían la sensibilidad. No trasladamos euros al dataset ni presentamos costos reales sin moneda verificada. El nuevo Ca=2,5 solo pertenece al diagnóstico del 09 y no crea un árbol adicional ni reescribe la rejilla histórica del 07/08.

## Reglas metodológicas

- La clasificación y la decisión económica son etapas distintas. El corte clasificatorio de 0.5 no se optimiza en este proyecto.
- El diseño temporal usa un bloque denominado test y el modelado no lo puntúa ni reporta métricas. **Limitación conocida:** el EDA inicial recorrió todo el conjunto etiquetado y la comparación histórica de gaps consultó conteos agregados de fraude del candidato a test; por tanto, este bloque no es un holdout estrictamente intocado. El código actual del notebook 02 ya oculta etiquetas de test en los resúmenes futuros. Consultar `docs/02_particion_temporal.md` antes de describirlo como evaluación independiente.
- No se remuestrea automáticamente: alterar la prevalencia puede deteriorar la interpretación probabilística.
- La selección del 05 usa AP media de ventanas temporales dentro de train. Sus atributos históricos excluyen operaciones simultáneas y congelan el estado de entrenamiento al evaluar. AP interna no demuestra ahorro, calibración ni superioridad estadística.
- La calibración se evalúa y documenta antes de aplicar Bayes Minimum Risk.
- El calibrador pertenece al predictor que generó sus puntuaciones. Después de calibrar no se reajusta XGBoost ni se actualiza el historial con validation. Las dos mitades de validation no tienen un gap adicional y la posterior sigue siendo desarrollo, no test.
- Todas las estrategias se comparan sobre las mismas observaciones y con la misma matriz de costos.
- La comparación del 07 es hipotética: Ca no se elige por el ahorro obtenido. El diagnóstico BMR raw/calibrado no selecciona retrospectivamente un calibrador. Minimizar riesgo esperado bajo las probabilidades del modelo no garantiza minimizar costo realizado.
- El 08 es descriptivo: bandas fijas, mismos grupos para probabilidades raw/calibradas y subperiodos de igual duración sin nuevos ajustes. Los grupos pequeños conservan sus conteos; no hay inferencia de superioridad. Los mínimos de referencias triviales por periodo no se suman como referencia global.
- El 09 ajusta isotónica únicamente en validación temprana, motivado por diagnósticos previos sobre la posterior; esta última sigue siendo desarrollo reutilizado. Leemos log-loss, Brier, AP, grupos y consecuencias BMR sin selección automática ni por Ca. La referencia fija conserva Platt. Documentamos empates y probabilidades extremas sin suavizar retrospectivamente para mejorar resultados. El calibrador y la rejilla finales siguen pendientes de aprobación.
- La métrica principal es costo total/ahorro. AUC-PR, recall y tasa de legítimas bloqueadas explican el comportamiento.
- `sample_weight` es una aproximación sensible al costo, no un sustituto conceptualmente idéntico de un árbol con costos dependientes del ejemplo.

## Trazabilidad

Cada cifra del informe final debe poder rastrearse a versión/hash de datos, partición, configuración, commit, notebook y archivo exportado en `results/`. La matriz está en [`docs/03_protocolo_costos_y_evaluacion.md`](docs/03_protocolo_costos_y_evaluacion.md), la referencia económica en [`docs/04_sensibilidad_hipotetica_validation.md`](docs/04_sensibilidad_hipotetica_validation.md), la selección en [`docs/05_preparacion_y_seleccion_temporal.md`](docs/05_preparacion_y_seleccion_temporal.md) y el reajuste en [`docs/06_reajuste_y_calibracion.md`](docs/06_reajuste_y_calibracion.md). La comparación de tres políticas y su diagnóstico de calibración están en [`docs/07_comparacion_economica_validation.md`](docs/07_comparacion_economica_validation.md). La calibración condicional, frontera económica y estabilidad temporal se documentan en [`docs/08_diagnostico_y_estabilidad.md`](docs/08_diagnostico_y_estabilidad.md). La comparación Platt/isotónica y el alcance de los antecedentes de Ca se desarrollan en [`docs/09_comparacion_calibradores_y_costos.md`](docs/09_comparacion_calibradores_y_costos.md). El test permanece sin puntuar.
