# Optimización basada en costos para el bloqueo de transacciones sospechosas

Proyecto Integrador II — Ingeniería de Sistemas, Universidad de Antioquia, 2026-2.

El proyecto prepara una comparación de tres estrategias sobre probabilidades calibradas de fraude IEEE-CIS: umbral convencional, Bayes Minimum Risk (BMR) y un árbol sensible al costo. Contamos con una línea base calibrada, un análisis exploratorio BMR en validación y una selección de variables/configuración mediante ventanas internas de train. La configuración seleccionada todavía requiere reajuste y un calibrador nuevo. Los costos de negocio no están aprobados y el test temporal no se ha puntuado; por ello no existe todavía una conclusión económica fuera de muestra.

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

   Ejecutar los notebooks en orden. El 05 necesita el manifiesto generado por el 02, pero no reutiliza las probabilidades ni el calibrador del 03.
7. Verificar la lógica reutilizable con `.\.venv\Scripts\python.exe -m pytest -q`. Las versiones utilizadas en la selección quedan registradas en `results/tables/seleccion_temporal_config.json`; `requirements.txt` declara las dependencias, no constituye un lockfile exacto.


## Secuencia experimental

| Notebook | Propósito | Salida esperada |
|---|---|---|
| `01_exploracion_datos.ipynb` | Validar estructura, objetivo, monto, tiempo y ausencia | Evidencia para decisiones de preparación |
| `02_particion_temporal.ipynb` | Congelar train/validación/test y gaps | Manifiesto de particiones |
| `03_modelado_y_decisiones_costos.ipynb` | Baseline calibrado en validation y comparación parametrizada de políticas | Métricas predictivas; comparación con la matriz original; escenarios finales de Ca pendientes |
| `04_sensibilidad_hipotetica_validation.ipynb` | Sensibilidad exploratoria de Ca sobre probabilidades guardadas | Tabla/gráfica hipotéticas; usa `src/fraud_cost/scenarios.py`; mantiene los controles de aprobación y reserva de test |
| `05_preparacion_y_seleccion_temporal.ipynb` | Comparar atributos y cuatro configuraciones mediante dos folds internos de train | Métricas por ventana, diagnóstico de variables, figura y configuración seleccionada; no puntúa validación externa ni test |

Los notebooks presentan el flujo experimental. La carga/unión, la partición, los costos, la ponderación del árbol y la evaluación compartida de escenarios viven en `src/fraud_cost/`; el script y el notebook 04 llaman a la misma función de escenarios. `features.py` construye atributos disponibles antes de cada operación y el preprocesamiento por fold; `selection.py` carga exclusivamente train, define ventanas y registra la comparación. No mantenemos una segunda implementación de esa lógica dentro de las celdas.

El siguiente paso es reajustar la configuración seleccionada con todo train, calibrar ese predictor con validación temprana y diagnosticarlo con validación posterior. Los resultados 03/04 se conservan como línea base histórica; no se mezclan con las métricas internas del 05.

## Reglas metodológicas

- La clasificación y la decisión económica son etapas distintas. El corte clasificatorio de 0.5 no se optimiza en este proyecto.
- El diseño temporal usa un bloque denominado test y el modelado no lo puntúa ni reporta métricas. **Limitación conocida:** el EDA inicial recorrió todo el conjunto etiquetado y la comparación histórica de gaps consultó conteos agregados de fraude del candidato a test; por tanto, este bloque no es un holdout estrictamente intocado. El código actual del notebook 02 ya oculta etiquetas de test en los resúmenes futuros. Consultar `docs/02_particion_temporal.md` antes de describirlo como evaluación independiente.
- No se remuestrea automáticamente: alterar la prevalencia puede deteriorar la interpretación probabilística.
- La selección del 05 usa AP media de ventanas temporales dentro de train. Sus atributos históricos excluyen operaciones simultáneas y congelan el estado de entrenamiento al evaluar. AP interna no demuestra ahorro, calibración ni superioridad estadística.
- La calibración se evalúa y documenta antes de aplicar Bayes Minimum Risk.
- Todas las estrategias se comparan sobre las mismas observaciones y con la misma matriz de costos.
- La métrica principal es costo total/ahorro. AUC-PR, recall y tasa de legítimas bloqueadas explican el comportamiento.
- `sample_weight` es una aproximación sensible al costo, no un sustituto conceptualmente idéntico de un árbol con costos dependientes del ejemplo.

## Trazabilidad

Cada cifra del informe final debe poder rastrearse a versión/hash de datos, partición, configuración, commit, notebook y archivo exportado en `results/`. La matriz del anteproyecto y su implementación están en [`docs/03_protocolo_costos_y_evaluacion.md`](docs/03_protocolo_costos_y_evaluacion.md). Los escenarios en [`docs/04_sensibilidad_hipotetica_validation.md`](docs/04_sensibilidad_hipotetica_validation.md) son retrospectivos e hipotéticos, no costos aprobados. El diseño y la interpretación de la selección están en [`docs/05_preparacion_y_seleccion_temporal.md`](docs/05_preparacion_y_seleccion_temporal.md). El test permanece sin puntuar.
