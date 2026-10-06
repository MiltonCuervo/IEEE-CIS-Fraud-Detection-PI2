# Optimización basada en costos para el bloqueo de transacciones sospechosas

Proyecto Integrador II — Ingeniería de Sistemas, Universidad de Antioquia, 2026-2.

El proyecto prepara una comparación de tres estrategias sobre probabilidades calibradas de fraude IEEE-CIS: umbral convencional, Bayes Minimum Risk (BMR) y un árbol sensible al costo. Actualmente contamos con un modelo predictivo de referencia y un análisis exploratorio BMR en validación. Los costos de negocio no están aprobados y el test temporal no se ha puntuado; por ello no existe todavía una conclusión económica fuera de muestra.

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

1. Ejecutar `scripts/setup.ps1`, que crea `.venv` con Python 3.12 o, si no está disponible, Python 3.11; instala dependencias y el paquete local en modo editable.
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
6. Abrir Jupyter desde la raíz del repositorio y ejecutar los notebooks en orden.


## Secuencia experimental

| Notebook | Propósito | Salida esperada |
|---|---|---|
| `01_exploracion_datos.ipynb` | Validar estructura, objetivo, monto, tiempo y ausencia | Evidencia para decisiones de preparación |
| `02_particion_temporal.ipynb` | Congelar train/validación/test y gaps | Manifiesto de particiones |
| `03_modelado_y_decisiones_costos.ipynb` | Baseline calibrado en validation y comparación parametrizada de políticas | Métricas predictivas; comparación con la matriz original; escenarios finales de Ca pendientes |
| `04_sensibilidad_hipotetica_validation.ipynb` | Sensibilidad exploratoria de Ca sobre probabilidades guardadas | Tabla/gráfica hipotéticas; usa `src/fraud_cost/scenarios.py`; mantiene los controles de aprobación y reserva de test |

Los notebooks presentan el flujo experimental. La carga/unión, la partición, los costos, la ponderación del árbol y la evaluación compartida de escenarios viven en `src/fraud_cost/`; el script y el notebook 04 llaman a la misma función de escenarios.

## Reglas metodológicas

- La clasificación y la decisión económica son etapas distintas. El corte clasificatorio de 0.5 no se optimiza en este proyecto.
- El diseño temporal usa un bloque denominado test y el modelado no lo puntúa ni reporta métricas. **Limitación conocida:** el EDA inicial recorrió todo el conjunto etiquetado y la comparación histórica de gaps consultó conteos agregados de fraude del candidato a test; por tanto, este bloque no es un holdout estrictamente intocado. El código actual del notebook 02 ya oculta etiquetas de test en los resúmenes futuros. Consultar `docs/02_particion_temporal.md` antes de describirlo como evaluación independiente.
- No se remuestrea automáticamente: alterar la prevalencia puede deteriorar la interpretación probabilística.
- La calibración se evalúa y documenta antes de aplicar Bayes Minimum Risk.
- Todas las estrategias se comparan sobre las mismas observaciones y con la misma matriz de costos.
- La métrica principal es costo total/ahorro. AUC-PR, recall y tasa de legítimas bloqueadas explican el comportamiento.
- `sample_weight` es una aproximación sensible al costo, no un sustituto conceptualmente idéntico de un árbol con costos dependientes del ejemplo.

## Trazabilidad

Cada cifra del informe final debe poder rastrearse a versión/hash de datos, partición, configuración, commit, notebook y archivo exportado en `results/`. La matriz del anteproyecto y su implementación están en [`docs/03_protocolo_costos_y_evaluacion.md`](docs/03_protocolo_costos_y_evaluacion.md). Los escenarios en [`docs/04_sensibilidad_hipotetica_validation.md`](docs/04_sensibilidad_hipotetica_validation.md) son retrospectivos e hipotéticos, no costos aprobados. El test permanece sin puntuar.
