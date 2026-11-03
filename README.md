# Optimización basada en costos para el bloqueo de transacciones sospechosas

**Proyecto Integrador II — Ingeniería de Sistemas, Universidad de Antioquia, 2026-2.**

Estudiamos cómo cambia el costo de intervenir transacciones cuando pasamos del corte convencional 0,5 a una decisión económica por operación. Comparamos regla fija, Bayes Minimum Risk (BMR) y árbol ponderado por costos. El predictor estima fraude; la política determina la acción.

## Estado metodológico

Seguimos los notebooks 01–03 revisados por la profesora. El 03 compara XGBoost, LightGBM, CatBoost y regresión logística sobre una representación ordinal de 432 entradas. Los cuatro reciben su propio calibrador Platt. CatBoost utiliza códigos numéricos, sin tratamiento nativo de categorías.

No hemos seleccionado el predictor final ni puntuado el test. Los costos administrativos son escenarios hipotéticos en unidades del dataset. La selección temporal interna es el siguiente experimento, todavía por construir.

El 2 de noviembre de 2026 retiramos de la secuencia activa los notebooks 04–09, sus informes, módulos y resultados del procesamiento anterior. Git conserva ese trabajo en el commit `089df7f` y sus antecedentes. Sus 447 entradas, atributos históricos y matriz one-hot de 2.642 columnas no describen la metodología actual. Los artefactos locales antiguos no autorizan reutilizar sus conclusiones.

## Organización

```text
data/raw/          CSV originales locales, sin modificaciones
data/interim/      Manifiesto y probabilidades reproducibles, locales
data/processed/    Reservado para datos preparados
notebooks/         Preguntas, parámetros, ejecución y lectura de resultados
src/fraud_cost/    Carga, partición, preparación, inventario y costos reutilizables
docs/              Informes y decisiones metodológicas
results/tables/    Tablas seleccionadas y registro de la corrida
results/figures/   Figuras utilizadas en los informes
results/models/    Predictores y calibradores locales, ignorados por Git
tests/             Contratos de preparación y de la matriz económica
```

## Preparación y ejecución

Desde la raíz, en PowerShell:

```powershell
scripts/setup.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Descargamos `train_transaction.csv` y `train_identity.csv` desde [IEEE-CIS en Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) y los ubicamos en `data/raw/`. No subimos datos, modelos ni credenciales a GitHub. No necesitamos las etiquetas desconocidas del test de la competencia: nuestro test temporal se separa dentro del conjunto etiquetado.

```powershell
.\.venv\Scripts\python.exe scripts/data_provenance.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m jupyter lab
```

Seleccionamos el kernel `ieee-cis-pi2` y ejecutamos en orden:

| Notebook | Propósito | Evidencia |
|---|---|---|
| 01_exploracion_datos | Estructura, objetivo, ausencia, montos y tiempo | Tablas y figuras EDA; catálogo de 434 originales |
| 02_particion_temporal | Separar pasado, gaps, validación y test | Manifiesto temporal y resúmenes |
| 03_modelado_y_decisiones_costos | Comparar cuatro predictores con preparación ordinal y Platt; explorar tres políticas con XGBoost | Métricas, inventario, costos hipotéticos, figuras y registro de corrida |

La ejecución completa del 03 requiere memoria para una matriz densa de 380.815 × 432 y los predictores. La matriz de train ocupa aproximadamente 627,6 MiB en float32; las copias y estructuras internas aumentan el consumo. Los cuatro modelos se ajustan de manera secuencial y utilizan hasta cuatro hilos cada uno; la regresión logística tiene sus propias reglas de paralelización.

`requirements.txt` declara rangos de versiones, no un entorno bloqueado. `comparacion_modelos_config.json` registra las versiones efectivas, los parámetros, advertencias y hashes. Si se modifica el procesamiento, se regenera la corrida completa; no se actualizan hashes para atribuir resultados anteriores a otra fuente.

## Variables, probabilidades y costos

Conservamos 431 columnas originales como predictores y añadimos `has_identity`: 432 entradas, 37 categóricas y 395 numéricas. Excluimos `TransactionID`, `TransactionDT`, `isFraud` y `split` de X. El tiempo organiza las particiones; el identificador alinea operaciones y la etiqueta sirve para ajustar o evaluar exclusivamente en los bloques permitidos.

Las categorías se representan con ordinal, desconocidas con −1 y ausencia con un token explícito. Las numéricas se imputan con medianas de train. No creamos one-hot, indicadores adicionales de ausencia, atributos históricos ni variables logarítmicas. Para la regresión logística también aprendemos un escalador en train. Se mantiene documentada la discrepancia de doce códigos de identidad tratados numéricamente.

Intervenir cuesta Ca en ambas clases. Aprobar fraude cuesta el monto original y aprobar una operación legítima cuesta cero. BMR interviene cuando `p × A > Ca`; en empate aprueba. Reportamos costo total y ahorro frente a la mejor política trivial y, por separado, mejora frente a la regla fija. No presentamos Ca como costo bancario real ni asignamos una moneda no verificada.

## Documentación y límites

- [Informe EDA](docs/01_informe_eda.md).
- [Partición temporal](docs/02_particion_temporal.md).
- [Protocolo y reporte de la etapa 03](docs/03_protocolo_costos_y_evaluacion.md).
- [Diccionario y recorrido de variables](docs/diccionario_variables_y_flujo_modelo.md).

Los resultados del 03 son exploratorios sobre validación posterior. La próxima selección interna no será independiente de las decisiones de desarrollo ya motivadas por esa exploración. El EDA y la comparación histórica de gaps consultaron etiquetas agregadas del periodo final; declaramos esta limitación y no presentamos el test como estrictamente ciego desde el inicio.

Antes de evaluarlo fijaremos predictor, calibrador, costos hipotéticos justificados y reglas. La comparación económica de esta etapa usa XGBoost; que otro modelo lidere en AP no implica que ya hayamos demostrado mayor ahorro con ese predictor.
