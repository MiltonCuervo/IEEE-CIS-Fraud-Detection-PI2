# Optimización basada en costos para el bloqueo de transacciones sospechosas

**Proyecto Integrador II — Ingeniería de Sistemas, Universidad de Antioquia, 2026-2.**

Estudiamos cómo cambia el costo de intervenir transacciones cuando pasamos del corte convencional 0,5 a una decisión económica por operación. Comparamos regla fija, Bayes Minimum Risk (BMR) y árbol ponderado por costos. El predictor estima fraude; la política determina la acción.

## Estado metodológico

Seguimos los notebooks 01–03 revisados por la profesora. El 03 compara XGBoost, LightGBM, CatBoost y regresión logística sobre una representación ordinal de 432 entradas. Los cuatro reciben su propio calibrador Platt. CatBoost utiliza códigos numéricos, sin tratamiento nativo de categorías.

El notebook 04 completó la selección temporal interna: ocho configuraciones, dos ventanas expansivas y 16 ajustes. Seleccionamos `lightgbm_referencia` con AP media raw de **0,5881**. La etapa 05 ya verificó y adoptó su ajuste y su propio Platt del 03, sin reentrenar. Reconstruyó probabilidades y las tres políticas en validación posterior: con Ca=10, BMR cuesta 84,169.73 unidades y mejora 57.51% frente al corte fijo de LightGBM. Son resultados exploratorios del bloque de desarrollo. Los costos administrativos siguen siendo hipotéticos y el test permanece sin puntuar.

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
tests/             Preparación, costos, selección temporal y recuperación de artefactos
```

## Preparación y ejecución

Desde la raíz, en PowerShell:

```powershell
scripts/setup.ps1 -ConstraintsFile .\requirements-reproduction.txt
```

Descargamos `train_transaction.csv` y `train_identity.csv` desde [IEEE-CIS en Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) y los ubicamos en `data/raw/`. No subimos datos, modelos ni credenciales a GitHub. No necesitamos las etiquetas desconocidas del test de la competencia: nuestro test temporal se separa dentro del conjunto etiquetado.

Utilizamos Python 3.11 o 3.12. Si el launcher `py` no encuentra uno, indicamos su ruta con `-PythonExecutable <ruta-a-python.exe>`. El script conserva un entorno anterior incompatible en `.venv.backup.*` y registra el kernel `ieee-cis-pi2`. Un error de acceso dentro de un entorno restringido no demuestra que Python esté dañado: primero comprobamos el mismo intérprete desde una terminal autorizada.

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
| 04_seleccion_temporal_modelos | Comparar ocho configuraciones solo dentro de train, sin calibrar | 16 ajustes, ranking por AP media, protocolo y configuración LightGBM seleccionada |
| 05_lightgbm_calibracion_y_costos | Adoptar LightGBM y su Platt verificados; diagnosticar probabilidades y comparar tres políticas | Métricas, grupos y montos; 18 costos/escenarios; modelo y scores LightGBM locales |

La ejecución completa del 03 requiere memoria para una matriz densa de 380.815 × 432 y los predictores. La matriz de train ocupa aproximadamente 627,6 MiB en float32; las copias y estructuras internas aumentan el consumo. Los cuatro modelos se ajustan de manera secuencial y utilizan hasta cuatro hilos cada uno; la regresión logística tiene sus propias reglas de paralelización.

`requirements.txt` declara rangos de versiones. `requirements-reproduction.txt` fija las siete versiones principales registradas en 03/04; no es un lockfile de todas las dependencias. Los registros de cada corrida identifican versiones, parámetros, advertencias y hashes. La reconstrucción local guarda además `pip freeze`. Si se modifica el procesamiento, se regenera la corrida completa; no se actualizan hashes para atribuir resultados anteriores a otra fuente.

### Artefactos locales y reconstrucción verificada

Hacer pull no descarga modelos ni scores ignorados por Git. Antes de cargar un artefacto comprobamos su procedencia:

```powershell
.\.venv\Scripts\python.exe scripts/recover_reference_artifacts.py --verify-only
```

Si faltan o pertenecen al procesamiento anterior, podemos reconstruir la corrida 03 manteniendo los registros históricos de 03/04:

```powershell
.\.venv\Scripts\python.exe scripts/recover_reference_artifacts.py
```

El script ejecuta el 03 en una copia aislada, contrasta cinco tablas con tolerancias declaradas antes de entrenar y verifica inferencia sobre las 42.047 operaciones posteriores. Solo después publica scores XGBoost, baseline, paquete ordinal de cuatro modelos y árboles, respaldando sus destinos anteriores. No puntúa test ni selecciona nuevamente. El recibo `results/tables/recuperacion_03_config.json` distingue los hashes reconstruidos de los originales; la evidencia detallada queda local en `data/interim/reference_recovery/`.

Para continuar desde código usamos `fraud_cost.artifacts.verify_reference_artifacts(PROJECT_ROOT)` antes de deserializar. Un consumidor que compare únicamente los hashes binarios originales debe detenerse si los archivos fueron reconstruidos. Véase [recuperación local y trazabilidad](docs/recuperacion_local_y_trazabilidad.md).

## Variables, probabilidades y costos

Conservamos 431 columnas originales como predictores y añadimos `has_identity`: 432 entradas, 37 categóricas y 395 numéricas. Excluimos `TransactionID`, `TransactionDT`, `isFraud` y `split` de X. El tiempo organiza las particiones; el identificador alinea operaciones y la etiqueta sirve para ajustar o evaluar exclusivamente en los bloques permitidos.

Las categorías se representan con ordinal, desconocidas con −1 y ausencia con un token explícito. Las numéricas se imputan con medianas de train. No creamos one-hot, indicadores adicionales de ausencia, atributos históricos ni variables logarítmicas. Para la regresión logística también aprendemos un escalador en train. Se mantiene documentada la discrepancia de doce códigos de identidad tratados numéricamente.

Intervenir cuesta Ca en ambas clases. Aprobar fraude cuesta el monto original y aprobar una operación legítima cuesta cero. BMR interviene cuando `p × A > Ca`; en empate aprueba. Reportamos costo total y ahorro frente a la mejor política trivial y, por separado, mejora frente a la regla fija. No presentamos Ca como costo bancario real ni asignamos una moneda no verificada.

## Documentación y límites

- [Informe EDA](docs/01_informe_eda.md).
- [Partición temporal](docs/02_particion_temporal.md).
- [Protocolo y reporte de la etapa 03](docs/03_protocolo_costos_y_evaluacion.md).
- [Selección temporal interna y resultados del 04](docs/04_seleccion_temporal_modelos.md).
- [Adopción LightGBM, calibración y costos](docs/05_lightgbm_calibracion_y_costos.md).
- [Diccionario y recorrido de variables](docs/diccionario_variables_y_flujo_modelo.md).

Los resultados del 03 son exploratorios sobre validación posterior. La selección interna del 04 utilizó solo train para ajustar y calcular su ranking, pero sus decisiones de desarrollo estuvieron precedidas por aquella exploración. El EDA y la comparación histórica de gaps consultaron etiquetas agregadas del periodo final; declaramos esta limitación y no presentamos el test como estrictamente ciego desde el inicio.

Antes de evaluarlo fijaremos predictor, calibrador, costos hipotéticos justificados y reglas. El 03 conserva su comparación económica con XGBoost. El 05 usa probabilidades LightGBM y los mismos árboles verificados, montos y escenarios; sus diferencias de costo son descriptivas sobre validación de desarrollo. La siguiente etapa estudiará estabilidad y errores antes de congelar el protocolo final.
