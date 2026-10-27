# Optimización basada en costos para el bloqueo de transacciones sospechosas

**Proyecto Integrador II — Ingeniería de Sistemas, Universidad de Antioquia, 2026-2**

Este proyecto compara tres políticas de decisión sobre la misma matriz de costos: umbral convencional, Bayes Minimum Risk (BMR) y árbol sensible al costo. Las dos primeras operan sobre probabilidades calibradas de XGBoost; el árbol aprende acciones con atributos y costos de entrenamiento.

**Estado actual:** el test temporal no ha sido puntuado; los costos de negocio y el calibrador final no están aprobados. Por tanto, no existe aún una conclusión económica definitiva. La calibración isotónica permanece como candidata y no sustituye los resultados anteriores.

---

## Pregunta de investigación

¿En qué medida la optimización de la regla de decisión basada en costos permite mejorar el ahorro financiero frente a un umbral fijo convencional para el bloqueo de transacciones sospechosas en el conjunto de datos IEEE-CIS, y cómo varía este beneficio ante diferentes costos administrativos de revisión Ca?

---

## Datos

Los datos provienen de la competencia [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) en Kaggle. Se requieren dos archivos, que deben descargarse y ubicarse sin modificar en `data/raw/`:

- `train_transaction.csv`
- `train_identity.csv`

No escribir credenciales en notebooks. Configurar la API de Kaggle fuera del repositorio siguiendo su [documentación oficial](https://www.kaggle.com/docs/api), o descargar los archivos manualmente desde la página de la competencia.

El repositorio del proyecto esta en GitHub:
[https://github.com/MiltonCuervo/IEEE-CIS-Fraud-Detection-PI2](https://github.com/MiltonCuervo/IEEE-CIS-Fraud-Detection-PI2)

Los CSV originales y los artefactos intermedios/modelos se mantienen localmente e ignorados por Git. Las tablas y figuras seleccionadas de `results/` si se versionan como evidencia de corridas documentadas; al regenerarlas, revisar el diff antes de confirmar.

---

## Estructura del repositorio

```text
.
├── data/
│   ├── raw/          # Archivos originales, inmutables y no versionados
│   ├── interim/      # Datos intermedios reproducibles
│   └── processed/    # Particiones listas para modelar
├── notebooks/        # Relato experimental numerado
├── src/fraud_cost/   # Logica reutilizable y comprobable
├── docs/             # Decisiones, protocolo y preparacion de sustentaciones
├── results/
│   ├── figures/      # Figuras finales exportadas
│   ├── tables/       # Metricas y tablas finales
│   └── models/       # Modelos y calibradores serializados
└── requirements.txt
```

---

## Preparación del entorno

**Paso 1 — Crear el entorno virtual**

Desde la raiz del repositorio, ejecutar:

```powershell
scripts/setup.ps1
```

El script crea `.venv` con Python 3.12 (o 3.11 si no esta disponible), instala las dependencias, el paquete local en modo editable y el kernel `ieee-cis-pi2`. Verifica si un entorno existente funciona antes de reutilizarlo; si no funciona, lo guarda en una carpeta de respaldo y crea uno nuevo sin eliminar el anterior. Para especificar un interprete explicito:

```powershell
scripts/setup.ps1 -PythonExecutable <ruta-al-interprete>
```

**Paso 2 — Reinstalación manual (si se requiere)**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

**Paso 3 — Descargar los datos**

Descargar los archivos desde la competencia [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) y ubicarlos en `data/raw/` sin modificarlos.

**Paso 4 — Registrar procedencia de datos**

```powershell
.\.venv\Scripts\python.exe scripts\data_provenance.py
```

Escribe el manifiesto local `data/interim/data_provenance.json` con hashes SHA-256 de los CSV, versiones de dependencias y commit actual. No se versiona.

**Paso 5 — Abrir Jupyter**

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

Seleccionar el kernel `ieee-cis-pi2` y ejecutar los notebooks en orden (ver secuencia mas abajo).

**Paso 6 — Verificar la lógica reutilizable y la procedencia**

Los notebooks incluyen comprobaciones ejecutables y las etapas 05–09 verifican hashes de sus entradas antes de reutilizarlas. Las corridas históricas documentaron una suite de pruebas, pero `tests/` fue retirada en la organización actual; no presentamos sus resultados como pruebas disponibles hoy. Antes de cambios algorítmicos debe recuperarse una suite versionada. La auditoría del inventario, descrita más abajo, verifica el predictor existente sin reentrenarlo.

Las versiones utilizadas quedan registradas en:

- `results/tables/seleccion_temporal_config.json`
- `results/tables/reajuste_config.json`
- `results/tables/comparacion_economica_config.json`
- `results/tables/diagnostico_estabilidad_config.json`
- `results/tables/comparacion_calibradores_config.json`

`requirements.txt` declara las dependencias; no constituye un lockfile exacto.

---

## Secuencia experimental

| Notebook | Proposito | Salida esperada |
|---|---|---|
| `01_exploracion_datos.ipynb` | Validar estructura, objetivo, monto, tiempo y ausencia | Evidencia para decisiones de preparacion |
| `02_particion_temporal.ipynb` | Congelar train/validacion/test y gaps | Manifiesto de particiones |
| `03_modelado_y_decisiones_costos.ipynb` | Baseline calibrado en validation y comparacion parametrizada de politicas | Metricas predictivas; comparacion con la matriz original; escenarios finales de Ca pendientes |
| `04_sensibilidad_hipotetica_validation.ipynb` | Sensibilidad exploratoria de Ca sobre probabilidades guardadas | Tabla/grafica hipoteticas; usa `src/fraud_cost/scenarios.py` |
| `05_preparacion_y_seleccion_temporal.ipynb` | Comparar atributos y cuatro configuraciones mediante dos folds internos de train | Metricas por ventana, diagnostico de variables, figura y configuracion seleccionada |
| `06_reajuste_y_calibracion.ipynb` | Reajustar la seleccion con todo train, calibrar en validacion temprana y diagnosticar en posterior | Metricas raw/calibradas, curva y grupos de calibracion, predictor completo y scores nuevos |
| `07_comparacion_economica_validation.ipynb` | Comparar politica fija, BMR y arbol ponderado por Ca ilustrativo | Costos, referencias, compromisos de accion y sensibilidad a calibracion |
| `08_diagnostico_y_estabilidad.ipynb` | Examinar calibracion por monto/riesgo, frontera BMR y cuatro subperiodos posteriores | Cuatro tablas, dos figuras y configuracion |
| `09_comparacion_calibradores.ipynb` | Contrastar raw, Platt e isotonica manteniendo el XGBoost del 06 y fundamentar Ca con literatura | Cuatro tablas, dos figuras, configuracion y candidato isotónico local |

## Matriz de costos

Intervenir (bloquear y revisar) cuesta `Ca` tanto en fraude como en operaciones legitimas. Aprobar un fraude cuesta su monto; aprobar una operacion legitima cuesta cero. BMR interviene cuando `p_i x TransactionAmt_i > Ca`. Se calcula ahorro frente al menor costo entre aprobar todo e intervenir todo y, por separado, mejora frente a la politica fija.

La matriz completa esta en [`docs/03_protocolo_costos_y_evaluacion.md`](docs/03_protocolo_costos_y_evaluacion.md).

---

## Reglas metodológicas

- La clasificacion y la decision economica son etapas distintas. El corte clasificatorio de 0.5 no se optimiza en este proyecto.
- **Limitacion conocida sobre el test:** el EDA inicial recorrio todo el conjunto etiquetado y la comparacion historica de gaps consulto conteos agregados de fraude del candidato a test. Por tanto, este bloque no es un holdout estrictamente intocado. Ver [`docs/02_particion_temporal.md`](docs/02_particion_temporal.md) antes de describirlo como evaluacion independiente.
- No se remuestrea automaticamente: alterar la prevalencia puede deteriorar la interpretacion probabilistica.
- La seleccion del **05** usa AP media de ventanas temporales dentro de train. Sus atributos historicos excluyen operaciones simultaneas y congelan el estado de entrenamiento al evaluar. AP interna no demuestra ahorro, calibracion ni superioridad estadistica.
- La calibracion se evalua y documenta antes de aplicar Bayes Minimum Risk.
- El calibrador pertenece al predictor que genero sus puntuaciones. Despues de calibrar no se reajusta XGBoost ni se actualiza el historial con validation.
- Todas las estrategias se comparan sobre las mismas observaciones y con la misma matriz de costos.
- La comparacion del **07** es hipotetica: Ca no se elige por el ahorro obtenido.
- El **08** es descriptivo: bandas fijas, mismos grupos y subperiodos de igual duracion sin nuevos ajustes. Los mínimos de referencias triviales por periodo no se suman como referencia global.
- El **09** ajusta isotónica unicamente en validacion temprana. La referencia fija conserva Platt. Los Ca = 1; 2,5; 5; 10 se inspiran en literatura de otros datos/empresas; 20; 50; 100 amplian la sensibilidad. No se trasladan euros al dataset ni se presentan costos reales sin moneda verificada.
- `sample_weight` es una aproximacion sensible al costo, no un sustituto conceptualmente identico de un arbol con costos dependientes del ejemplo.
- La metrica principal es costo total/ahorro. AUC-PR, recall y tasa de legitimas bloqueadas explican el comportamiento.

---

## Trazabilidad y documentación

Cada cifra del informe final debe poder rastrearse a: version/hash de datos, particion, configuracion, commit, notebook y archivo exportado en `results/`.

| Documento | Contenido |
|---|---|
| [`docs/01_informe_eda.md`](docs/01_informe_eda.md) | Estructura, ausencia, variables y hallazgos del EDA |
| [`docs/02_particion_temporal.md`](docs/02_particion_temporal.md) | Particiones, límites y usos permitidos de cada bloque |
| [`docs/03_protocolo_costos_y_evaluacion.md`](docs/03_protocolo_costos_y_evaluacion.md) | Matriz de costos |
| [`docs/04_sensibilidad_hipotetica_validation.md`](docs/04_sensibilidad_hipotetica_validation.md) | Referencia economica |
| [`docs/05_preparacion_y_seleccion_temporal.md`](docs/05_preparacion_y_seleccion_temporal.md) | Seleccion temporal |
| [`docs/06_reajuste_y_calibracion.md`](docs/06_reajuste_y_calibracion.md) | Reajuste y calibracion |
| [`docs/07_comparacion_economica_validation.md`](docs/07_comparacion_economica_validation.md) | Comparacion de tres politicas y diagnostico de calibracion |
| [`docs/08_diagnostico_y_estabilidad.md`](docs/08_diagnostico_y_estabilidad.md) | Calibracion condicional, frontera economica y estabilidad temporal |
| [`docs/09_comparacion_calibradores_y_costos.md`](docs/09_comparacion_calibradores_y_costos.md) | Comparacion Platt/isotonica y alcance de antecedentes de Ca |
| [`docs/diccionario_variables_y_flujo_modelo.md`](docs/diccionario_variables_y_flujo_modelo.md) | Inventario por variable, ingeniería, transformaciones y contrato de entrada/salida |

El test permanece sin puntuar. Antes de evaluarlo es necesario acordar escenarios finales de Ca y unidades, discutir la calibracion y congelar el protocolo.

## Detalle de variables y reproducción del inventario

El notebook 01 conserva las explicaciones y visualizaciones de la revisión de la profesora e incorpora nuestro inventario completo. Exporta `results/tables/eda_diccionario_variables.csv`, una fila por cada una de las 434 originales. El [diccionario y recorrido](docs/diccionario_variables_y_flujo_modelo.md) distingue tipo semántico, almacenamiento, tratamiento efectivo, columnas auxiliares, ingeniería y resultados del modelo.

El predictor actual tiene 447 entradas: 37 categóricas y 410 numéricas. Su matriz de 2.642 columnas combina 410 valores numéricos, 342 indicadores de ausencia y 1.890 one-hot. Los doce campos id que el proveedor declara categóricos y que el pipeline trata numéricamente permanecen señalados como discrepancia pendiente; esta integración no cambia sus tipos ni sus resultados.

Con el entorno del proyecto y el artefacto del 06 ya disponible, desde la raíz:

```python
from pathlib import Path
from fraud_cost.schema import export_model_inventory

catalogo, matriz, registro = export_model_inventory(Path.cwd())
```

Verifica las fuentes registradas en 05/06 y exporta `modelo_diccionario_variables.csv` (451 filas, 447 entradas directas), `modelo_matriz_variables.csv` (2.642 posiciones) e `inventario_variables_config.json`. No entrena ni produce nuevas predicciones. El perfil EDA global es descriptivo; las medianas, indicadores y vocabularios del predictor se aprendieron solo con train.

Los resultados 03/04 siguen siendo la línea base histórica, no los scores seleccionados del 06. El 04 consume `validation_scores.parquet`; el 07 usa `validation_scores_selected.parquet` y `selected_xgboost_platt.joblib`. El 08 necesita los scores/acciones locales del 06/07; el 09 carga el predictor del 06 y ajusta solo la candidata isotónica temprana. Modelos y archivos por operación permanecen locales e ignorados por Git. Un clon sin ellos debe regenerar las etapas previas.

El notebook 02 comprueba la asignación antes de reutilizar un manifiesto existente, sin sobrescribirlo innecesariamente: una regeneración idéntica de filas puede cambiar bytes Parquet y romper la procedencia. Si cambia una fuente o asignación, se revisa y regenera la cadena necesaria; no se reescriben hashes para aparentar una corrida anterior.
