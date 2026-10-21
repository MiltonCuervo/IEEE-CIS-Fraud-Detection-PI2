# Optimización basada en costos para el bloqueo de transacciones sospechosas

**Proyecto Integrador II — Ingeniería de Sistemas, Universidad de Antioquia, 2026-2**

Este proyecto compara tres políticas de decisión sobre la misma matriz de costos: umbral convencional, Bayes Minimum Risk (BMR) y árbol sensible al costo. Las dos primeras operan sobre probabilidades calibradas de XGBoost; el árbol aprende acciones con atributos y costos de entrenamiento.

**Estado actual:** el test temporal no ha sido puntuado; los costos de negocio y el calibrador final no están aprobados. Por tanto, no existe aún una conclusión económica definitiva. La calibración isotónica permanece como candidata y no sustituye los resultados anteriores.

---

## Pregunta de investigación

En que medida la optimización de la regla de decisión basada en costos permite mejorar el ahorro financiero frente a un umbral fijo convencional para el bloqueo de transacciones sospechosas en el conjunto de datos IEEE-CIS, y como varia este beneficio ante diferentes costos administrativos de revision Ca.

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

## Preparacion del entorno

**Paso 1 — Crear el entorno virtual**

Desde la raiz del repositorio, ejecutar:

```powershell
scripts/setup.ps1
```

El script crea `.venv` con Python 3.12 (o 3.11 si no esta disponible), instala las dependencias, el paquete local en modo editable y el kernel `ieee-cis-pi2`. Verifica si un entorno existente funciona antes de reutilizarlo; si no funciona, lo guarda en una carpeta de respaldo y crea uno nuevo sin eliminar el anterior. Para especificar un interprete explicito:

```powershell
scripts/setup.ps1 -PythonExecutable <ruta-al-interprete>
```

**Paso 2 — Reinstalacion manual (si se requiere)**

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

**Paso 6 — Verificar la logica reutilizable**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

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

### Dependencias entre notebooks

- El **05** necesita el manifiesto generado por el **02**, pero no reutiliza probabilidades ni calibrador del **03**.
- El **06** necesita la seleccion del **05** y verifica sus hashes antes de entrenar.
- El **07** verifica fuentes y artefactos del **06/07**; reutiliza preprocesamiento y ajusta solo los arboles por Ca.
- El **08** verifica fuentes y artefactos del **06/07**; reutiliza scores/acciones. No entrena ni carga modelos, no puntua otros bloques, y requiere los dos Parquet locales de esas etapas (las tablas agregadas no bastan).
- El **09** verifica la seleccion/reajuste, carga el predictor y los scores del **06**, y ajusta unicamente un candidato isotónico con validacion temprana. No reajusta XGBoost/Platt ni sobrescribe sus artefactos. Requiere datos originales y artefactos locales del **02/05/06**.

---

## Diseno de la logica reutilizable (`src/fraud_cost/`)

La carga/union, particion, costos, ponderacion del arbol y evaluacion compartida de escenarios viven en `src/fraud_cost/`. El script del notebook **04** llama a la misma funcion de escenarios.

| Modulo | Responsabilidad |
|---|---|
| `features.py` | Construye atributos disponibles antes de cada operacion; preprocesamiento por fold |
| `selection.py` | Registra la seleccion dentro de train |
| `refit.py` | Reajusta y calibra |
| `economics.py` | Verifica artefactos y compara acciones mediante las funciones de costos, sin duplicar la matriz |
| `diagnostics.py` | Describe probabilidades y acciones guardadas por grupos y subperiodos |
| `calibration.py` | Contrasta calibradores sobre el mismo predictor/poblacion; conserva el candidato vinculado al modelo original |

No se mantiene una segunda implementacion de esta logica dentro de las celdas de los notebooks.

---

## Matriz de costos

Intervenir (bloquear y revisar) cuesta `Ca` tanto en fraude como en operaciones legitimas. Aprobar un fraude cuesta su monto; aprobar una operacion legitima cuesta cero. BMR interviene cuando `p_i x TransactionAmt_i > Ca`. Se calcula ahorro frente al menor costo entre aprobar todo e intervenir todo y, por separado, mejora frente a la politica fija.

La matriz completa esta en [`docs/03_protocolo_costos_y_evaluacion.md`](docs/03_protocolo_costos_y_evaluacion.md).

---

## Reglas metodologicas

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

## Trazabilidad y documentacion

Cada cifra del informe final debe poder rastrearse a: version/hash de datos, particion, configuracion, commit, notebook y archivo exportado en `results/`.

| Documento | Contenido |
|---|---|
| [`docs/03_protocolo_costos_y_evaluacion.md`](docs/03_protocolo_costos_y_evaluacion.md) | Matriz de costos |
| [`docs/04_sensibilidad_hipotetica_validation.md`](docs/04_sensibilidad_hipotetica_validation.md) | Referencia economica |
| [`docs/05_preparacion_y_seleccion_temporal.md`](docs/05_preparacion_y_seleccion_temporal.md) | Seleccion temporal |
| [`docs/06_reajuste_y_calibracion.md`](docs/06_reajuste_y_calibracion.md) | Reajuste y calibracion |
| [`docs/07_comparacion_economica_validation.md`](docs/07_comparacion_economica_validation.md) | Comparacion de tres politicas y diagnostico de calibracion |
| [`docs/08_diagnostico_y_estabilidad.md`](docs/08_diagnostico_y_estabilidad.md) | Calibracion condicional, frontera economica y estabilidad temporal |
| [`docs/09_comparacion_calibradores_y_costos.md`](docs/09_comparacion_calibradores_y_costos.md) | Comparacion Platt/isotonica y alcance de antecedentes de Ca |
| [`docs/guia_defensa_tecnica.md`](docs/guia_defensa_tecnica.md) | Explicacion del dataset, tecnicas, costos, metricas, hallazgos y respuestas para sustentacion |
| [`docs/estado_actual_proyecto.md`](docs/estado_actual_proyecto.md) | Artefactos, decisiones pendientes y orden para retomar (corte al 12 de octubre de 2026) |

El test permanece sin puntuar. Antes de evaluarlo es necesario acordar escenarios finales de Ca y unidades, discutir la calibracion y congelar el protocolo.
