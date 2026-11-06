# Optimización basada en costos para el bloqueo de transacciones sospechosas

**Proyecto Integrador II — Ingeniería de Sistemas, Universidad de Antioquia, 2026-2.**

Estudiamos cómo convertir una probabilidad de fraude en una decisión económica por transacción. Comparamos el corte fijo 0,5, Bayes Minimum Risk (BMR) y un árbol ponderado por costos. El clasificador estima fraude; la política decide aprobar o intervenir. Conservamos el corte clasificatorio 0,5 y diferenciamos la frontera económica por monto.

## Resultado consolidado

La secuencia **01–08 está ejecutada**. Seleccionamos LightGBM referencia mediante AP media de dos ventanas temporales internas de train: **0,5881** entre ocho configuraciones. Conservamos su Platt propio y evaluamos el bloque final de **84.233 operaciones y 2.960 fraudes** sin reajustar estimadores.

Con **Ca=10 hipotético**, obtuvimos:

| Política | Costo total | Recall de fraude | Legítimas intervenidas | Monto fraudulento capturado |
|---|---:|---:|---:|---:|
| Regla fija 0,5 | 363.466,629 | 37,26% | 0,40% | 22,71% |
| BMR | **181.140,258** | **55,84%** | 7,91% | **77,79%** |
| Árbol ponderado | 228.707,974 | 42,43% | 7,33% | 65,34% |

BMR mejora el costo **50,16% frente a fija** y ahorra **59,91% frente a la mejor referencia trivial** en este escenario; son denominadores diferentes. Tiene menor costo en los seis Ca evaluados. No elegimos Ca retrospectivamente ni promediamos ahorros entre escenarios.

La AP final es **0,5178**. Platt mejora Brier/log-loss en validación, pero los empeora ligeramente en test: conservamos el calibrador fijado y documentamos el resultado. Las métricas no prueban una superioridad universal, calibración perfecta o ahorro bancario real. El detalle está en el [informe final de evaluación](docs/08_evaluacion_temporal_final.md).

## Flujo metodológico

La tabla describe el orden del experimento. En una copia con evaluación terminada, **no se deben reejecutar etapas de entrenamiento o generación de registros sobre las mismas rutas congeladas**.

| Etapa | Pregunta y operación | Salida |
|---|---|---|
| [01 EDA](notebooks/01_exploracion_datos.ipynb) | Qué datos, variables, ausencias, montos y variación temporal tenemos | Inventario de 434 originales; tablas y figuras descriptivas |
| [02 Partición](notebooks/02_particion_temporal.ipynb) | Cómo aprender del pasado y evaluar el futuro | Train → gap 7 días → validación → gap 7 días → test |
| [03 Modelos](notebooks/03_modelado_y_decisiones_costos.ipynb) | Comparar cuatro clasificadores con sus propios Platt | Matriz ordinal de 432 entradas; comparación predictiva y referencia económica XGBoost |
| [04 Selección](notebooks/04_seleccion_temporal_modelos.ipynb) | Elegir configuración con ventanas internas de train | 16 ajustes; seis métricas raw y ranking por AP media |
| [05 Adopción](notebooks/05_lightgbm_calibracion_y_costos.ipynb) | Verificar LightGBM seleccionado y evaluar sus acciones | Platt propio, probabilidades y tres políticas en validación posterior |
| [06 Estabilidad](notebooks/06_estabilidad_y_errores_economicos.ipynb) | Explicar variación, pérdida residual y frontera económica | Tres subperiodos, concentración de pérdidas y sensibilidad de probabilidades |
| [07 Protocolo](notebooks/07_protocolo_final.ipynb) | Fijar decisiones antes de test | Modelo, preparación, calibrador, políticas, costos y métricas congelados |
| [08 Evaluación](notebooks/08_evaluacion_temporal_final.ipynb) | Medir las decisiones fijadas en el periodo final | Siete tablas, dos figuras y registro exclusivo de ejecución completa |

Train tiene 380.815 operaciones. La mitad temprana de validación, 42.046, ajusta Platt; las 42.047 posteriores diagnostican probabilidades y políticas. Los gaps no participan en ajuste ni evaluación. La selección del 04 usa únicamente pasados y futuros internos de train, aunque estuvo precedida por exploración externa ya consultada.

## Datos y preparación de variables

Usamos `train_transaction.csv` y `train_identity.csv` de [IEEE-CIS en Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data). La unión izquierda uno a uno conserva 590.540 transacciones y produce 434 columnas originales. El test de la competencia carece de etiquetas públicas; nuestro test se separa temporalmente de los datos etiquetados.

Excluimos ID, tiempo crudo y etiqueta de X: quedan 431 atributos originales. Añadimos `has_identity` y obtenemos **432 entradas: 37 categóricas y 395 numéricas**. `split` solo controla las poblaciones. Las medianas y vocabularios se aprenden con el pasado autorizado; categorías desconocidas reciben −1. Logística añade escalado. CatBoost recibe ordinal, sin categorías nativas.

Las doce variables id tratadas numéricamente pese a su clasificación categórica del proveedor están documentadas. No añadimos los atributos históricos, one-hot ni el calibrador isotónico del pipeline retirado. El [diccionario y recorrido de variables](docs/diccionario_variables_y_flujo_modelo.md) identifica cada tratamiento; las tablas `eda_diccionario_variables`, `modelo_variables_ordinal` y `modelo_vocabulario_ordinal` registran el inventario real.

## Matriz económica y alcance

Intervenir cuesta Ca tanto en fraude como en legítimas. Aprobar fraude cuesta su monto original A y aprobar una legítima cuesta cero. BMR interviene si `p_calibrada × A > Ca`; en empate aprueba. Los árboles aprenden acciones ponderadas por arrepentimiento de costo y su ajuste por impureza es aproximado.

Conservamos Ca = **1, 5, 10, 20, 50 y 100**, hipotéticos y en unidades de TransactionAmt, sin moneda verificada. La literatura del [protocolo 07](docs/07_protocolo_final.md) fundamenta escenarios, pero no estima gastos IEEE-CIS. No modelamos recuperación, capacidad, revisión imperfecta ni perjuicio comercial adicional.

El EDA y el diseño histórico de gaps consultaron etiquetas agregadas del periodo final; test no fue estrictamente ciego desde el inicio. Declaramos también la consulta de validación durante desarrollo, el orden artificial de ordinal y la variación temporal. Esas limitaciones forman parte del resultado consolidado.

## Organización

```text
data/raw/          CSV originales locales e inmutables
data/interim/      Manifiesto, scores, acciones y copia de ejecución congelada
data/processed/    Reserva de estructura; no contiene otra matriz activa
notebooks/         Ocho etapas con preguntas, parámetros y lectura de resultados
src/fraud_cost/    Operaciones reutilizables, sin duplicar lógica en cada notebook
docs/              Informes de etapa, diccionario y recuperación
results/tables/    Evidencia agregada y registros de procedencia
results/figures/   Figuras usadas para interpretar resultados
results/models/    Modelos locales verificados, ignorados por Git
scripts/           Instalación, adquisición, recuperación y auditoría
tests/             Contratos de preparación, tiempo, costos, protocolo y ejecución
```

`reference.py` prepara la matriz; `split.py` controla el tiempo; `costs.py` define una sola matriz económica; selección, adopción y estabilidad tienen módulos propios. `artifacts.py`, `final_protocol.py`, `execution_workspace.py` y `final_evaluation.py` mantienen la procedencia y evitan repetir test accidentalmente. Los módulos que aparecen en una congelación se conservan incluso si no son puntos de entrada.

## Consultar y verificar el proyecto cerrado

En otro computador se pueden abrir los notebooks ejecutados y las tablas versionadas. Con el entorno instalado, la auditoría publicada verifica estructura, enlaces, inventario, código, resultados y aritmética sin necesitar CSV originales o modelos:

```powershell
.\.venv\Scripts\python.exe scripts/audit_project.py
```

Con las dependencias locales conservadas, la comprobación completa incluye CSV originales, manifiesto, identidad de operaciones, scores, acciones y protocolo activo:

```powershell
.\.venv\Scripts\python.exe scripts/audit_project.py --local --output results/tables/auditoria_proyecto.json
.\.venv\Scripts\python.exe -m pytest -q
```

Estos comandos no entrenan ni generan nuevas predicciones. El 08 recupera una corrida completa verificando sus salidas; una incompleta se detiene para auditarla. Los flags `test_evaluated=False` de etapas previas describen sus momentos históricos; `evaluacion_test_config.json` acredita la evaluación completada.

## Instalar y reproducir

Usamos Python 3.11/3.12. Desde la raíz, en PowerShell:

```powershell
.\scripts\setup.ps1 -ConstraintsFile .\requirements-reproduction.txt
```

Si el launcher no encuentra Python, indicamos `-PythonExecutable <ruta>`. Obtenemos los dos CSV desde Kaggle tras aceptar sus reglas y los colocamos en `data/raw/`; `scripts/download_data.ps1` admite la CLI autenticada. Los datos, credenciales, modelos y archivos por operación no se suben a Git. `requirements-reproduction.txt` fija siete paquetes principales, no todas las dependencias transitivas.

Una matriz train de 380.815 × 432 en float32 ocupa unos 627,6 MiB, además de copias y modelos. El EDA carga las tablas completas. Los registros guardan parámetros, versiones y hashes; no atribuimos resultados anteriores a fuentes nuevas actualizando solo las huellas.

El protocolo 07 recibido referencia artefactos reconstruidos en otra máquina. Para 08 se creó una **versión local explícita y aislada antes de test**, se contrastaron doce tablas y se conservaron las decisiones. La copia de `data/interim/final_evaluation_v2/` es evidencia de esa versión, no un experimento sobrante. Hacer pull no la descarga.

Una reproducción nueva requiere su propia área y versión de experimento; no debe sobrescribir las dependencias congeladas ni presentarse como un test independiente nuevo. Las herramientas históricas de [recuperación](docs/recuperacion_local_y_trazabilidad.md) explican cómo contrastar artefactos. Tras observar test, cambiar modelos, calibración o atributos exige declarar otro experimento.

## Informes y trabajo restante

El [informe consolidado de resultados](docs/informe_resultados_consolidados.md) reúne los hallazgos de las etapas 01–08, la comparación económica final, las diferencias de calibración y los límites de las conclusiones. Es el punto de entrada para revisar los resultados completos.

[01 EDA](docs/01_informe_eda.md) · [02 Partición](docs/02_particion_temporal.md) · [03 Modelos y costos](docs/03_protocolo_costos_y_evaluacion.md) · [04 Selección](docs/04_seleccion_temporal_modelos.md) · [05 LightGBM](docs/05_lightgbm_calibracion_y_costos.md) · [06 Estabilidad](docs/06_estabilidad_y_errores_economicos.md) · [07 Protocolo](docs/07_protocolo_final.md) · [08 Evaluación final](docs/08_evaluacion_temporal_final.md).

Los informes describen cada etapa en su momento; las referencias a pasos futuros no implican que sigan pendientes. La limpieza retira salidas locales del pipeline descartado, conservado como código histórico en Git, y mantiene toda evidencia utilizada por la versión vigente.

La construcción técnica está concluida. El [borrador del artículo IEEE](docs/informe_final/README.md) está preparado en PDF de seis páginas con su fuente editable. Restan la revisión del equipo y de la profesora, la incorporación de observaciones, el aval docente y la sustentación. El resultado defendible es una reducción observada del costo bajo una matriz explícita, acompañada del compromiso de intervención y las limitaciones; no una garantía de desempeño bancario en producción.
