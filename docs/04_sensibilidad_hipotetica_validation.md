# Sensibilidad del costo administrativo en validación

## Propósito

Exploramos la matriz definida en el anteproyecto: intervenir cuesta Ca en ambas clases; aprobar fraude cuesta su monto y aprobar legítimas cuesta cero. Usamos probabilidades calibradas de la ventana posterior de validación. No entrenamos aquí el predictor ni accedemos a test.

## Escenarios e interpretación

Variamos Ca en 1, 5, 10, 20, 50 y 100 unidades del dataset. Son valores ilustrativos, no costos observados ni escenarios finales aprobados. La grilla es absoluta y no depende de la mediana de la ventana evaluada. Una vez justificado el rango operativo, podremos sustituirla sin alterar la matriz.

BMR interviene si `p_i A_i > Ca`; la política fija conserva el corte 0,5. Para cada Ca medimos el costo administrativo de todas las intervenciones y el monto fraudulento aprobado. Reportamos ahorro respecto al menor costo entre aprobar todo e intervenir todo, y por separado mejora respecto a la política fija.

Al elevar Ca, BMR reduce las intervenciones y su recall no aumenta. La regla fija conserva sus acciones, pero también cambia su costo administrativo. El costo total y el ahorro no necesariamente son monótonos: dependen de los montos de fraude y de la referencia, que cambia con Ca.

## Evidencia reproducible

El notebook `04_sensibilidad_hipotetica_validation.ipynb` y `scripts/escenarios_hipoteticos_validation.py` utilizan el mismo evaluador en `src/fraud_cost/scenarios.py`. La entrada es `data/interim/validation_scores.parquet`, producida por notebook 03. Exportamos únicamente la tabla `results/tables/escenarios_hipoteticos_validation.csv` y la figura `results/figures/ahorro_hipotetico_validation.png`.

Las cifras antiguas de ahorro entre 14,49% y 52,89% correspondían a otra matriz y se retiran de este informe. La tabla actual se interpreta por Ca y estrategia; no contiene λ ni B.

## Resultados de la corrida del 5 de octubre de 2026

Ejecutamos de nuevo 03 y 04 sobre los CSV locales y el manifiesto existente. La ventana posterior de validación contiene 42.047 transacciones y 1.517 fraudes. Su monto fraudulento total es 254.536,81 unidades. El modelo calibrado obtiene AP 0,4681, Brier 0,02500 y recall 27,55% al corte 0,5. Estas cifras corresponden al modelo de referencia de esta corrida, no a una búsqueda final de hiperparámetros.

| Ca ilustrativo | Costo BMR | Ahorro BMR vs. referencia trivial | Reducción BMR vs. fijo | Recall BMR | Legítimas intervenidas |
|---:|---:|---:|---:|---:|---:|
| 1 | 25.946,72 | 38,29% | 86,92% | 85,43% | 37,07% |
| 5 | 66.807,82 | 68,22% | 66,69% | 64,40% | 11,77% |
| 10 | 88.439,28 | 65,25% | 56,50% | 53,66% | 6,23% |
| 20 | 117.791,22 | 53,72% | 43,57% | 38,76% | 2,99% |
| 50 | 163.253,10 | 35,86% | 27,46% | 19,38% | 0,88% |
| 100 | 192.356,93 | 24,43% | 23,75% | 10,22% | 0,32% |

Con Ca=10, BMR interviene 3.339 operaciones: 814 fraudes y 2.525 legítimas. Su costo administrativo es 33.390 y el monto de los fraudes aprobados es 55.049,28. La suma produce 88.439,28. La política fija interviene 544 operaciones y cuesta 203.296,60. La referencia trivial es aprobar todo, cuyo costo es 254.536,81; intervenir todo costaría 420.470.

La reducción de 56,50% frente al fijo y el ahorro de 65,25% frente a la referencia describen comparaciones diferentes. El mayor ahorro porcentual de esta cuadrícula aparece con Ca=5, pero no significa que debamos elegir ese costo administrativo: Ca representa un supuesto operativo externo, no un hiperparámetro que ajustamos para maximizar el porcentaje obtenido.

Con Ca=1, la política fija cuesta más que intervenir todo y su ahorro frente a la referencia es negativo (-371,85%). Esto no es un error de cálculo: muestra por qué conviene mantener visible la referencia trivial y no reportar exclusivamente mejora contra el clasificador.

Las versiones y hashes de los CSV se registraron en `data/interim/data_provenance.json`. Los hashes de notebooks ejecutados y módulos económicos, la grilla y la confirmación de test no evaluado están en `data/interim/validation_run_metadata.json`. Ambos son artefactos locales ignorados por Git. En aquella corrida se utilizó un entorno temporal de Python 3.12 por un problema del intérprete local. Las etapas posteriores verificaron y utilizaron la `.venv` del repositorio; la situación de aquella corrida no describe el estado actual del entorno.

## Límites

Este análisis es exploratorio en validación. Sus costos son simulados y la unidad monetaria sigue por confirmar. La matriz supone que intervenir evita la pérdida fraudulenta. La calibración imperfecta puede producir ahorro realizado negativo incluso cuando BMR elige el riesgo esperado menor. La mejora frente a una política fija no garantiza mejora frente a la mejor política trivial.

El árbol requiere entrenamiento por Ca. El notebook 07 lo incorpora a una comparación hipotética con el predictor seleccionado y calibrado del 06, manteniendo separadas las salidas de esta referencia histórica. Los escenarios finales siguen pendientes de acuerdo. La consulta histórica de etiquetas agregadas del periodo test se declara en el protocolo; el notebook 04 no accede a ese bloque.
