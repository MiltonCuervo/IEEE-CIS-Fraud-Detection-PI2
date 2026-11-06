# Etapa 04 — Selección temporal interna de modelos

> Alcance de etapa: este informe conserva la metodología y resultados al cerrar esta fase. La secuencia completa y el test ya evaluado se consolidan en el [informe 08](08_evaluacion_temporal_final.md). Los pasos futuros aquí descritos se leen en ese orden histórico.

## 1. Propósito y alcance

Comparamos ocho configuraciones de XGBoost, LightGBM, CatBoost y regresión logística dentro del bloque train. La exploración del notebook 03 ya fue consultada; esta selección no se presenta como independiente de todas las decisiones de desarrollo. No utiliza las probabilidades ni las métricas externas para calcular el ranking actual.

El protocolo se define antes de los ajustes de esta corrida. Es un registro local verificable, no una preregistración externa. Los archivos de esta etapa usan el prefijo `seleccion_ordinal_` para distinguirlos de los experimentos retirados.

## 2. Ventanas expansivas

Conservamos el manifiesto temporal y retenemos exclusivamente las 380.815 operaciones de train. Los cortes se calculan sobre su duración, no sobre su número de filas:

| Ventana | Pasado para ajustar | Futuro interno para evaluar |
|---|---|---|
| 1 | Inicio hasta el 50 % de duración | Después de siete días de gap, hasta el 75 % |
| 2 | Inicio hasta el 75 % | Después de siete días de gap, hasta el final |

Ambos futuros internos tienen la misma duración teórica: un cuarto del tramo de train menos siete días. La cantidad de operaciones puede variar con la actividad. Darles igual peso representa por igual ambos periodos, sin favorecer automáticamente al más denso. Los límites incluyen instantes completos. La evaluación comienza estrictamente después del corte más el gap; su separación observada respecto al último evento de entrenamiento supera siete días. El segundo pasado incluye la primera ventana evaluada. Eso es admisible en un esquema expansivo: cada ajuste aprende desde cero con información anterior a su propio futuro. Las dos ventanas evaluadas no se superponen, aunque sus estimaciones no son réplicas independientes.

La documentación de [TimeSeriesSplit de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) describe pasados acumulativos, pero su parámetro gap cuenta observaciones, no segundos. Como las transacciones no llegan a intervalos uniformes, implementamos los cortes directamente sobre TransactionDT para representar siete días reales y respetar instantes completos. Es una adaptación de diseño para este dataset, no una configuración equivalente de gap por filas.

Verificamos unicidad, tiempo finito ordenado, pertenencia exclusiva a train y presencia de ambas clases en cada bloque. El lector recorre los CSV originales, pero ningún agregado ni ajuste de esta etapa utiliza los bloques externos.

## 3. Representación constante, preparación por ventana

Conservamos las 432 entradas de la profesora: 431 originales permitidas y has_identity; 37 categóricas ordinales y 395 numéricas. No añadimos los quince atributos del experimento anterior ni seleccionamos columnas por correlación, ausencia o importancia.

Cada pasado aprende sus propias medianas y vocabularios. La categoría desconocida recibe −1 y las ausencias categóricas se normalizan con el token definido. Una columna completamente vacía se conserva con cero y se registra como limitación. La matriz es densa float32. El diagnóstico de categorías desconocidas cubre las 37 variables de la rama categórica; los doce códigos id tratados numéricamente no tienen vocabulario ni reciben −1 por un código nuevo. Las doce discrepancias de códigos id tratados numéricamente permanecen documentadas.

Los ocho candidatos comparten las mismas matrices dentro de cada ventana. La regresión logística añade su escalador ajustado únicamente en ese pasado. CatBoost recibe números ordinales, no categorías nativas. Un orden artificial puede perjudicar a los modelos de maneras distintas; esta comparación no evalúa sus mejores representaciones posibles.

## 4. Configuraciones y presupuesto

| Algoritmo | Referencia | Alternativa regularizada |
|---|---|---|
| XGBoost | Parámetros del 03: profundidad 6, min_child_weight 10 | Profundidad 4, min_child_weight 20, reg_lambda 10 |
| LightGBM | Parámetros del 03: 63 hojas, min_child_samples 50 | 31 hojas, min_child_samples 100, reg_lambda 5 |
| CatBoost | Parámetros del 03: profundidad 6 | Profundidad 4, l2_leaf_reg 10 |
| Logística | C=0,1, saga, máximo 500 iteraciones | C=0,01; mismo solver y límite |

Los predictores de boosting conservan 250 iteraciones y learning_rate=0,05. XGBoost y LightGBM conservan sus parámetros de muestreo; en LightGBM el subsample del 03 no activa bagging porque subsample_freq permanece en cero. CatBoost mantiene desactivada la escritura de archivos automáticos. Los parámetros completos y las semillas están en el protocolo JSON; los algoritmos con control explícito de hilos utilizan cuatro.

Reducir C incrementa la regularización L2 de logística. min_child_weight de XGBoost controla suma de Hessianos; no equivale a min_child_samples de LightGBM. Comparar una alternativa que cambia varios controles no permite aislar el efecto causal de uno de ellos. El presupuesto de dos configuraciones por algoritmo organiza la búsqueda, pero no iguala capacidad ni tiempo de ejecución.

## 5. Criterio de selección

Elegimos la mayor AP media de las dos ventanas, con igual peso por ventana. No seleccionamos la mejor ventana de cada modelo ni cambiamos el criterio a Brier, log-loss o ahorro después de observar los resultados. Una igualdad exacta se resuelve por identificador lexicográfico.

AP es el criterio intermedio de selección del predictor, mientras que costo y ahorro siguen siendo el objetivo principal del proyecto. Elegir por AP no demuestra haber elegido el predictor de menor costo económico. AP evalúa el orden de riesgo sin fijar el corte 0,5. No calibramos dentro de esta etapa: el paso siguiente exige un predictor aprendido con todo train y su propio
Platt de validación temprana. En la ejecución realizada, el ganador coincidió
con el LightGBM del 03: el 05 verificó y reutilizó ambos ajustes existentes,
sin repetir el entrenamiento ni la calibración. Brier y log-loss raw son diagnósticos; no implican que las probabilidades ya estén calibradas ni autorizan decisiones económicas finales. Tampoco ajustamos early stopping con el futuro evaluado.

Reportamos AP individual, media, mínimo, máximo y desviación estándar muestral. Dos ventanas no proporcionan un intervalo de confianza ni una prueba de superioridad estadística. La prevalencia por ventana se conserva para contextualizar AP. Las advertencias de convergencia y los tiempos forman parte de los resultados; no se ocultan ni se usan como un criterio improvisado para descartar candidatos.

## 6. Evidencia reproducible

- `seleccion_ordinal_protocolo.json`: decisiones registradas antes de ajustar, candidatos y hashes de fuentes/entradas.
- `seleccion_ordinal_ventanas.csv`: población, tiempo y huellas de IDs por bloque interno.
- `seleccion_ordinal_folds.csv`: las 16 combinaciones, métricas raw, tiempos y advertencias.
- `seleccion_ordinal_categorias.csv`: categorías desconocidas por variable y ventana.
- `seleccion_ordinal_resumen.csv`: ranking definido por AP media.
- `seleccion_ordinal_ap.png`: resultados separados por ventana.
- `seleccion_ordinal_config.json`: estado completo, configuración seleccionada, versiones y hashes de salidas.

La tabla de ajustes se guarda progresivamente. Sin el registro final completo no se interpreta una corrida parcial como una selección concluida. El hash del notebook cubre celdas ejecutables; el relato puede ampliarse con hallazgos reales sin atribuir resultados a otro código.

## 7. Resultados observados

El registro final tiene estado `complete` y contiene las 16 combinaciones esperadas, sin advertencias de convergencia registradas. Seleccionamos **lightgbm_referencia** según el criterio previamente declarado:

| Configuración | AP ventana 1 | AP ventana 2 | AP media |
|---|---:|---:|---:|
| LightGBM referencia | 0,62154 | 0,55469 | 0,58812 |
| LightGBM regularizada | 0,60543 | 0,53171 | 0,56857 |
| XGBoost referencia | 0,59096 | 0,51876 | 0,55486 |
| XGBoost regularizada | 0,55844 | 0,48562 | 0,52203 |
| CatBoost referencia | 0,55739 | 0,47283 | 0,51511 |
| CatBoost regularizada | 0,53243 | 0,45002 | 0,49123 |
| Logística referencia | 0,42472 | 0,31666 | 0,37069 |
| Logística regularizada | 0,42470 | 0,31666 | 0,37068 |

LightGBM referencia lidera ambas ventanas. Todos los candidatos tienen menor AP en la segunda; esto describe sensibilidad entre periodos y poblaciones, no demuestra por sí solo deriva de concepto. Las prevalencias evaluadas son aproximadamente 4,15% y 3,57%, respectivamente. La ventaja media sobre la siguiente configuración es 0,01955; dos ventanas no permiten atribuirle significación estadística.

Estas AP raw internas no son una repetición de la AP 0,5044 calibrada y externa del 03: cambian periodos y poblaciones de ajuste/evaluación. Ninguna constituye evidencia nueva de ahorro con LightGBM. Conservamos el registro del 04 y sus resultados al recuperar artefactos locales.

## 8. Continuación

La salida es una configuración seleccionada, no un predictor entrenado con todo train. El paso posterior verificará si la configuración ya dispone de un ajuste coincidente en el 03; de ser así, podrá reutilizarlo tras comprobar su procedencia. En caso contrario se ajustará con todo train. El calibrador deberá pertenecer al mismo predictor y aprenderse solo en validación temprana; el diagnóstico posterior conservará su carácter de desarrollo. La comparación económica se reconstruirá con sus propias probabilidades. No reutilizaremos el Platt de otro modelo.

Ca no interviene aquí. El test permanece sin puntuarse y mantenemos declarada la consulta histórica de sus etiquetas agregadas. Si el ganador cambia, cambia también la cadena experimental posterior, no las métricas históricas del 03.

La configuración ganadora coincide exactamente con los parámetros LightGBM del 03. La etapa 05 podrá utilizar ese modelo y su propio Platt solo después de verificar su procedencia. Si se reconstruyeron, verificaremos también el recibo de recuperación y sus pruebas de equivalencia; no compararemos sus bytes únicamente con los hashes de los binarios originales. Véase [recuperación local](recuperacion_local_y_trazabilidad.md).


## 9. Diagnósticos completos y entradas de cada algoritmo

La revisión del 4 de noviembre incorpora las seis métricas propuestas en el 03 a los ajustes internos: AP, Brier y log-loss raw, junto con recall, precision y bloqueo de legítimas con corte fijo 0,5. Conservamos TP, FP, FN, TN y cantidad de intervenciones por candidato y ventana. El ranking sigue usando exclusivamente AP media; los diagnósticos no cambian retrospectivamente la regla de selección.

Volvimos a ejecutar los 16 ajustes con las fuentes actualizadas. AP, Brier, log-loss y probabilidad media reproducen la corrida anterior dentro de atol=rtol=10⁻⁸; la configuración seleccionada y su AP media permanecen iguales. Es una nueva corrida que amplía los diagnósticos, no evidencia independiente ni métricas añadidas a una ejecución histórica sin recalcular.

| Configuración | AP media raw | Recall medio a 0,5 | Precision media a 0,5 | Bloqueo medio de legítimas |
|---|---:|---:|---:|---:|
| lightgbm_referencia | 0.58812 | 34.32% | 87.25% | 0.20% |
| lightgbm_regularizada | 0.56857 | 33.63% | 87.48% | 0.19% |
| xgboost_referencia | 0.55486 | 32.28% | 87.42% | 0.19% |
| xgboost_regularizada | 0.52203 | 28.48% | 87.06% | 0.17% |
| catboost_referencia | 0.51511 | 29.07% | 88.59% | 0.15% |
| catboost_regularizada | 0.49123 | 26.85% | 87.40% | 0.16% |
| logistica_referencia | 0.37069 | 16.44% | 82.62% | 0.14% |
| logistica_regularizada | 0.37068 | 16.44% | 82.62% | 0.14% |

Cada tasa media da igual peso a las dos ventanas. No es la tasa de una matriz de confusión combinada. Las métricas son raw y se obtienen en futuros internos de train; no se comparan por diferencia directa con las métricas Platt de validación posterior, porque cambian tanto población como preparación aprendida y ajuste.

Las métricas no dan un ganador universal. En la primera ventana, CatBoost referencia alcanza precision 89,11% frente a 87,90% de LightGBM referencia al corte 0,5, mientras LightGBM tiene mayor AP y recall. Una precision mayor entre las alertas no significa detectar una proporción mayor de todos los fraudes. Esta diferencia refuerza la lectura conjunta y no modifica el criterio AP previamente fijado.

La tabla por ajuste también registra input_rows, evaluation_rows, input_features, input_dtype, input_representation y additional_scaling. Los tres modelos de boosting reciben las mismas 432 columnas ordinales/imputadas por ventana. Logística recibe esa matriz a través de un pipeline con StandardScaler ajustado solo en su pasado. No seleccionamos atributos diferentes para cada algoritmo. CatBoost sigue sin categorías nativas y permanecen las doce discrepancias de tipos.

El 04 exporta una configuración. El 05 comprueba si el modelo del 03 coincide y tiene su propio calibrador antes de adoptarlo. Reutilizar ese ajuste coincidente es válido; los modelos de folds internos no se presentan como predictores aprendidos sobre todo train.
