# Protocolo de costos y evaluación

## 1. Formulación de la propuesta

La metodología entregada define dos acciones: aprobar e intervenir. Intervenir significa bloquear y enviar a revisión y cuesta `Ca` independientemente de la etiqueta. Bajo este modelo simplificado, la intervención evita la pérdida fraudulenta; su eficacia imperfecta, recuperación y capacidad diaria quedan fuera del experimento principal.

| Estado real | Aprobar | Intervenir |
|---|---:|---:|
| Fraude | `A_i = TransactionAmt_i` | `Ca` |
| Legítima | `0` | `Ca` |

Así, `C(TP)=C(FP)=Ca`, `C(FN)=A_i` y `C(TN)=0`. Estos términos describen la acción frente a la etiqueta real. La propuesta se consulta en su sección 9, páginas 3 y 4.

La implementación principal mantiene esta matriz. Los parámetros `B`, `λ`, costo residual de bloquear fraude, sensibilidad/especificidad de revisión y capacidad diaria se retiraron porque correspondían a otro modelo operativo. Los resultados anteriores de esa matriz no son evidencia del experimento actual.

## 2. Probabilidad, etiqueta y acción

### 2.1. Antes de decidir debemos especificar la entrada

La línea base del notebook 03 utiliza las 431 columnas originales permitidas y has_identity, sin los quince atributos ampliados del 05. Retira de X TransactionID, TransactionDT, isFraud y split; conserva las tres primeras fuera de X para alinear, ordenar y evaluar. No aplica selección automática por correlación o ausencia. Las categorías explícitas y las columnas de texto pasan a one-hot; las numéricas se imputan con medianas e indicadores aprendidos solo en train. El notebook 03 tiene su propio preprocesador; no le atribuimos automáticamente las 2.642 columnas ni todos los detalles de implementación del reajuste del 06.

El predictor seleccionado del 06 añade los quince atributos y utiliza 447 entradas, 37 categóricas y 410 numéricas, con una matriz de 2.642 columnas. Su [diccionario de variables y contrato de flujo](diccionario_variables_y_flujo_modelo.md) identifica cada columna y la discrepancia pendiente en doce códigos de identidad. No mezclamos el modelo baseline con selected ni usamos un calibrador de otro predictor.

### 2.2. Qué produce el predictor y qué recibe cada política

XGBoost devuelve probabilidades por clase; usamos la columna de fraude como p_raw. Platt transforma el logit recortado de esa probabilidad para obtener p_calibrada. La etiqueta convencional se obtiene después con >=0,5. BMR consume p_calibrada, TransactionAmt original y Ca, no la etiqueta binaria ni isFraud. El árbol recibe los atributos transformados y entrega una acción aprendida, no una probabilidad calibrada de fraude.

TransactionID asocia cada salida con su transacción. isFraud se utiliza después de decidir para medir, aunque aparezca junto a los scores en archivos de evaluación. Su presencia en una tabla de resultados no lo convierte en una entrada de inferencia. Ca no es una columna original ni modifica la pérdida logística de XGBoost: se usa al decidir BMR y al construir objetivos/pesos del árbol en train.

### 2.3. Regla y riesgo esperado

XGBoost estima una probabilidad que calibramos antes de decidir. El corte clasificatorio 0,5 se mantiene: la estrategia convencional interviene cuando `p_i >= 0,5`. BMR calcula riesgos esperados:

```text
R(aprobar) = p_i A_i
R(intervenir) = p_i Ca + (1-p_i) Ca = Ca
```

Elegimos intervenir si `p_i A_i > Ca`; en igualdad aprobamos. Para `A_i>0`, esto equivale a `p_i > Ca/A_i`. Implementamos la comparación multiplicativa para evitar división por cero. Si `A_i=0`, aprobamos; si `A_i<=Ca`, intervenir no reduce el riesgo. Esta regla utiliza solo probabilidad, monto y Ca, nunca la etiqueta real al decidir.

## 3. Costo realizado y ahorro

Después de tomar las acciones, utilizamos las etiquetas para evaluar:

```text
C_total = Ca × número de intervenciones
          + suma de montos de fraudes aprobados
C_aprobar_todo = suma de montos de todos los fraudes
C_intervenir_todo = N × Ca
C_referencia = min(C_aprobar_todo, C_intervenir_todo)
S_referencia = 1 - C_total/C_referencia
```

La referencia es la mejor de las dos políticas triviales, como establece el anteproyecto. Se calcula sobre las mismas filas y con el mismo Ca para todas las estrategias. Usa etiquetas exclusivamente al evaluar, no para construir acciones. El ahorro puede ser negativo y es indefinido si la referencia vale cero; en ese caso registramos `NaN`.

También reportamos `C_convencional-C_estrategia` y su proporción respecto al costo convencional si este es positivo. Esta mejora frente al corte 0,5 es diferente del ahorro respecto a la referencia trivial. Descomponemos siempre costo administrativo y monto fraudulento aprobado.

## 4. Estrategias y árbol ponderado

Comparamos la política fija, BMR y el árbol sensible al costo sobre las mismas observaciones. La contingencia con ponderación está contemplada en la página 4 de la propuesta; declaramos que el árbol ajustado por impureza ponderada es una aproximación al método de costos por ejemplo. La etapa 07 implementa una comparación exploratoria de las tres políticas con Ca ilustrativo y el predictor del 06; no equivale a aprobar escenarios finales.

Para respetar esta matriz no basta ponderar fraudes por su monto. Con etiqueta conocida, la acción de menor costo es intervenir si `y_i A_i > Ca`. La diferencia entre costos es `|y_i A_i-Ca|`:

- fraude con `A_i>Ca`: objetivo de acción intervenir y peso `A_i-Ca`;
- fraude con `A_i<Ca`: objetivo de acción aprobar y peso `Ca-A_i`;
- legítima: objetivo de acción aprobar y peso `Ca`;
- igualdad: peso cero; ambas acciones cuestan igual.

El costo de cualquier acción equivale al mínimo por fila más el peso cuando la acción difiere de la óptima. Ese mínimo es una constante para el entrenamiento. Ajustamos un árbol por Ca con estos objetivos de acción y pesos normalizados por una constante común. Sus predicciones representan acciones, no una nueva etiqueta de fraude. La probabilidad del XGBoost sigue disponible para evaluar discriminación y calibración.

## 5. Protocolo temporal

Conservamos train, gaps, validación y test del manifiesto. Ajustamos preprocesamiento y predictor solo con train. El primer modelo tiene hiperparámetros fijos. El notebook 05 compara atributos y cuatro configuraciones en dos ventanas expansivas internas de train, con preprocesamiento propio por fold y siete días de separación. Selecciona por AP media, sin consultar validación externa ni test; su diseño y resultados se explican en `docs/05_preparacion_y_seleccion_temporal.md`.

La etapa 06 reajusta el predictor elegido con todo train y ajusta un calibrador nuevo sobre sus propias puntuaciones tempranas. Conserva el estado histórico de train congelado para ambas mitades de validation y guarda artefactos con nombres distintos. El diagnóstico está en `docs/06_reajuste_y_calibracion.md`. No reutilizamos un calibrador entrenado para otro predictor ni mezclamos las métricas internas con las de validación posterior.

Usamos validación temprana para calibrar y la posterior para diagnóstico y exploración de políticas. El predictor que genera ambas puntuaciones permanece fijo. Los gaps no participan en entrenamiento ni evaluación. El test no se puntúa mientras `RUN_FINAL_TEST=False`; antes de habilitarlo fijamos predictor, calibrador, escenarios de Ca y árboles correspondientes.

El EDA y la comparación histórica de gaps consultaron etiquetas agregadas del periodo final. Declaramos esa limitación; la evaluación temporal no se presenta como un bloque estrictamente ciego desde el inicio. Sus resultados no se usan para ajustar modelos ni elegir costos.

## 6. Escenarios, unidades y métricas

La matriz está definida por la propuesta, pero IEEE-CIS no aporta un costo administrativo observado. Notebook 04 explora `Ca ∈ {1,5,10,20,50,100}` como valores ilustrativos absolutos en unidades del dataset. No se deducen de la mediana de validación ni se presentan como estimación operativa. Antes de la evaluación final acordamos rango y justificación y registramos su referencia.

La propuesta plantea reportar USD, pero la moneda debe verificarse antes de poner ese rótulo en resultados. Ca y TransactionAmt deben compartir escala; mientras tanto usamos unidades del dataset. El costo evaluado es un proxy bajo los supuestos del estudio, no una pérdida bancaria observada.

Reportamos costo, componentes, ahorro con ambas referencias, recall de fraude, tasa de legítimas intervenidas y recall ponderado por monto. Evaluamos las probabilidades con AP, Brier score, log-loss y curva de calibración. `average_precision_score` calcula average precision; no necesariamente coincide con el área trapezoidal AUC-PR. La comparación de acciones no modifica el orden de las probabilidades del modelo base.

## 7. Organización y reproducción

`src/fraud_cost/costs.py` concentra decisiones, costos, comparación y ponderación. `scenarios.py` valida scores y recorre Ca. Los notebooks conservan las preguntas, parámetros y lectura de salidas. Las corridas históricas verificaron las cuatro celdas de la matriz, empates, montos cero, referencias y equivalencia del error ponderado. La carpeta tests fue retirada en la organización actual; no se afirma que aquella suite esté disponible para ejecutar hoy. La auditoría de variables no modifica esa matriz ni reemplaza sus verificaciones.

Ejecutamos 03 antes de 04 para regenerar la referencia y su sensibilidad. La etapa 06 guarda `validation_scores_selected.parquet` y `selected_xgboost_platt.joblib`; la etapa 07 los verifica y utiliza conjuntamente sin sobrescribir la referencia. Reconstruye atributos de train con pasado estricto, aplica el preprocesamiento ya aprendido y ajusta árboles solo con train. El diagnóstico raw/calibrado de BMR es separado y no elige retrospectivamente un calibrador. Los CSV y figuras históricos de la matriz con B y λ no son evidencia del protocolo actual. Los valores finales de Ca siguen pendientes, por lo que la evaluación de test permanece cerrada.
