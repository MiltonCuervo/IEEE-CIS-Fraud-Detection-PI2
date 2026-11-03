# Etapa 03 — Comparación de modelos y decisiones basadas en costos

## 1. Pregunta y alcance

Con la partición del notebook 02 buscamos establecer una referencia probabilística y económica reproducible. Seguimos la preparación ordinal y los cuatro predictores incorporados por la profesora: XGBoost, LightGBM, CatBoost y regresión logística. Sus parámetros son fijos; esta etapa no selecciona el predictor final.

La consolidación del 2 de noviembre corrige explicaciones, guarda la imputación en el preprocesador, preserva el monto económico original y recupera las exportaciones. Estos cambios requieren una nueva corrida: no atribuimos las métricas del commit de la profesora a estas fuentes modificadas. Las salidas ejecutadas y su registro son la evidencia numérica vigente.

## 2. Poblaciones y preparación

Train contiene 380.815 operaciones y 13.011 fraudes. Validación temprana tiene 42.046 operaciones y 1.558 fraudes; validación posterior tiene 42.047 y 1.517. Se conserva el manifiesto y los gaps de siete días. Las operaciones de un mismo instante no se reparten entre calibración y diagnóstico.

El preprocesador aprende exclusivamente en train. Conserva 432 entradas: 431 originales permitidas y has_identity; 37 se codifican ordinalmente y 395 se imputan numéricamente. Categorías desconocidas reciben −1; la ausencia categórica usa un token explícito. No agregamos indicadores ni los quince atributos del experimento anterior. Las medianas se calculan vector por vector para reducir memoria y se guardan para su reutilización. El [diccionario](diccionario_variables_y_flujo_modelo.md) detalla familias, exclusiones, orden, vocabularios y las doce discrepancias de tipos pendientes.

CatBoost recibe códigos como números, sin categorías nativas. La regresión logística añade escalado aprendido en train. Esta representación condiciona la comparación: sus resultados no prueban el desempeño máximo de cada algoritmo. Conservamos la configuración de LightGBM de la profesora; `subsample=0.8` con `subsample_freq=0` no activa muestreo de filas. No afirmamos que los presupuestos o las capacidades sean idénticos entre bibliotecas.

## 3. Aprendizaje y calibración

Cada predictor aprende isFraud con train. Su Platt se ajusta sobre el logit de sus propias probabilidades de validación temprana, recortadas a [10⁻⁶,1−10⁻⁶]. Después no reajustamos el predictor. Evaluamos raw y Platt en las mismas filas posteriores mediante AP, Brier, log-loss y curvas de calibración. La tabla comparativa incorpora precision, recall y bloqueo de legítimas a 0,5.

AP no equivale necesariamente al área trapezoidal de precision–recall. Brier y log-loss no son pruebas aisladas de calibración suficiente, especialmente con desbalance. Una advertencia de convergencia de logística debe leerse como limitación del ajuste, no ocultarse ni interpretarse como rendimiento óptimo.

### Cómo leer las métricas

Precision es la fracción de fraudes entre las operaciones marcadas: TP/(TP+FP). Recall es la fracción de fraudes reales intervenidos: TP/(TP+FN). La tasa de bloqueo de legítimas es FP/(FP+TN). TP y FN corresponden a fraude real intervenido y aprobado; FP y TN, a legítimas intervenidas y aprobadas. Al comparar políticas, estos conteos describen sus acciones y no necesariamente la etiqueta del clasificador a 0,5.

Brier promedia (p−y)²; log-loss penaliza especialmente probabilidades confiadas que contradicen la etiqueta. En ambos, menor es mejor, pero los valores absolutos deben leerse con la prevalencia. AP resume precision a medida que aumenta recall y no depende de fijar el corte 0,5. La calibración pregunta si riesgos anunciados y frecuencias observadas son compatibles; discriminación y calibración son propiedades diferentes.

## 4. Matriz y regla económica

| Estado observado | Aprobar | Intervenir |
|---|---:|---:|
| Fraude | A = monto original | Ca |
| Legítima | 0 | Ca |

Intervenir significa bloquear y revisar; asumimos que evita toda la pérdida fraudulenta y cuesta Ca en ambas clases. No modelamos capacidad, recuperación o revisión imperfecta. La moneda no está verificada; expresamos montos y costos en unidades del dataset.

Ca es el costo administrativo por intervención; A_i es el monto original de la operación i y p_i su probabilidad estimada de fraude. Si A_i>0, p*=Ca/A_i es el límite económico por operación, distinto del corte clasificatorio fijo. El corte clasificatorio 0,5 se conserva. Para BMR, R(aprobar)=pA y R(intervenir)=Ca. Intervenimos cuando pA>Ca y aprobamos en empate. La comparación multiplicativa evita dividir por cero; A≤Ca no justifica intervenir con probabilidades válidas.

El árbol aprende la acción 1[yA>Ca] y pondera cada error por abs(yA−Ca). Esa transformación reproduce el costo empírico como error ponderado más una constante. Su ajuste mediante impureza es una aproximación; no garantiza el mínimo global. Ajustamos un árbol para cada Ca únicamente con train y sus montos.

## 5. Comparación y referencias

Comparamos regla fija, BMR y árbol con XGBoost + Platt sobre las mismas filas. No confundimos la comparación de cuatro predictores con una evaluación económica de los cuatro: esa extensión todavía está pendiente.

```text
C_total = Ca × intervenciones + monto de fraudes aprobados
C_aprobar_todo = monto total de fraude
C_intervenir_todo = N × Ca
C_referencia = min(C_aprobar_todo, C_intervenir_todo)
ahorro_referencia = 1 − C_total/C_referencia
mejora_fijo = 1 − C_total/C_fijo
```

Las referencias son diferentes y pueden producir ahorros negativos. Para Ca=5, intervenir todo cuesta 210.235, menor que el monto fraudulento aproximado de 254.537; para Ca=10, aprobar todo es la mejor referencia trivial. No afirmamos que un rango de Ca sea operativo sin evidencia de costos aplicable al dataset.

Los escenarios 1,5,10,20,50,100 son hipotéticos. Su elección final y justificación quedan pendientes. BMR minimiza riesgo esperado con las probabilidades suministradas; no garantiza el menor costo realizado en toda muestra o periodo.

## 6. Archivos y reproducción

Ejecutamos 01, 02 y 03 en ese orden con las dependencias declaradas. El 03 exporta:

- `comparacion_modelos_validation.csv`: métricas Platt de los cuatro predictores.
- `modelos_calibracion_validation.csv`: diagnóstico raw/Platt.
- `costos_validation_sensibilidad.csv`: tres estrategias, seis escenarios y componentes.
- `modelo_variables_ordinal.csv` y `modelo_vocabulario_ordinal.csv`: tratamiento real por variable.
- `comparacion_modelos_config.json`: parámetros, versiones, advertencias, hashes y test_evaluated=false.
- `comparacion_modelos.png` y `costos_validation_sensibilidad.png`: figuras utilizadas.
- Artefactos locales: scores por operación, referencia XGBoost, paquete de cuatro modelos y árboles por Ca.

Los nombres y el orden de columnas de la tabla económica proceden de `src/fraud_cost/costs.py`. Evitamos mantener una segunda matriz económica dentro del notebook. Las pruebas versionadas verifican imputación reutilizable, categorías nuevas/ausentes, serialización, empates y equivalencia de los pesos del árbol con el costo realizado.

## 7. Transición y continuación

Retiramos de la estructura activa los notebooks 04–09 del pipeline anterior, sus informes y sus resultados. No borramos CSV originales ni modificamos el manifiesto. Git conserva los archivos retirados en `089df7f`; los modelos locales antiguos siguen ignorados, pero no forman parte del flujo actual. También retiramos los registros automáticos de CatBoost y desactivamos su escritura en nuevas corridas.

Después de esta exploración, el notebook 04 comparó ocho configuraciones de los cuatro modelos con dos ventanas temporales internas de train y seleccionó LightGBM referencia por AP media. La exploración posterior ya fue consultada y forma parte del desarrollo; no afirmamos que las decisiones siguientes sean independientes de ella. La etapa 05 deberá formalizar el predictor seleccionado y diagnosticar su calibración; la comparación económica y estabilidad se reconstruirán sobre sus probabilidades.

El test permanece sin puntuar y no incluimos una celda para habilitarlo en esta exploración. El EDA y la comparación histórica de gaps consultaron etiquetas agregadas del periodo final: mantenemos esa limitación explícita. La evaluación final requiere congelar primero el protocolo.

## 8. Hallazgos de la corrida consolidada

Ejecutamos el notebook completo con la preparación ordinal consolidada. El registro `comparacion_modelos_config.json` identifica las fuentes y salidas de esta corrida. Se mantuvieron 380.815 operaciones de train, 42.046 de calibración y 42.047 de diagnóstico. El manifiesto no se modificó y el test no se puntuó.

| Modelo | AP | Recall a 0,5 | Precision a 0,5 | Brier | Log-loss |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.5044 | 0.3085 | 0.7852 | 0.02406 | 0.09645 |
| XGBoost | 0.4673 | 0.2769 | 0.7527 | 0.02507 | 0.10181 |
| CatBoost | 0.4251 | 0.2353 | 0.7376 | 0.02639 | 0.10814 |
| Regresión logística | 0.2910 | 0.1213 | 0.6996 | 0.02983 | 0.12615 |

Observamos que **LightGBM obtiene la mayor AP en esta comparación**. Es evidencia para estudiar este candidato dentro de train, no una elección final ni una prueba de mayor ahorro: los costos siguientes se calcularon únicamente con XGBoost. La regresión logística es una referencia sobre códigos ordinales y parámetros fijos; no deducimos de su posición que hayamos demostrado una estructura causal o una propiedad universal del problema.

Al comparar raw y Platt sobre las mismas filas, Brier disminuye en 4 de los cuatro modelos y log-loss en 4. La tabla `modelos_calibracion_validation.csv` conserva los ocho registros. La calibración se interpreta por predictor; no trasladamos automáticamente un calibrador ni una conclusión a los demás modelos.

La prevalencia observada en validación posterior es 3.61%. Las probabilidades medias calibradas van de 3.03% a 3.13%: todavía observamos una subestimación global en esta muestra. Mejorar Brier y log-loss no demuestra una calibración perfecta ni uniformidad por rango de monto. AP permanece igual entre raw y Platt en los cuatro modelos de esta corrida, consistente con conservar el orden de riesgo al transformar las probabilidades.

| Estrategia, Ca=10 | Costo total | Recall | Bloqueo de legítimas |
|---|---:|---:|---:|
| Regla fija | 204,786.21 | 0.2769 | 0.34% |
| BMR | 88,182.74 | 0.5438 | 6.38% |
| Árbol ponderado | 125,992.24 | 0.4054 | 6.60% |

Con Ca=10, BMR interviene 3,410 operaciones. Su costo se descompone en 34,100.00 de intervención y 54,082.74 de fraude aprobado. La mejora frente al corte fijo es 56.94%; el ahorro frente a la mejor referencia trivial es 65.36%. Son porcentajes con denominadores distintos, no dos estimaciones del mismo indicador. Las cantidades monetarias de estas tablas están expresadas en unidades del dataset; los separadores numéricos siguen el formato de las exportaciones.

El beneficio económico se acompaña de un cambio en intervenciones sobre legítimas. Por eso no lo interpretamos solo como una mejora de recall. La tabla completa permite revisar los seis escenarios sin elegir Ca retrospectivamente por el ahorro observado.

### Advertencias del ajuste

No se registraron advertencias durante el ajuste de los cuatro predictores.

La ausencia de advertencias no certifica optimalidad de hiperparámetros ni calibración suficiente.

## 9. Recuperación de los artefactos locales

Las cifras anteriores y `comparacion_modelos_config.json` corresponden a la corrida histórica. Un pull no incorpora sus binarios ignorados. Si no están disponibles, `scripts/recover_reference_artifacts.py` reconstruye el 03 en una copia aislada con sus parámetros y versiones principales, compara las cinco tablas y comprueba predicciones sin puntuar test. Publica solo después de verificar y conserva los archivos anteriores como respaldo.

La reconstrucción utiliza Python 3.12 frente al 3.11.9 original y tiene hashes propios. Su recibo separado no sustituye los registros 03/04. Las métricas históricas siguen identificadas como tales; equivalencia numérica dentro de tolerancias declaradas no significa identidad binaria ni evidencia independiente. Antes de reutilizar el paquete ejecutamos `--verify-only` o `verify_reference_artifacts`.

Los mensajes guardados del 03 que mencionan selección en «NB-05» pertenecen a la numeración anterior. La etapa de selección vigente es el 04. Conservamos el código ejecutable histórico para que sus hashes sigan verificables; esta nota y el README indican la secuencia actual.
