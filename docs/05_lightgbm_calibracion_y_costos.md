# Etapa 05 — Adopción de LightGBM, calibración y decisiones por costos

> Alcance de etapa: este informe conserva la metodología y resultados al cerrar esta fase. La secuencia completa y el test ya evaluado se consolidan en el [informe 08](08_evaluacion_temporal_final.md). Los pasos futuros aquí descritos se leen en ese orden histórico.

## 1. Pregunta y alcance

Después de seleccionar `lightgbm_referencia` por AP media interna (0,588117), necesitamos disponer de un predictor completo, comprobar sus probabilidades y evaluar acciones económicas. La configuración coincide con el LightGBM ya entrenado en el 03; reutilizamos ese modelo y su propio Platt tras verificar su procedencia. No repetimos ajustes idénticos ni presentamos la reproducción de sus métricas como una confirmación independiente.

La etapa acepta artefactos originales o una reconstrucción con recibo y evidencia verificados por `verify_reference_artifacts`. Los registros históricos 03/04 permanecen intactos. Una selección distinta exigiría revisar este protocolo: no ajustamos otra configuración de manera automática.

## 2. Entrada y aprendizaje previamente realizado

El modelo se aprendió con las 380.815 operaciones de train. El preprocesador conserva 432 entradas: 431 originales permitidas y has_identity; 37 categóricas ordinales y 395 numéricas imputadas. Sus vocabularios y medianas son de train. El LightGBM adoptado tiene 250 estimadores, tasa de aprendizaje 0,05, 63 hojas y min_child_samples=50; los parámetros completos proceden del registro del 04.

El Platt asociado se aprendió en las 42.046 operaciones tempranas de validación sobre el logit de sus propias probabilidades, recortadas a [10⁻⁶,1−10⁻⁶]. No procede de XGBoost ni recibe las 432 variables originales. En esta etapa no volvemos a llamar fit sobre predictor, preprocesador, calibrador ni árboles.

Conservamos el tratamiento de las doce variables id semánticamente categóricas que siguen numéricas. No agregamos atributos históricos, logaritmos ni indicadores nuevos. El [diccionario vigente](diccionario_variables_y_flujo_modelo.md) identifica estas decisiones y sus límites.

## 3. Población de diagnóstico

Evaluamos exactamente las 42.047 operaciones posteriores y sus 1.517 fraudes. Verificamos igualdad de TransactionID, TransactionDT, TransactionAmt e isFraud con los scores XGBoost del 03. El monto original se conserva en float64 para costos; la matriz predictiva es float32. Los instantes simultáneos no se reparten entre la mitad temprana y la posterior.

En esta etapa no puntuamos test ni agregamos sus etiquetas para resumir,
ajustar o elegir parámetros; el lector común sí recorre los CSV antes de
retener las filas de validación. Conservamos la limitación histórica de consulta de etiquetas agregadas durante EDA/partición. Validación posterior ya se ha utilizado como desarrollo; sus métricas no son una evaluación independiente del diseño completo.

## 4. Diagnóstico de probabilidades

Comparamos raw y Platt con AP, Brier, log-loss, probabilidad media y errores a 0,5. Las métricas deben reproducir las del LightGBM del 03 dentro de atol=rtol=10⁻⁸, tolerancias fijadas antes de inferencia. Si no coinciden, detenemos la adopción en lugar de cambiar las tolerancias retrospectivamente.

AP evalúa discriminación, mientras que Brier y log-loss consideran la calidad probabilística. Ninguna demuestra por sí sola calibración suficiente. Los diez grupos por cuantiles de Platt son compartidos para raw y calibrada; cada fila informa población, fraude y media de ambas probabilidades. No constituyen intervalos de confianza ni diagnostican automáticamente la cola de riesgo extremo.

Para conectar con el objetivo económico agrupamos montos en [0,50), [50,100), [100,250), [250,500), [500,1000) y [1000,∞). Comparamos frecuencia observada, probabilidad media, monto fraudulento observado y exposición esperada ΣpA. La igualdad global no garantizaría calibración dentro de cada banda; una diferencia muestral tampoco prueba por sí sola sesgo poblacional.

## 5. Tres políticas, una misma matriz

| Estado observado | Aprobar | Intervenir |
|---|---:|---:|
| Fraude | A: monto original | Ca |
| Legítima | 0 | Ca |

La regla fija interviene con p≥0,5; el corte no se optimiza. BMR interviene si pA>Ca y aprueba en empate; con A≤Ca intervenir no reduce estrictamente el riesgo esperado. El árbol ponderado predice acciones aprendidas en train mediante objetivo 1[yA>Ca] y peso abs(yA−Ca). Su ajuste por impureza es una aproximación, no un mínimo global garantizado.

Reutilizamos los seis árboles del 03 porque sus atributos, preprocesador, etiquetas, montos y Ca no cambiaron. Un árbol no utiliza la probabilidad de XGBoost o LightGBM como entrada. Comprobamos que conserve el costo realizado histórico en las mismas filas. La comparación de regla fija/BMR sí cambia porque utiliza las probabilidades LightGBM.

Mantenemos Ca=1,5,10,20,50,100 como escenarios hipotéticos en unidades del dataset. No seleccionamos Ca por el ahorro observado ni afirmamos que representan gastos operativos. Asumimos que intervenir evita toda pérdida fraudulenta; quedan fuera recuperación, capacidad e imperfección de revisión.

## 6. Cómo leer costo y ahorro

```text
C_total = Ca × intervenciones + monto de fraudes aprobados
C_referencia = min(monto de todo fraude, N × Ca)
Ahorro_referencia = 1 − C_total/C_referencia
Mejora_fijo = 1 − C_total/C_fijo_LightGBM
Diferencia_vs_XGBoost = C_LightGBM − C_XGBoost, misma política y Ca
```

La última diferencia es descriptiva; negativa favorece a LightGBM. Los ahorros usan denominadores distintos y pueden ser negativos. Reportamos recall, legítimas intervenidas y recall ponderado por monto junto con costo y componentes.

El diagnóstico BMR raw/Platt mantiene la regla fija calibrada como referencia común. No es otra estrategia principal ni selecciona automáticamente calibrador. Una mejora de Brier o log-loss no garantiza menor costo en cada escenario: cambiar probabilidades cerca de pA=Ca puede alterar acciones sobre montos diferentes.

## 7. Evidencia y continuación

Las tablas `lightgbm_metricas_validation`, `lightgbm_calibracion_grupos`, `lightgbm_calibracion_montos`, `lightgbm_costos_validation` y `lightgbm_bmr_raw_platt` se exportan como CSV. Dos figuras muestran calibración/exposición y costos/compromiso de intervención. El registro `lightgbm_adopcion_config.json` enlaza fuentes, artefactos, selección, ventanas y salidas.

Los archivos por operación `lightgbm_validation_scores.parquet` y `lightgbm_validation_actions.parquet` quedan locales en data/interim; el paquete mínimo `lightgbm_seleccionado_platt.joblib` queda local en results/models. No reemplazan `validation_scores.parquet`, que continúa siendo XGBoost. No copiamos los otros tres modelos dentro del paquete seleccionado ni duplicamos los árboles.

La siguiente etapa analizará estabilidad por subperiodos, errores de alto monto y sensibilidad alrededor de la frontera económica. Antes de puntuar test congelaremos predictor, calibrador, reglas y justificación de los escenarios de costos. La adopción verificada no equivale por sí sola a aprobar un protocolo final ni a demostrar ahorros bancarios reales.

## 8. Hallazgos de la ejecución

Adoptamos los artefactos originales verificados del 03 y reproducimos las métricas LightGBM dentro de las tolerancias declaradas. Se preservaron manifiesto, scores XGBoost, tablas y modelos previos. Esta etapa generó 42.047 probabilidades y una fila de acciones por operación y escenario (252.282 filas, cada una con las tres políticas). No se ajustó ningún estimador ni se puntuó test.

### Probabilidades: mejora global con una limitación pendiente

| Probabilidad | AP | Brier | Log-loss | Probabilidad media | Recall a 0,5 |
|---|---:|---:|---:|---:|---:|
| raw | 0.50440 | 0.02435 | 0.09750 | 2.76% | 28.74% |
| platt | 0.50440 | 0.02406 | 0.09645 | 3.03% | 30.85% |

La prevalencia observada es 3.61%. Platt mejora Brier y log-loss, pero su probabilidad media de 3.03% sigue por debajo de esa prevalencia. AP permanece igual en esta corrida, consistente con preservar el orden de riesgo. Al cambiar los valores absolutos, sí cambia la clasificación con corte 0,5: las intervenciones pasan de 543 a 596.

Al ponderar probabilidades por monto, la exposición esperada con Platt suma 195,725.58, frente a 254,536.81 de monto fraudulento observado: una diferencia de 58,811.23 (23.11% del observado). Es un diagnóstico de subestimación en esta muestra, no una estimación de pérdidas bancarias reales.

La banda [500,1000) reúne 1,209 operaciones y 116 fraudes: frecuencia 9.59%, frente a probabilidad media 6.10%. Su monto fraudulento observado es 88,800.31, mientras la exposición esperada es 54,610.99. Esta banda merece atención en el análisis posterior. En [1000,∞), la probabilidad media 2.70% supera la frecuencia 2.14%; allí solo hay 13 fraudes entre 608 operaciones. Evitamos generalizar que todos los montos altos están subestimados o atribuir significación a tasas sin incertidumbre estimada.

### Costos de las tres políticas

| Ca hipotético | Regla fija | BMR | Árbol ponderado | Diferencia BMR vs XGBoost |
|---|---:|---:|---:|---:|
| 1 | 192,740.92 | 22,990.59 | 41,177.40 | -2,596.90 |
| 5 | 195,124.92 | 61,579.25 | 90,918.33 | -4,742.96 |
| 10 | 198,104.92 | 84,169.73 | 125,992.24 | -4,013.01 |
| 20 | 204,064.92 | 112,284.12 | 147,707.63 | -5,350.28 |
| 50 | 221,944.92 | 161,647.51 | 194,356.59 | -2,779.49 |
| 100 | 251,744.92 | 193,192.68 | 207,923.01 | -3,460.49 |

En este bloque BMR obtiene el menor costo de las tres políticas en 6 de los seis escenarios. Frente a BMR con XGBoost, LightGBM tiene menor costo en 6 escenarios. Se trata de una comparación descriptiva sobre validación de desarrollo ya consultada; no demuestra superioridad estadística ni garantiza el mismo resultado en test.

Con Ca=10, BMR interviene 3,186 operaciones: 843 fraudes y 2,343 legítimas. Su recall es 55.57% y la fracción de legítimas intervenidas, 5.78%. Captura 79.45% del monto fraudulento, que es distinto de la proporción de fraudes intervenidos.

El costo 84,169.73 se descompone en 31,860.00 por intervención y 52,309.72 por fraude aprobado. La mejora frente a la regla fija de LightGBM es 57.51%; el ahorro frente a la mejor referencia trivial es 66.93%. La referencia trivial en este escenario es aprobar todo, con 254,536.81. Ambos porcentajes usan denominadores diferentes.

El BMR de XGBoost cuesta 88,182.74; la diferencia de -4,013.01 favorece a LightGBM en este escenario. Los seis árboles conservaron sus costos del 03: no cambiaron sus estimadores, preprocesador ni filas, y no reciben las probabilidades del clasificador.

### Por qué calibrar no garantiza mejorar cada costo realizado

| Ca | BMR raw | BMR Platt | Platt − raw |
|---|---:|---:|---:|
| 1 | 22,793.77 | 22,990.59 | 196.82 |
| 5 | 63,499.84 | 61,579.25 | -1,920.59 |
| 10 | 86,759.50 | 84,169.73 | -2,589.77 |
| 20 | 114,689.94 | 112,284.12 | -2,405.83 |
| 50 | 166,296.73 | 161,647.51 | -4,649.22 |
| 100 | 192,708.94 | 193,192.68 | 483.74 |

Platt reduce el costo BMR en 4 de seis escenarios, pero lo aumenta en los otros 2. Esto ocurre a pesar de mejorar Brier y log-loss globales. Las probabilidades determinan acciones al cruzar pA=Ca; una modificación puede agregar revisiones legítimas y evitar fraudes de montos distintos. El riesgo esperado con probabilidades estimadas y el costo realizado con etiquetas observadas son magnitudes diferentes. BMR minimiza el primero bajo la matriz asumida, sin garantizar el mínimo realizado en cada muestra.

Conservamos Platt como calibrador adoptado del protocolo. Esta tabla no selecciona retrospectivamente raw ni otro Ca. El siguiente análisis debe revisar estabilidad temporal y errores de mayor exposición antes de congelar el protocolo final.


## 9. Continuidad de métricas y forma de las entradas

En la revisión del 4 de noviembre explicitamos las seis métricas del 03 para raw y Platt: AP, Brier, log-loss, recall, precision y bloqueo de legítimas. Las tres últimas utilizan 0,5. La nueva ejecución del 05 verifica la cadena del 04 ampliada y conserva exactamente sus métricas y costos previos. No incorpora ajustes nuevos de predictor, Platt ni árboles y mantiene test sin puntuar.

El dataframe de evaluación conserva las columnas originales y controles; make_features excluye ID, tiempo, etiqueta y split y conserva 432 entradas. El preprocesador produce una matriz float32 de 42.047 × 432 para LightGBM, sin el escalador de logística. El dataframe scores tiene 42.047 filas y siete columnas: ID, tiempo, monto original, etiqueta, probabilidades raw/Platt y acción fija. El Platt recibe una columna de logits. Las acciones se organizan por operación/Ca y los costos agregados por política/Ca.

La selección del 04 usa métricas raw en otros periodos internos. Aquí el contraste raw/Platt usa exactamente las mismas operaciones externas posteriores. Por ello una diferencia entre AP del 04 y del 05 no mide el efecto de la calibración. Las curvas de calibración y la evaluación económica complementan las métricas globales; ninguna diferencia de Brier demuestra por sí sola menor costo en cada escenario.
