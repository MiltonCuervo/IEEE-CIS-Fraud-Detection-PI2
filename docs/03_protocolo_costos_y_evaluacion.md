# Protocolo propuesto de costos y evaluación

**Estado:** propuesta metodológica v0.1; pendiente de aprobación de parámetros económicos por el equipo.  
**Fecha:** 2026-10-03.  
**Propósito:** convertir la pregunta del proyecto en una comparación reproducible de decisiones sin inventar costos que no aparecen en IEEE-CIS ni en los documentos actuales.

## 1. Base observada

- El objetivo documentado es comparar una decisión convencional, Bayes Minimum Risk (BMR) y un árbol sensible al costo, priorizando costo/ahorro fuera de muestra.
- El corte convencional de 0,5 no se optimiza, según el README.
- La probabilidad debe calibrarse antes de aplicar BMR.
- Todas las estrategias deben medirse en las mismas observaciones y con la misma matriz de costos.
- `TransactionAmt` se conserva en escala original como aproximación de pérdida potencial. El dataset no informa recuperación, contracargos ni pérdida bancaria neta.
- La partición temporal ya fue regenerada: train 380.815 filas, gaps 41.399, validation 84.093 y test 84.233. El manifiesto contiene 590.540 IDs únicos en bloques cronológicos.

Estos hechos están descritos en [README.md](../README.md), [informe EDA](01_informe_eda.md) e [informe de partición](02_particion_temporal.md). La partición de 7 días queda congelada para los siguientes experimentos.

## 2. Unidad de decisión y parámetros económicos

La unidad es una transacción `i`, con etiqueta real `y_i` (`1` fraude, `0` legítima), monto `A_i = TransactionAmt_i` y probabilidad calibrada `p_i`. Todos los términos monetarios (`λ A_i`, `B_i`, `R_i`, `C_block_fraud`) deben estar expresados en unidades compatibles. Los informes actuales llaman USD a los montos, pero falta confirmar la unidad en la fuente/metadatos del dataset; hasta entonces los resultados se rotulan como monto proxy en la escala declarada por los informes.

Parámetros que el equipo debe fijar antes de comparar estrategias:

| Símbolo | Significado | Estado |
|---|---|---|
| `λ` | Fracción de `A_i` que representa pérdida al aprobar fraude. `λ = 1` equivale a tratar todo el monto como exposición perdida. | Supuesto de modelado; falta evidencia de recuperación/pérdida neta. |
| `B_i` | Costo de bloquear una transacción legítima: margen perdido, fricción, abandono, soporte u otro concepto acordado. Puede ser constante `B` si el equipo no dispone de estimación por transacción. | Dato requerido; no equivale automáticamente a `A_i`. |
| `C_block_fraud` | Costo residual al bloquear una transacción fraudulenta. | Puede aproximarse a cero si se justifica prevención completa y costo residual despreciable; acuerdo requerido. |
| `R_i` | Costo administrativo de revisar la transacción, incluyendo los componentes que el equipo decida contabilizar. `R` si es constante. | Dato requerido; motiva el análisis de sensibilidad. |
| `s_R` | Sensibilidad de la revisión: probabilidad de que una revisión detecte un fraude. | Dato requerido para modelar revisión imperfecta. |
| `q_R` | Especificidad de la revisión: probabilidad de que una revisión deje pasar como legítima una operación legítima. | Dato requerido para modelar revisión imperfecta. |
| `K_R` | Capacidad máxima de revisiones en el periodo, si existe. | Dato requerido solo si la operación tiene cupo limitado. |

No se asignan valores monetarios a `B`, `R`, `s_R`, `q_R` o `K_R` porque las fuentes revisadas no los proporcionan. Si no hay estimaciones de `s_R` y `q_R`, la revisión perfecta podrá usarse únicamente como **escenario idealizado** (`s_R = q_R = 1`), nunca como hecho observado.

## 3. Matriz de costo propuesta

Se proponen tres acciones: aprobar (`A`), bloquear (`B`) y enviar a revisión (`R`). La siguiente matriz distingue costos conocidos como aproximaciones de los que dependen de información operativa:

| Estado real | Aprobar | Bloquear | Revisar |
|---|---:|---:|---:|
| Fraude (`y=1`) | `λ A_i` | `C_block_fraud` | `R_i + (1-s_R) λ A_i + s_R C_block_fraud` |
| Legítima (`y=0`) | `0` | `B_i` | `R_i + (1-q_R) B_i` |

Interpretación propuesta:

- `λ A_i` aproxima la exposición no evitada cuando se aprueba fraude.
- `C_block_fraud` representa cualquier costo residual al bloquear fraude; se puede fijar en cero solo si se acuerda que bloquear evita completamente esa exposición y no añade costo relevante.
- `B_i` representa el perjuicio de bloquear una transacción legítima; necesita definición del equipo.
- En revisión, `R_i` se incurre por cada caso revisado. Los términos residuales representan errores de revisión. Se asume que una revisión que detecta fraude lleva a bloquearlo y que un caso legítimo aceptado no añade costo. Si el flujo operativo difiere, se ajusta la matriz antes del análisis.

En el escenario idealizado de revisión perfecta y bloqueo de fraude sin costo (`s_R=q_R=1`, `C_block_fraud=0`), revisar cuesta `R_i` tanto para fraude como para legítima. Este escenario simplifica el análisis, pero puede sobrestimar el valor de revisión si en la práctica los analistas se equivocan.

## 4. Riesgo esperado para BMR

Para una probabilidad calibrada `p_i`, el costo esperado de cada acción es:

```text
E[C_i | aprobar] = p_i λ A_i
E[C_i | bloquear] = (1-p_i) B_i + p_i C_block_fraud
E[C_i | revisar] = R_i
                       + p_i [(1-s_R) λ A_i + s_R C_block_fraud]
                       + (1-p_i) [(1-q_R) B_i]
```

BMR selecciona por transacción la acción con menor costo esperado. En empates exactos se predefine una regla determinista: preferir no revisar; entre aprobar y bloquear, aprobar. Si la revisión no tiene capacidad `K_R`, la elección es independiente por fila. Si sí tiene capacidad, se necesita definir una regla de asignación de cupos y medir también qué transacciones quedan fuera de revisión.

Como comprobación para el caso binario sin revisión y `C_block_fraud=0`, el umbral económico de bloqueo depende del monto:

```text
p_i >= B_i / (B_i + λ A_i)
```

Por eso un único umbral 0,5 y una política sensible al costo pueden tomar decisiones distintas incluso con la misma probabilidad.

## 5. Estrategias que se compararán

1. **Convencional:** bloquear si `p_i >= 0,5`; aprobar en caso contrario. No se optimiza el corte con validation.
2. **BMR:** usar probabilidades calibradas y elegir entre aprobar, bloquear y revisar por costo esperado mínimo. Si faltan parámetros de revisión, reportar primero la variante binaria aprobar/bloquear solo cuando `B_i` y `C_block_fraud` estén acordados; aplazar la acción de revisar hasta parametrizarla.
3. **Árbol sensible al costo:** entrenar un árbol de clasificación binario con `sample_weight` derivado de la matriz aprobada. Candidato inicial, si `B_i=B` y `C_block_fraud=0`: `w_i = λ A_i` para fraude y `w_i = B` para legítimas; normalizar pesos por una constante común para estabilidad. Evaluar sus decisiones aprobar/bloquear con la misma matriz que las demás estrategias.

El peso por transacción es una aproximación para errores binarios: ponderar impurezas del árbol no equivale exactamente a optimizar la matriz completa, no representa una acción de revisión y puede alterar la escala de sus scores. El README ya exige declarar esta limitación. Si se requiere calibrar el score del árbol, se hará fuera de muestra y con la prevalencia natural. No se debe volver a ponderar el costo en la evaluación: la matriz se aplica una sola vez a las acciones finales.

La comparación principal del árbol usa acciones binarias aprobar/bloquear. Añadir revisión como tercera acción al árbol queda como extensión, posterior a definir y documentar el mecanismo de asignación y cualquier límite de capacidad. Las tres estrategias principales se evalúan sobre las mismas filas de test y con la misma matriz.

## 6. Protocolo temporal y prevención de leakage

1. Mantener train, gaps, validation y test del manifiesto; los gaps no se entrenan ni se evalúan.
2. Ajustar imputación, selección de columnas, codificación, atributos agregados y modelo únicamente con train. Todo agregado por entidad debe respetar el pasado respecto a cada transacción.
3. Elegir hiperparámetros del predictor mediante validación cruzada temporal dentro de train o fijarlos antes de mirar validation. No hacer tuning de hiperparámetros en validation.
4. Para el protocolo inicial, ajustar preprocesador y predictor con train y mantener ese predictor fijo. Dividir validation cronológicamente en una ventana temprana para ajustar el calibrador y otra posterior para comparar configuraciones de calibración/política. El modelo que generó los scores de calibración debe ser el mismo que produce scores posteriores de validation y test.
5. Si más adelante se decide reajustar el predictor con train+validation, no reutilizar sin más un calibrador aprendido con scores de otro modelo. En ese caso se necesita calibración temporal cross-fit compatible con el procedimiento de reajuste, documentada antes de ejecutarse.
6. Fijar matriz, valores/rango de sensibilidad, predictor, calibrador, baseline, regla BMR, desempate y métricas antes de consultar test.
7. Evaluar test una sola vez para los tres enfoques y los escenarios de costos predeclarados. No ajustar hiperparámetros, calibración, umbrales, costos o alcance de revisión a partir de los resultados de test.

El tamaño de la subdivisión de validation se debe decidir y registrar en el notebook de modelado antes de ejecutarlo; el informe de partición identifica esta decisión como pendiente.

## 7. Métricas y reporte

**Primaria**

- Costo total realizado: `C_total = Σ_i c(y_i, a_i; A_i, λ, B_i, C_block_fraud, R_i, s_R, q_R)`, sumando el costo de la matriz para la etiqueta real y acción tomada en cada transacción.
- Costo promedio por transacción.
- Ahorro absoluto frente a la estrategia convencional: `C_convencional - C_estrategia`.
- Ahorro relativo: `(C_convencional - C_estrategia) / C_convencional`, solo si el costo de referencia es positivo.

El componente asociado a `TransactionAmt` debe titularse **exposición/costo proxy**, no pérdida bancaria real. Separar sus componentes del costo administrativo y de bloqueo para hacer visibles los supuestos.

**Diagnóstico predictivo y operativo**

- AUC-PR (average precision) y recall de fraude, exigidos por la documentación.
- Tasa y conteo de legítimas bloqueadas.
- Fraudes bloqueados y fraude enviado a revisión, reportados por separado. Un caso enviado a revisión no cuenta automáticamente como detectado/bloqueado.
- Recall operativo observado si se dispone de resultado humano; si solo se dispone de `s_R`, recall esperado = `(fraudes bloqueados + s_R × fraudes revisados) / fraudes totales`.
- Recall ponderado por monto proxy: `(monto de fraudes bloqueados + s_R × monto de fraudes revisados) / monto proxy fraudulento total`; declarar que es esperado si no hay resultado humano fila a fila.
- Tasa, conteo y costo de revisión; fraudes y legítimas enviados a revisión.
- Brier score o log-loss y curva de calibración antes de aplicar BMR.
- Resultados por escenario de `R`, y por `B` si el equipo acuerda sensibilidad para ese parámetro.

Accuracy puede aparecer como dato secundario, nunca como criterio principal debido a la prevalencia de fraude de 3,499%.

## 8. Decisiones requeridas para cerrar la matriz

El protocolo queda listo para implementar cuando el equipo responda y registre:

1. ¿Se acepta `λ=1` como proxy base, o hay una tasa de recuperación/pérdida distinta?
2. ¿Qué representa `B_i` para bloquear una transacción legítima? ¿Existe una estimación, margen o rango de sensibilidad sustentado?
3. ¿Qué rubros incluye `R_i` y cuáles serán los valores o rango de escenarios de revisión?
4. ¿Hay datos para `s_R` y `q_R`? Si no, ¿se reportará revisión perfecta como límite ideal y se aplazará la estimación realista?
5. ¿Existe capacidad máxima `K_R` por día/periodo?
6. ¿Se acepta mantener fijo el baseline convencional en 0,5?
7. ¿Se acepta dividir validation en orden temporal para calibración y selección/evaluación de políticas?
8. ¿Se confirma la unidad/moneda de `TransactionAmt` con metadatos/fuente? Mientras tanto, usar la denominación neutral “monto proxy” y anotar que los informes actuales lo etiquetan USD.

Hasta resolver estas decisiones se pueden desarrollar componentes con parámetros configurables, pero no reportar una cifra de ahorro como resultado del proyecto.

## 9. Notebook de implementación

El primer notebook de modelado se implementa como pipeline parametrizado en [`notebooks/03_modelado_y_decisiones_costos.ipynb`](../notebooks/03_modelado_y_decisiones_costos.ipynb). Puede producir métricas predictivas en la ventana posterior de validation con costos pendientes; mantiene cerrados el bloque económico y la evaluación de test hasta que se completen y aprueben sus parámetros.

Después de aprobar los costos, el mismo notebook puede comparar la política fija y BMR, incorporar el árbol con `sample_weight` y ejecutar test una sola vez tras congelar el protocolo.

**Criterio de éxito:** el mismo manifiesto y filas alimentan todas las políticas; la calibración no usa test; cada costo es rastreable a una fuente o supuesto aprobado; y la tabla final muestra costo, ahorro, AUC-PR, recall, legítimas bloqueadas y revisión para cada estrategia.
