# Etapa 07 — Protocolo final de evaluación temporal

> Alcance de etapa: este informe conserva la metodología y resultados al cerrar esta fase. La secuencia completa y el test ya evaluado se consolidan en el [informe 08](08_evaluacion_temporal_final.md). Los pasos futuros aquí descritos se leen en ese orden histórico.

## 1. Propósito y resultado

Congelamos técnicamente el experimento académico después de seleccionar LightGBM en el 04, adoptarlo en el 05 y diagnosticar estabilidad en el 06. El [notebook 07](../notebooks/07_protocolo_final.ipynb) fue ejecutado: produjo ocho comprobaciones satisfactorias, seis decisiones explícitas y un protocolo con estado `frozen_for_academic_evaluation`.

La congelación fija cómo evaluaremos. No genera probabilidades ni utiliza etiquetas test. Tampoco acredita aval docente o costos bancarios observados. La evaluación temporal final corresponde a la etapa posterior.

## 2. Decisiones fijadas

| Componente | Decisión | Evidencia y alcance |
|---|---|---|
| Preparación | 432 entradas: 37 categóricas ordinales y 395 numéricas imputadas | Medianas/vocabularios de train; monto económico original float64 |
| Predictor | LightGBM referencia: 250 estimadores, learning_rate 0,05, 63 hojas y min_child_samples 50 | Mayor AP media entre ocho candidatos; no óptimo global demostrado |
| Ajuste | Modelo completo del 03, verificado en 05 | Sin reajuste con validación o test |
| Calibrador | Platt propio sobre logit, clipping 10⁻⁶ | Ajustado con 42.046 operaciones tempranas |
| Fija | Intervenir si p calibrada ≥ 0,5 | Corte sin optimización |
| BMR | Intervenir si p calibrada × A > Ca | Aprobar en empate; sin dividir por A |
| Árbol | Un árbol del 03 por Ca; profundidad 12, mínimo 250 por hoja, semilla 42 | Objetivo 1[yA>Ca], peso abs(yA−Ca); no recibe p de LightGBM |
| Escenarios | Ca = 1, 5, 10, 20, 50, 100 | Sensibilidad hipotética; Ca=10 ilustrativo |

El Platt conservado tiene a=1,0442286259 y b=0,2746296646; transforma mediante `sigmoid(a × logit(clip(p_raw)) + b)`. El protocolo contiene los valores completos. No recodificamos los doce id documentados ni incorporamos categorías nativas, ingeniería o remuestreo. Cambiar esas decisiones produciría otra configuración.

## 3. Matriz, ahorro y fundamentación de Ca

| Estado observado | Aprobar | Intervenir |
|---|---:|---:|
| Fraude | A, monto original | Ca |
| Legítima | 0 | Ca |

Conservamos intervención eficaz, sin modelar capacidad limitada, recuperación, revisión imperfecta o perjuicio comercial adicional. El costo es `Ca × intervenciones + monto de fraudes aprobados`. La referencia trivial es `min(monto total fraudulento, N × Ca)`. El ahorro trivial y la mejora frente al fijo se reportan por separado; no promediamos ahorros entre escenarios ni elegimos Ca por el resultado test.

[Bahnsen et al. (2013), sección V.E y tabla VI](https://albahnsen.github.io/files/Cost%20Sensitive%20Credit%20Card%20Fraud%20Detection%20using%20Bayes%20Minimum%20Risk%20-%20Publish.pdf) estudian 1, 2,50 y 5 euros con otra procesadora. Los extremos inspiran nuestros escenarios 1 y 5; el antecedente 2,50 no se añade retrospectivamente a la cuadrícula desarrollada.

[Bahnsen et al. (2014), sección 4](https://albahnsen.github.io/files/%20Improving%20Credit%20Card%20Fraud%20Detection%20by%20using%20Calibrated%20Probabilities%20-%20Publish.pdf) utilizan 10 euros estimados con el equipo de riesgo de otra empresa. Ese número inspira el escenario 10. Los valores 20, 50 y 100 son sensibilidad ampliada retenida de desarrollo.

Estos antecedentes no estiman costos IEEE-CIS. No convertimos euros ni asignamos una moneda no verificada: Ca debe interpretarse hipotéticamente en la misma unidad matemática que TransactionAmt. Citamos un fundamento para el diseño, sin afirmar que reproducimos exactamente los datos o algoritmos de esos trabajos.

## 4. Población y plan de resultados

Congelamos las 84.233 operaciones asignadas a test. El 07 lee del manifiesto solo TransactionID, TransactionDT y split: verifica unicidad, población y orden temporal y guarda una huella de ID/tiempo. No lee isFraud de test ni llama predict/predict_proba. El modelo permanece aprendido con 380.815 operaciones de train; Platt, con validación temprana. No reajustamos utilizando las 42.047 operaciones posteriores de desarrollo.

El plan fijado comprende:

- Costos y componentes por cada Ca y cada una de las tres políticas, costo por operación, ahorro trivial y mejora frente al fijo.
- AP, Brier, log-loss, recall, precision y legítimas marcadas a 0,5 para raw/Platt. AP significa `average_precision_score`, no área trapezoidal de precision–recall.
- TP/FP/FN/TN por política, recall, número de intervenciones, fracción de legítimas intervenidas y recall por monto.
- Diagnósticos descriptivos en tres subperiodos de igual duración y las seis bandas de monto ya definidas: [0,50), [50,100), [100,250), [250,500), [500,1000) y [1000,∞).

Usaremos las mismas operaciones, montos y matriz para todas las políticas. Los resultados serán observados y descriptivos, sin afirmación de significación estadística. No cambiaremos parámetros, calibrador, costos o atributos después de consultar test. Los diagnósticos no servirán para elegir retrospectivamente una variante favorable sobre ese mismo periodo.

## 5. Integridad y ejecución posterior

La lógica está en `src/fraud_cost/final_protocol.py`. La congelación vincula fuentes, datos, manifiesto, modelos, registros y evidencias de 03–06. Antes de evaluar:

```python
from fraud_cost.final_protocol import verify_before_test

protocol = verify_before_test(PROJECT_ROOT)
```

La comprobación se detiene si cambia una dependencia o ya existe `results/tables/evaluacion_test_config.json`. El futuro evaluador deberá crear ese registro **de forma exclusiva antes de puntuar**, guardar la huella del protocolo y conservar el resultado o fallo. La comprobación de existencia por sí sola no es un bloqueo de concurrencia. Una interrupción exige auditar y recuperar esa misma corrida antes de repetirla; este notebook todavía no implementa el evaluador.

Reejecutar el 07 verifica la congelación sin sobrescribirla. Una revisión necesaria antes de test exige una nueva versión explícita del protocolo. Los flags `final_protocol_frozen=False` de 05/06 son estados históricos de esas etapas; el registro posterior de 07 acredita la congelación actual. Tampoco inventamos un aval docente a partir de ese estado técnico.

## 6. Reproducción local previa

Faltaban los archivos LightGBM por operación y el recibo de reconstrucción del 03 enlazaba el 04 anterior a su ampliación de métricas. Revalidamos el vínculo: la corrida 03 no cambió, los artefactos/evidencia estaban íntegros y la configuración seleccionada coincidía. Se conservó el recibo anterior sin ajustar estimadores.

`scripts/revalidate_reference_recovery.py` implementa esa comprobación; rechaza cambios de corrida 03 o parámetros que exijan otro ajuste. `scripts/reproduce_validation.py` ejecutó 05/06 con respaldo: sus doce tablas coincidieron con las anteriores dentro de atol=rtol=10⁻⁸, fijados previamente. Ya están disponibles los scores, acciones y paquete seleccionado. Los registros distinguen esta reconstrucción de los originales de otra máquina.

Después de congelar no debemos regenerar etapas previas sobre esas mismas rutas para continuar una evaluación pendiente: cambiarían sus huellas. Una reproducción nueva requiere un área y una versión de experimento propias. Otro computador debe disponer de los mismos artefactos verificados para usar esta congelación; un pull no descarga binarios ignorados. Las evidencias anteriores permanecen recuperables en Git y en respaldos locales.

## 7. Evidencia y límites

- `results/tables/protocolo_final.json`: decisiones y dependencias congeladas.
- `results/tables/protocolo_final_config.json`: estado completo y huellas de salidas.
- `results/tables/protocolo_final_comprobaciones.csv`: ocho comprobaciones satisfactorias.
- `results/tables/protocolo_final_decisiones.csv`: seis decisiones y fundamentos.

Conservamos la subestimación en [500,1000), la variación temporal y los supuestos simplificados de costos. Validación posterior ya influyó en el desarrollo y el EDA/selección histórica de gaps consultó etiquetas agregadas de test. Por eso no describiremos la evaluación final como estrictamente ciega ni como ahorro bancario real o superioridad universal.

La siguiente etapa es construir y ejecutar la evaluación temporal final bajo este protocolo, registrando el inicio de forma exclusiva y manteniendo trazabilidad de los resultados.


## 8. Consulta de esta evidencia después de la evaluación

La evaluación se completó en el 08. El ejemplo `verify_before_test` anterior
documenta el control previo a puntuar; no es el comando de consulta de una copia
ya evaluada y debe detenerse si encuentra el registro final existente. Para
consultarla se ejecuta `scripts/audit_project.py`, como explica el README.

El protocolo recibido conserva sus dependencias reconstruidas en otro equipo.
El 08 no sustituyó sus hashes: verificó equivalencia de las doce tablas y de las
decisiones y congeló una versión local explícita antes de evaluar. La ruta y la
huella de esa versión activa están en `protocolo_ejecucion_local_v2.json`.
Verificar el protocolo recibido contra binarios originales diferentes puede
fallar por esa procedencia; no demuestra que debamos reemplazarlo o repetir test.
