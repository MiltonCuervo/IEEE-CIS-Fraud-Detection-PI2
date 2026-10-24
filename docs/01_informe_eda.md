# Informe de exploración de datos

## 1. Propósito de la etapa

Esta etapa tuvo como objetivo conocer la estructura del conjunto IEEE-CIS Fraud Detection antes de definir transformaciones o entrenar modelos. Nos concentramos en los aspectos que afectan directamente la pregunta del proyecto: disponibilidad de la etiqueta, desbalance, distribución del monto, orden temporal, cobertura de la información de identidad y ausencia de datos.

El análisis reproducible se encuentra en [`notebooks/01_exploracion_datos.ipynb`](../notebooks/01_exploracion_datos.ipynb). Este documento sintetiza los resultados y las decisiones que se derivan de ellos; no reemplaza el notebook ni repite todo su código.

## 2. Datos utilizados

Trabajamos con los archivos etiquetados de la competencia:

- `train_transaction.csv`: 590.540 transacciones y 394 columnas.
- `train_identity.csv`: 144.233 registros y 41 columnas, incluida la clave compartida; aporta 40 columnas no clave.

Realizamos un `left join` mediante `TransactionID`. La tabla resultante conserva las 590.540 transacciones y contiene 434 columnas. La cobertura de identidad es 24,42%, por lo que una unión interna eliminaría la mayoría de las observaciones etiquetadas y cambiaría la población analizada.

### 2.1. Inventario completo y significado de los tipos

Exportamos `results/tables/eda_diccionario_variables.csv` con **434 filas, una por columna original**. Conserva tabla de origen, familia, significado verificable, dtype leído, tipo semántico, rol, cardinalidad, ausencia y relación con ingeniería. No sustituimos ese inventario por ejemplos de cinco columnas. La ausencia/cardinalidad globales describen el EDA; no se usan como medianas, vocabulario ni selección para entrenar.

| Familia original | Cantidad | Tipo semántico o rol |
|---|---:|---|
| TransactionID, TransactionDT, isFraud | 3 | Identificador, tiempo relativo y objetivo; no entradas directas |
| TransactionAmt | 1 | Numérica y monto para evaluación económica |
| ProductCD | 1 | Categórica |
| card1–card6 | 6 | Códigos categóricos, aunque algunos se almacenan como números |
| addr1–addr2 | 2 | Categóricas |
| dist1–dist2 | 2 | Numéricas |
| P_emaildomain, R_emaildomain | 2 | Categóricas |
| C1–C14 | 14 | Descriptores numéricos anonimizados |
| D1–D15 | 15 | Descriptores numéricos anonimizados |
| M1–M9 | 9 | Categóricas |
| V1–V339 | 339 | Descriptores numéricos del proveedor |
| id_01–id_38 | 38 | id_01–id_11 numéricas; id_12–id_38 declaradas categóricas por el proveedor |
| DeviceType, DeviceInfo | 2 | Categóricas |

La [descripción oficial](https://www.kaggle.com/competitions/ieee-fraud-detection/data) aporta la clasificación categórica y el carácter relativo de TransactionDT. No conocemos la interpretación individual de todos los códigos anonimizados: el diccionario lo señala expresamente, en vez de atribuir significado a cada C, D, M, V o id. Las variables V ya vienen en el archivo y no son ingeniería creada por nosotros.

### 2.2. Qué conservamos y qué no entra directamente

Conservamos **431 columnas originales como predictores**, sin selección automática por correlación, importancia, cardinalidad o ausencia. TransactionID se conserva para unir/alinear, isFraud como y y TransactionDT para ordenar, partir y construir derivados, pero las tres quedan fuera de X. Split, creado después, es control temporal y tampoco entra. Que no sean entradas directas no implica borrarlas de los datos fuente.

La siguiente descripción es una reconstrucción documental de las etapas 05/06, no una afirmación de que el EDA original ajustó el preprocesador. Con has_identity tenemos 432 atributos base; con quince derivados posteriores llegamos a 447. El artefacto del 06 distribuye esas entradas en 37 categóricas y 410 numéricas antes de la matriz codificada de 2.642 columnas. El [diccionario y flujo hasta el modelo](diccionario_variables_y_flujo_modelo.md) enumera tratamientos, fórmulas y contratos de entrada/salida; no confunde las 434 originales con los 447 atributos ni con las 2.642 columnas codificadas.

Hay una discrepancia pendiente: id_13, id_14, id_17, id_18, id_19, id_20, id_21, id_22, id_24, id_25, id_26 e id_32 quedaron numéricas en nuestro código pese a la clasificación categórica del proveedor. Los árboles pueden usar un orden no justificado entre códigos. Lo documentamos por columna sin cambiar retrospectivamente la preparación, los modelos ni sus resultados.

## 3. Resultados principales

### 3.1 Variable objetivo

Encontramos 20.663 transacciones fraudulentas, equivalentes al 3,499% del conjunto. Las 569.877 transacciones restantes son legítimas. Este desbalance hace que la exactitud sea insuficiente para evaluar el proyecto: una estrategia que aprobara todo acertaría la clase mayoritaria, pero no detectaría ningún fraude.

![Distribución de la variable objetivo](../results/figures/eda_distribucion_objetivo.png)

La evaluación predictiva deberá incluir AUC-PR y recall. Sin embargo, la comparación principal utilizará costo total y ahorro, porque el objetivo del proyecto es económico.

### 3.2 Monto de las transacciones

El monto mediano observado fue 68,50 unidades de `TransactionAmt` en transacciones legítimas y 75,00 en fraudulentas. Los promedios fueron 134,51 y 149,24 unidades, respectivamente. La distancia entre promedio y mediana muestra que la distribución es asimétrica y contiene montos altos. **La unidad/moneda no está confirmada por la fuente consultada**, por lo que no se interpreta como USD.

![Distribución de los montos](../results/figures/eda_distribucion_montos.png)

Las operaciones fraudulentas representan 3,87% del monto total, una proporción ligeramente superior a su participación por cantidad. Además, el 1% de los fraudes con mayor monto concentra 10,63% del monto fraudulento. Por lo tanto, reportaremos recall y costo de manera conjunta y revisaremos la sensibilidad a montos extremos sin eliminarlos automáticamente.

### 3.3 Comportamiento temporal

`TransactionDT` cubre aproximadamente 182 días y está ordenado de forma creciente. Al dividir el periodo en diez intervalos exploratorios, la tasa de fraude varió entre 2,68% y 4,44%. También se observaron cambios en el volumen de operaciones y en el monto mediano.

![Evolución temporal](../results/figures/eda_evolucion_temporal.png)

Estos resultados no demuestran por sí solos deriva de concepto, pero muestran que la prevalencia no permanece constante. Una partición aleatoria mezclaría periodos distintos y no representaría el escenario de entrenar con el pasado para decidir sobre transacciones futuras. En consecuencia, la evaluación principal será temporal.

### 3.4 Ausencia e información de identidad

De las 434 columnas unidas, 414 contienen al menos un valor ausente y 12 superan 90% de ausencia. La falta de información no tiene un único origen: parte ya existe en `train_transaction` y `train_identity`, y otra parte aparece porque la tabla de identidad solo cubre algunas operaciones.

La tasa de fraude fue 7,85% en transacciones con registro de identidad y 2,09% en aquellas sin identidad. A su vez, 54,77% de los fraudes tiene identidad, frente a 23,32% de las operaciones legítimas. Interpretamos este resultado como una asociación descriptiva, no como evidencia causal. Conservaremos un indicador de disponibilidad y evaluaremos su estabilidad temporal.

### 3.5 Variables categóricas

Algunas categorías presentan diferencias considerables. `ProductCD=C` registra 11,69% de fraude frente a 2,04% en `W`; las tarjetas de crédito presentan 6,68% frente a 2,43% en débito; y las operaciones con dispositivo móvil alcanzan 10,17%.

Estas cifras ayudan a comprender el conjunto, pero fueron calculadas con todos los datos y no se utilizarán directamente como codificaciones. Cualquier transformación basada en la etiqueta deberá aprenderse únicamente con el bloque de entrenamiento.

## 4. Decisiones derivadas del EDA

Con base en la evidencia anterior acordamos:

1. conservar el `left join` para mantener todas las transacciones etiquetadas;
2. no usar accuracy como criterio principal debido al desbalance;
3. conservar `TransactionAmt` en su escala original para la matriz de costos y declarar que representa una aproximación de la pérdida potencial;
4. construir train, validación y test en orden temporal;
5. ajustar imputación, eliminación de columnas y codificación después de separar los bloques;
6. diferenciar ausencia estructural de identidad y ausencia interna de las variables;
7. evaluar los montos extremos mediante sensibilidad, no eliminarlos únicamente por su magnitud.

## 5. Limitaciones de esta etapa

- `TransactionAmt` no es una pérdida bancaria neta confirmada: el dataset no informa recuperaciones ni contracargos. La unidad/moneda también debe confirmarse antes de asignarle interpretación financiera.
- `TransactionDT` no tiene un origen calendario público; las horas y días derivados son relativos.
- Las asociaciones categóricas no implican causalidad.
- El EDA utiliza el conjunto etiquetado completo para descripción. Ninguna estadística global se reutilizará para ajustar transformaciones o seleccionar modelos.
- Esta etapa todavía no define imputadores, codificadores, atributos finales ni hiperparámetros.
- El diccionario actualizado distingue las descripciones originales de la preparación que se ejecutó después. Su columna tipo_regla_vigente describe esa implementación, no una recomendación semántica ni un ajuste aprendido del conjunto global.

## 6. Resultado de la etapa y continuación

La exploración confirma que el problema combina desbalance, ausencia extensa, cobertura parcial de identidad y variación temporal. También muestra por qué contar aciertos no es suficiente: el monto individual cambia la consecuencia económica de cada error.

La siguiente etapa será definir y congelar entrenamiento, validación, intervalos de separación y prueba. Esa división debe realizarse antes de construir transformaciones o atributos agregados.

## 7. Archivos reproducibles

Las evidencias seleccionadas se regeneran al ejecutar el notebook:

```text
results/
├── figures/
│   ├── eda_distribucion_objetivo.png
│   ├── eda_distribucion_montos.png
│   └── eda_evolucion_temporal.png
└── tables/
    ├── eda_cobertura_identidad_por_clase.csv
    ├── eda_distribucion_objetivo.csv
    ├── eda_diccionario_variables.csv
    ├── eda_fraude_segun_identidad.csv
    ├── eda_montos_por_clase.csv
    ├── eda_resumen_general.csv
    └── eda_resumen_temporal.csv
```
