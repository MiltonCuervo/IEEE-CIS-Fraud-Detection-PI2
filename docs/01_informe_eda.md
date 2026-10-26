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

### 2.3. Cómo leemos una variable de principio a fin

Para defender su tratamiento distinguimos tres niveles: **tipo semántico**, qué representa la variable; **dtype**, cómo llegó almacenada a pandas; y **rama de preparación**, qué operación ejecutó realmente nuestro pipeline. Un código entero puede ser categórico y una columna one-hot numérica sigue representando una categoría. Estos ejemplos proceden de los dos diccionarios versionados:

| Variable | Tipo semántico | Dtype de la unión | Tratamiento del predictor 06 | Columnas de matriz |
|---|---|---|---|---:|
| TransactionAmt | Numérica | float64 | Monto finito no negativo validado; valor predictivo float32 y original conservado para costos | 1 |
| ProductCD | Categórica | object | Token de ausencia, vocabulario de train y one-hot | 5 |
| card1 | Categórica | int64 | Se fuerza a categoría; agrupación de raras y one-hot | 884 |
| id_13 | Categórica según proveedor | float64 | Permanece numérica: mediana e indicador de ausencia | 2 |

Las anchuras describen el preprocesador concreto del reajuste; no son parámetros fijados para cualquier entrenamiento. En id_13, las dos columnas son valor numérico e indicador, no dos categorías. En card1, 884 columnas no significan 884 variables originales ni 884 personas. No cambiamos esa preparación al documentarla.

TransactionAmt tiene además una validación previa: `features._validated` rechaza montos ausentes, no finitos o negativos antes de construir atributos. Aunque el imputador numérico conserve una mediana registrada para esta columna, esa estadística no significa que la interfaz fuente permita rescatar una operación sin monto. La imputación de las otras variables numéricas no debe confundirse con ese contrato de entrada.

En los CSV puede seguirse una columna por su nombre: `origen` y `significado` explican de dónde viene; `tipo_semantico` y `dtype_union_eda` separan concepto y almacenamiento; `entra_modelo` identifica su participación; `tipo_pipeline_verificado`, `tratamiento` y `columnas_matriz` describen su transformación efectiva. `variables_padre` y `formula_o_procedencia` permiten rastrear la ingeniería. Los campos `ausencia_eda` y `cardinalidad_eda` describen el conjunto global, mientras `mediana_ajustada_train` corresponde al pasado que ajustó el modelo.

### 2.4. Qué se aprende después de esta exploración

La ingeniería y la preparación se ejecutan en las etapas posteriores. Para explicar su relación con el dato dejamos este resumen retrospectivo; el EDA no ajusta estas reglas con la población global:

| Componente | Regla implementada en 05/06 | Límite que debemos explicar |
|---|---|---|
| Numéricas | Infinitos a NaN, mediana aprendida en train, indicadores cuando hubo ausencia al ajustar y float32; el monto requiere validación finita no negativa previa | Un indicador no se añade automáticamente si una columna empieza a faltar por primera vez en el futuro; un monto fuente ausente se rechaza antes de imputar |
| Categóricas | Texto, `__MISSING__`, one-hot y categorías infrecuentes con mínimo 50 en train | Desconocidas dejan su bloque en ceros; no crean una categoría nueva ni equivalen automáticamente al grupo de raras |
| Monto y tiempo | Monto original más log1p; fases relativas de día y semana | Conservamos A original para costos; no asignamos fechas calendario no verificadas |
| Historia | Conteo, monto medio y recencia por proxies de tarjeta, dispositivo y dominio | Pasado estricto en train; estado congelado de train al transformar validación; sin etiquetas en los agregados |

Las doce discrepancias de tipo y las doce columnas con más de 90% de ausencia son recuentos de problemas diferentes. Ninguno autoriza una eliminación automática. Conservamos 431 originales permitidas, `has_identity` y quince derivados; no aplicamos PCA, selección por importancia, escalamiento ni recorte de montos para entrenar.

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

**Origen y mecanismo de ausencia son conceptos distintos.** Llamamos estructural a la falta causada por no encontrar una fila de identidad en la unión; una fila de identidad que sí existe también puede contener campos vacíos. No hemos identificado el mecanismo MCAR (ausencia independiente de los valores), MAR (dependencia de información observada) o MNAR (dependencia de valores no observados). El contraste de tasas no valida por sí solo uno de esos mecanismos ni garantiza utilidad predictiva fuera de muestra. Las formulaciones breves conservadas en salidas históricas deben leerse con este alcance.

### 3.5 Variables categóricas

Algunas categorías presentan diferencias considerables. `ProductCD=C` registra 11,69% de fraude frente a 2,04% en `W`; las tarjetas de crédito presentan 6,68% frente a 2,43% en débito; y las operaciones con dispositivo móvil alcanzan 10,17%.

Estas cifras ayudan a comprender el conjunto, pero fueron calculadas con todos los datos y no se utilizarán directamente como codificaciones. Cualquier transformación basada en la etiqueta deberá aprenderse únicamente con el bloque de entrenamiento.

No establecemos con estas tasas que una categoría cause fraude o que mejore el modelo en un periodo futuro. El soporte de cada grupo y su estabilidad temporal forman parte de la interpretación. La medida de cardinalidad cuenta niveles observados; no es una importancia de variable. Una alta cardinalidad justifica revisar memoria y representación, pero no es por sí sola una razón para eliminar una columna.

### 3.6. Ampliaciones de la revisión de la profesora

La revisión del 22 de octubre amplía la lectura de las celdas, el ratio legítimas/fraude, el resumen de ausencia por familias y las gráficas categóricas. Conservamos esos aportes como base del notebook e incorporamos después el diccionario completo. El ratio es aproximadamente 27,58 legítimas por fraude; se añade al resumen calculado junto con la fracción de monto fraudulento y su concentración superior.

La nueva figura `eda_tasas_fraude_categoricas.png` compara las categorías de mayor volumen de ProductCD, card4, card6, P_emaildomain y DeviceType, con una referencia de prevalencia global. Es una presentación de asociaciones, no codificación supervisada: el predictor actual usa one-hot aprendido con train y no tasas de fraude como atributos.

La figura `eda_señal_numericas.png` contrasta C1, D1, TransactionAmt y V258 entre clases. Son representantes ilustrativos, no una selección final ni un ranking de importancia. Las distribuciones se muestran sobre valores observados y se recortan al percentil 99 exclusivamente para visualizar; no truncamos los datos que alimentan modelos o costos. V258 tiene 77,91% de ausencia y se identifica como alta ausencia. Un histograma similar entre clases tampoco demuestra utilidad conjunta ni inutilidad: hace falta evaluación predictiva controlada.

Para conciliar explicación y código distinguimos exactitud insuficiente de macro-F1, que pondera ambas clases por igual; mantenemos dispositivos separados de la familia D y exportamos todas las figuras versionadas. Estas precisiones no cambian datos, etiquetas, atributos del predictor ni matriz de costos.

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
│   ├── eda_evolucion_temporal.png
│   ├── eda_tasas_fraude_categoricas.png
│   └── eda_señal_numericas.png
└── tables/
    ├── eda_cobertura_identidad_por_clase.csv
    ├── eda_distribucion_objetivo.csv
    ├── eda_diccionario_variables.csv
    ├── eda_fraude_segun_identidad.csv
    ├── eda_montos_por_clase.csv
    ├── eda_resumen_general.csv
    └── eda_resumen_temporal.csv
```
