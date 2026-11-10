# Artículo final: borrador revisado

Proyecto Integrador II, Ingeniería de Sistemas, Universidad de Antioquia, 2026-2.
Equipo: Milton Alejandro Cuervo Ramírez, Oswald David Gutiérrez Cortina y Diego Alejandro Rendón Suaza.
Profesora acompañante: María Bernarda Salazar Sánchez.

## Estado y entrega

Versión revisada el 10 de noviembre de 2026 para lectura del equipo y de la profesora; no acredita aval docente. El PDF conserva seis páginas, incluidas las referencias, con la clase oficial IEEEtran en modo conference, tamaño carta y dos columnas. La entrega del curso exige aval explícito y recepción a más tardar el 16 de noviembre de 2026.

## Archivos necesarios

- `informe_final.pdf`: artículo revisado visualmente, de seis páginas.
- `informe_final.tex`: fuente editable, con siete tablas, dos figuras y trece referencias.
- `IEEEtran.cls`: clase oficial sin modificaciones, con licencia en la cabecera.
- `compromiso_operativo.pdf`: Figura 1, compromiso entre intervención de legítimas y captura monetaria.
- `calibracion_por_monto.pdf`: Figura 2, boxplot de probabilidades Platt por banda de monto.
- `resumen_boxplot.csv`: población, fraudes, cuartiles, bigotes, media, frecuencia observada, extremos y número de puntos externos omitidos para las seis bandas; probabilidades expresadas en porcentaje.

Para editar en Overleaf, subir la fuente .tex, la clase y ambas figuras y compilar `informe_final.tex` con XeLaTeX. Localmente se puede ejecutar `xelatex informe_final.tex` dos veces, o `tectonic informe_final.tex`. La versión revisada se compiló con Tectonic 0.17.0. Después de editar debe comprobarse nuevamente el máximo de seis páginas; .aux, .log y .out son archivos temporales.

## Criterios de la revisión

La redacción se normalizó en forma impersonal y en tercera persona, con párrafos que relacionan decisiones, evidencia e interpretación. Los antecedentes y los objetivos general y específicos quedaron en la introducción. La metodología inicia con el dataset y presenta después partición, matriz, selección y calibración. Las configuraciones se muestran en una tabla; la comparación predictiva reúne AP, Brier, log-loss, recall, precisión y FPR en otra tabla común. Las alternativas son ocho configuraciones acopladas, no una búsqueda cartesiana exhaustiva.

Resultados y discusión analiza el intercambio entre revisión y fraude aprobado, las diferencias entre discriminación y calibración y las restricciones que afectan la transferencia temporal. Las conclusiones se articulan con los cuatro objetivos de la propuesta y no repiten la tabla de resultados. La sección de limitaciones conserva el alcance económico y metodológico. Las menciones usan «Figura» y «Tabla» sin abreviar; los títulos de las figuras coinciden con las referencias en el texto.

## USD y escenarios administrativos

La [aclaración de Vesta en el foro oficial de IEEE-CIS](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/100304) indica que TransactionAmt fue convertido a USD. Esta fuente se incorporó como referencia del artículo al revisar la unidad monetaria el 10 de noviembre. Por eso los montos y costos se presentan en dólares estadounidenses, sin aplicar una conversión nueva a los valores guardados.

Ca = 1, 5, 10, 20, 50 y 100 USD sigue siendo una cuadrícula hipotética; no constituye una estimación de gastos administrativos de una entidad. Los antecedentes en euros orientan los escenarios, sin trasladar moneda ni costos reales. Las referencias históricas a «moneda no verificada» documentan la información disponible al cerrar esas etapas y quedan superadas por esta comprobación. Los registros congelados no se reescriben.

## Evidencia y Figura 2

Las tablas proceden de las exportaciones existentes de selección interna, métricas de validación y prueba y costos finales. Sus valores y poblaciones no se modificaron. El boxplot se construyó desde las 84.233 probabilidades Platt ya guardadas en `data/interim/lightgbm_test_scores.parquet`, sin cargar modelos ni generar predicciones. Los conteos y medias se contrastaron con `test_calibracion_montos.csv` con tolerancia de 10⁻⁸.

Cada caja representa los cuartiles 1 y 3 de las probabilidades individuales de una banda; la línea interna es la mediana y los bigotes son las observaciones extremas dentro de 1,5 rangos intercuartílicos. Se omiten los puntos externos por legibilidad y su cantidad se conserva en el CSV. Los diamantes indican la media Platt y las cruces, la frecuencia de fraude de la misma banda. El eje vertical es logarítmico. La frecuencia no es otro cuartil ni una observación de la distribución de probabilidades, y las cajas no son intervalos de confianza o una prueba de calibración.

El CSV permite reconstruir la presentación con las mismas estadísticas sin acceder a los archivos por operación. La nueva figura es un diagnóstico descriptivo de evidencia ya evaluada, no una nueva selección de calibrador o política. La revisión no modifica notebooks, datos, modelos, fuentes ni resultados congelados.

Antes de entregar corresponden la lectura conjunta, la incorporación de observaciones y la conservación del aval explícito de la versión PDF definitiva. La revisión técnica no sustituye el aval requerido por el curso.
