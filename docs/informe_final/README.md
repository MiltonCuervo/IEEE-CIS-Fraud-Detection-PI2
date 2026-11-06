# Artículo final: borrador para revisión

Proyecto Integrador II, Universidad de Antioquia, 2026-2.
Equipo: Milton Alejandro Cuervo Ramírez, Oswald David Gutiérrez Cortina y Diego Alejandro Rendón Suaza.
Profesora acompañante: María Bernarda Salazar Sánchez.

## Estado y entrega

Este es un borrador completo para revisión del equipo y de la profesora. No acredita aval docente.
La entrega del curso exige PDF, máximo seis páginas, formato IEEE para conferencias y aval explícito antes de su entrega, a más tardar el 16 de noviembre de 2026.

`informe_final.pdf` tiene seis páginas, incluidas las referencias. Usa la clase oficial IEEEtran en modo conference, tamaño carta, cuerpo de 10 puntos y dos columnas. Conserva el título y los autores de la propuesta.

## Archivos necesarios

- `informe_final.tex`: fuente editable; incluye texto, ecuaciones, seis tablas y doce referencias.
- `IEEEtran.cls`: clase oficial sin modificaciones, obtenida de CTAN; conserva su licencia en la cabecera.
- `compromiso_operativo.pdf`: figura vectorial del compromiso de intervención.
- `calibracion_por_monto.pdf`: figura vectorial del diagnóstico por monto.
- `informe_final.pdf`: artículo compilado y revisado visualmente.

Para editar en Overleaf, subir el archivo .tex, la clase y las dos figuras, seleccionar `informe_final.tex` como documento principal y compilar con XeLaTeX. En una instalación LaTeX local compatible, ejecutar `xelatex informe_final.tex` dos veces. También puede compilarse con `tectonic informe_final.tex`. La versión revisada se compiló con Tectonic 0.17.0. Verificar de nuevo la paginación después de editar o cambiar de compilador; los archivos .aux, .log y .out son temporales.

## Evidencia utilizada

Las tablas y figuras proceden de los resultados agregados ya publicados en `results/tables/`, especialmente `seleccion_ordinal_resumen.csv`, `test_metricas_probabilidades.csv`, `test_costos.csv` y `test_calibracion_montos.csv`. Las cifras adicionales corresponden a los informes de etapas 01-08 y a sus tablas. Esta redacción no entrena modelos ni vuelve a puntuar test. Las figuras usan únicamente CSV agregados.

La representación, calibración y políticas describen el experimento realmente ejecutado. El artículo declara los cambios frente a la propuesta, la consulta histórica de etiquetas agregadas finales, el carácter hipotético de Ca, los límites del ordinal y del árbol ponderado, y el deterioro de Brier/log-loss de Platt en test.

Antes de entregar: revisión conjunta del contenido y autoría, incorporación de observaciones de la profesora, comprobación de la versión PDF definitiva y conservación de la evidencia de aval explícito. La revisión técnica y la compilación no sustituyen ese aval.
