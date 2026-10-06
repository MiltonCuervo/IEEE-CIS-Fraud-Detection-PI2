# Estado vivo del proyecto

**Actualizado:** 2026-10-03  
**Rama revisada:** `codex/bmr-validation-notebook`
**Commit base al iniciar este ciclo:** `223d656` (`Add hypothetical validation sensitivity notebook`)
**Alcance de este ciclo:** auditoría e implementación de correcciones de código y documentación. No se ejecutaron los notebooks ni se recalcularon modelos/resultados.

## Objetivo y estado

El proyecto estudia decisiones sensibles al costo para detección de fraude IEEE-CIS. La implementación disponible comprende EDA, partición temporal, un baseline XGBoost calibrado y una sensibilidad BMR binaria con costos hipotéticos.

**Estado:** prototipo de investigación técnicamente estructurado. Los costos no están aprobados, la unidad de `TransactionAmt` no está confirmada y no hay evaluación de scores sobre el bloque test. Por tanto, no hay una conclusión de ahorro financiero ni validación de producción.

## Arquitectura y flujo implementado

`train_transaction.csv` + `train_identity.csv` → validación/unión por `TransactionID` → EDA → partición temporal con gaps → manifiesto Parquet → preprocesamiento ajustado en train → XGBoost fijo → calibración Platt en validation temprana → métricas y scores en `validation_policy` → análisis BMR hipotético.

Los notebooks 01–04 narran las etapas. La lógica reutilizable para carga/unión está en `src/fraud_cost/data.py`, para splits en `src/fraud_cost/split.py` y para validar scores/calcular escenarios en `src/fraud_cost/scenarios.py`. `scripts/escenarios_hipoteticos_validation.py` y el notebook 04 llaman a esta lógica compartida.

## Datos y artefactos observados en la auditoría anterior

- Datos raw locales: `train_transaction.csv` (683.351.067 bytes) y `train_identity.csv` (26.529.680 bytes), ignorados por Git.
- Manifiesto local: `data/interim/split_manifest.parquet` (5.288.710 bytes), ignorado por Git.
- Scores locales de `validation_policy`: `data/interim/validation_scores.parquet`, ignorados por Git.
- Artefactos versionados: tablas/figuras EDA, partición, métricas de validación y sensibilidad; el modelo serializado está ignorado por Git.
- La auditoría cruzada reportó para `validation_policy` 42.047 filas / 1.517 fraudes; AP 0,4671, recall fijo 0,5 = 0,2657, precision = 0,7589, Brier = 0,02518 y log-loss = 0,10184.
- El notebook 03 guardado no contiene execution counts ni outputs; sus artefactos corresponden a una corrida previa, pero no quedan vinculados inequívocamente al código guardado.
- En este ciclo se editaron fuentes de notebook 02/03/04 y se limpiaron outputs de 02/04 para no mostrar resultados ejecutados por código anterior. Sus artefactos versionados no se regeneraron y requieren ejecución posterior.

## Cambios hechos en este ciclo

- `src/fraud_cost/split.py` ahora rechaza índices desalineados y tiempos ausentes al validar la partición.
- `src/fraud_cost/scenarios.py` centraliza validación de scores y cálculo de escenarios; exige IDs únicos, clases válidas, probabilidades dentro de [0,1], acciones de baseline coherentes y parámetros válidos.
- `summarize_temporal_split()` puede ocultar etiquetas de bloques reservados; notebook 02 ahora omite métricas de fraude de test en resúmenes y candidatos futuros.
- El script y notebook 04 consumen el evaluador de escenarios compartido. El script de descarga ya informa que espera dos CSV.
- `scripts/setup.ps1` elige Python 3.12 y luego 3.11 cuando crea el entorno y comprueba los fallos de instalación.
- `scripts/data_provenance.py` genera hashes SHA-256 de los CSV y registra Python, dependencias y commit en un manifiesto local ignorado por Git.
- Notebook 03 requiere unidades explícitas y coincidentes para `TransactionAmt` y costos, acota `loss_fraction` a [0,1], fortalece validación de entradas y no imprime métricas agregadas de test.
- README e informes declaran el uso histórico de conteos agregados de fraude del candidato a test durante la comparación de gaps. El bloque test actual no se describe como holdout estrictamente intocado.
- Se corrigió el desempate: el código aprueba en igualdad de riesgos y la condición analítica de bloqueo es estricta.
- Los documentos usan “unidades del dataset” en lugar de USD mientras la fuente no confirme moneda/unidad; aclaran que los 12 puntos hipotéticos representan seis políticas distintas.

## Lo que sigue pendiente

1. Ejecutar `scripts/data_provenance.py` y conservar el manifiesto junto con la corrida.
2. Ejecutar en un entorno Python soportado 01→02→03→04, guardar outputs actuales y regenerar el CSV/figura de sensibilidad usando el módulo compartido. Notebook 02 no debe resumir etiquetas de test.
3. Registrar la configuración junto con el manifiesto de datos para cada artefacto generado.
4. Confirmar la unidad de `TransactionAmt` y aprobar `λ`, `B`, `C_block_fraud`; revisión sigue opcional y sin costos operativos.
5. Acordar si se usará el test actual como evaluación condicionada, declarando su exposición agregada, o si se obtendrá un periodo/dataset nuevo para evaluación independiente estricta.
6. Añadir pruebas automatizadas y ejecutarlas en CI o localmente; `pyproject.toml` configura pytest pero el repositorio no contiene carpeta `tests/`.
7. Corregir posibles diferencias adicionales de inventario tras la próxima ejecución reproducible.

## Riesgos y datos requeridos

- **Metodológico:** el test temporal no es estrictamente ciego porque sus conteos agregados de fraude participaron en comparar gaps.
- **Económico:** todos los costos usados en notebook 04 son hipotéticos; `C_block_fraud=0` idealiza prevención perfecta.
- **Unidad/procedencia:** falta confirmar unidad monetaria, hash/versión de los CSV y metadatos del dataset.
- **Reproducibilidad:** notebook 03 no tiene resultados incrustados; notebooks 02 y 04 quedaron deliberadamente sin outputs después de cambios y aún no se han reejecutado.
- **Validación de software:** no hay suite `tests/`; este ciclo no ejecutó tests ni notebooks.
- **Gate de test:** sus flags en notebook 03 son controles de proceso editables, no un sistema de aprobación/auditoría inmutable.

## Siguiente acción recomendada

En un entorno compatible y con dependencias instaladas, ejecutar 03 y 04 desde cero con los CSV/manifiesto locales, guardar salidas y revisar que el CSV/figura generados coincidan con los artefactos versionados. Después, registrar hash y versiones. Mantener costos y test sin aprobar hasta resolver unidad, matriz económica y estrategia de evaluación independiente.
