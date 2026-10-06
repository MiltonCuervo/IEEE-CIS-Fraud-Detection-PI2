# Sensibilidad hipotética de costos en validation

**Estado:** análisis exploratorio de escenarios; ningún costo está aprobado como dato del negocio.
**Datos usados:** 42.047 scores guardados de la ventana `validation_policy` (1.517 fraudes).
**Alcance:** BMR binario vs. umbral fijo 0,5; sin revisión, sin árbol y sin cargar/puntuar test.

## Supuestos de escenario

- `TransactionAmt` conserva su unidad del dataset; **no se denomina USD** porque la fuente/unidad aún no se verificó.
- La mediana de monto de `validation_policy` (la mitad posterior de validation), sin filtrar por etiqueta, es **65 unidades**. Se usa solo como referencia descriptiva para construir `B` hipotético; no es una referencia fijada antes de observar esta ventana.
- `λ ∈ {0,25; 0,50; 1,00}`: fracción hipotética del monto que se perdería al aprobar fraude.
- `B ∈ {0,25; 0,50; 1,00; 2,00} × 65`, es decir **16,25; 32,50; 65,00; 130,00** unidades como costo hipotético de bloquear legítimas.
- `C_block_fraud = 0`: escenario idealizado de bloqueo perfecto, no hecho observado.
- No se simula revisión: no hay estimaciones de su costo, sensibilidad, especificidad ni capacidad.

La política BMR binaria bloquea cuando el costo esperado de bloquear es menor que el de aprobar; el costo realizado se calcula después con las etiquetas reales de validation. Las cifras son proxies de estos supuestos, no dinero ahorrado observado ni una estimación generalizable.

## Resultados

| λ hipotético | B / mediana | B (unidades) | Ahorro proxy vs. fijo | Recall fraude BMR | Legítimas bloqueadas |
|---:|---:|---:|---:|---:|---:|
| 0,25 | 0,25× | 16,25 | 25,46% | 32,10% | 0,95% |
| 0,25 | 0,50× | 32,50 | 18,80% | 26,17% | 0,45% |
| 0,25 | 1,00× | 65,00 | 14,49% | 20,17% | 0,20% |
| 0,25 | 2,00× | 130,00 | 14,94% | 13,65% | 0,10% |
| 0,50 | 0,25× | 16,25 | 40,83% | 40,41% | 2,09% |
| 0,50 | 0,50× | 32,50 | 25,46% | 32,10% | 0,95% |
| 0,50 | 1,00× | 65,00 | 18,80% | 26,17% | 0,45% |
| 0,50 | 2,00× | 130,00 | 14,49% | 20,17% | 0,20% |
| 1,00 | 0,25× | 16,25 | 52,89% | 50,82% | 4,41% |
| 1,00 | 0,50× | 32,50 | 40,83% | 40,41% | 2,09% |
| 1,00 | 1,00× | 65,00 | 25,46% | 32,10% | 0,95% |
| 1,00 | 2,00× | 130,00 | 18,80% | 26,17% | 0,45% |

BMR redujo el costo proxy frente al umbral fijo en los 12 puntos de la grilla: el rango fue **14,49%–52,89%**. Como las decisiones dependen de `λ/B` cuando `C=0` y no hay revisión, los 12 puntos producen seis políticas distintas; las repeticiones son esperadas, no observaciones independientes. El mayor ahorro hipotético ocurrió con `λ=1`, `B=16,25`: recall 50,82% y bloqueo de 4,41% de legítimas. Con `λ=0,25`, `B=65`, el ahorro fue 14,49%, recall 20,17% y bloqueo de legítimas 0,20%.

## Interpretación y límites

El resultado muestra que BMR responde a la relación entre pérdida hipotética por fraude y perjuicio hipotético de bloquear legítimas. Cuanto mayor es `λ/B`, la política acepta bloquear más operaciones: sube el recall y también la tasa de legítimas bloqueadas. No hay una configuración “mejor” sin una función de costos validada por el equipo.

Este es un análisis retrospectivo sobre `validation_policy`, la misma ventana donde se miden los resultados; además, su mediana se usa para escalar `B`. Sirve para sensibilidad y depuración metodológica, no para afirmar ahorro fuera de muestra. Los porcentajes dependen de `λ`, `B` y del supuesto ideal `C_block_fraud=0`; también dependen de la calibración observada. No se probó sensibilidad a `C_block_fraud>0`, revisión, cambios temporales ni incertidumbre estadística. El script/notebook no cargan ni puntúan test. La selección histórica de la partición sí consultó conteos agregados de fraude del candidato a test; por eso el bloque actual no es estrictamente ciego (ver informe de partición).

## Reproducibilidad

- Notebook reproducible: [`notebooks/04_sensibilidad_hipotetica_validation.ipynb`](../notebooks/04_sensibilidad_hipotetica_validation.ipynb)
- Script equivalente: [`scripts/escenarios_hipoteticos_validation.py`](../scripts/escenarios_hipoteticos_validation.py)
- Tabla completa fixed/BMR: [`results/tables/escenarios_hipoteticos_validation.csv`](../results/tables/escenarios_hipoteticos_validation.csv)
- Gráfica: [`results/figures/ahorro_hipotetico_validation.png`](../results/figures/ahorro_hipotetico_validation.png)
- Entrada: `data/interim/validation_scores.parquet`, producido por notebook 03.
- Ejecución en Jupyter: abrir el notebook 04 con kernel `python3` después de ejecutar notebook 03. También se puede ejecutar el script con `.venv\Scripts\python.exe scripts/escenarios_hipoteticos_validation.py`.
- La función de validación/cálculo vive en `src/fraud_cost/scenarios.py` y se comparte entre notebook y script.
- Tras extraer esa lógica compartida, se limpiaron los outputs del notebook para evitar mostrar una corrida asociada al código anterior. La tabla y figura versionadas son resultados de la implementación previa y deben regenerarse para vincularlos a la versión actual.

El `COST_CONFIG` de notebook 03 permanece sin aprobar y el gate de test permanece cerrado. Para una evaluación final, el equipo debe reemplazar estos escenarios por costos sustentados/aprobados, congelar la política y ejecutar test una sola vez.
