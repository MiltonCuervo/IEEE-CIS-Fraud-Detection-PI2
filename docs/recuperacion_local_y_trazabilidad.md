# Recuperación local y trazabilidad de las etapas ordinales

## Alcance

> Este documento describe la recuperación histórica en el entorno de ejecución
> documentado; no supone una configuración idéntica en todas las instalaciones. La evaluación final ya se
> completó en el [08](08_evaluacion_temporal_final.md) con otra versión local
> explícita y verificada. No ejecutar publicaciones de recuperación sobre las
> dependencias congeladas de una copia cerrada. Para consultarla, usar la auditoría
> del README; una reproducción nueva requiere su propia área y versión.

Restablecemos los prerrequisitos para continuar después del notebook 04. Conservamos datos originales, manifiesto, parámetros, representación ordinal y selección. La recuperación no introduce otro candidato, no cambia Ca, no puntúa test y no constituye la etapa 05.

## Diagnóstico observado

- Los CSV y el manifiesto local coincidían con sus hashes registrados.
- El paquete `comparacion_modelos_ordinal.joblib` y los árboles de la corrida ordinal no estaban en sus rutas locales esperadas.
- El baseline y los scores presentes no coincidían con los hashes de la corrida vigente. Su presencia no autorizaba reutilizarlos.
- El Python de `.venv` sí iniciaba fuera del aislamiento, pero era 3.13.2, le faltaban LightGBM/CatBoost y tenía versiones diferentes de NumPy/XGBoost. El error de acceso inicial no demostraba que el intérprete estuviera roto.
- `costs.py` coincidía con la fuente histórica únicamente al convertir CRLF a LF. Restauramos los bytes LF y añadimos una regla específica en `.gitattributes`; no cambiamos sus instrucciones ni los hashes históricos.
- El README y algunos informes no incorporaban todavía la selección completada del 04.

## Entorno

Preservamos el entorno anterior en `.venv.backup.*`. Preparamos `.venv` con Python 3.12 y las siete versiones principales de 03/04 fijadas en `requirements-reproduction.txt`. El setup admite restricciones opcionales, comprueba que un entorno reutilizado use Python 3.11/3.12 y conserva los entornos incompatibles antes de reemplazarlos.

La corrida original utilizó Python 3.11.9: no afirmamos reproducción binaria. El registro de reconstrucción conserva la versión efectiva y un `pip freeze` local de todas las dependencias. El kernel `ieee-cis-pi2` apunta al entorno del proyecto. Si este Python base pertenece a una instalación administrada que cambia de ubicación, recreamos el entorno con un intérprete compatible mediante `-PythonExecutable`; no editamos manualmente `pyvenv.cfg`.

## Procedimiento reproducible

Desde la raíz:

```powershell
.\scripts\setup.ps1 -ConstraintsFile .\requirements-reproduction.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/recover_reference_artifacts.py
.\.venv\Scripts\python.exe scripts/recover_reference_artifacts.py --verify-only
```

Si no se detecta Python 3.11/3.12, añadimos `-PythonExecutable <ruta-a-python.exe>` al setup. La reconstrucción requiere tiempo de entrenamiento, espacio para copias y memoria comparable al 03.

1. Verificamos fuentes, CSV, manifiesto y la cadena histórica 03→protocolo 04→registro 04. Comprobamos las versiones principales antes de ajustar.
2. Creamos un área local nueva en `data/interim/reference_recovery/<run_id>/`. Guardamos el plan y las tolerancias antes de entrenar. Los CSV se enlazan dentro del mismo volumen para evitar duplicarlos; el notebook solo los lee.
3. Ejecutamos una copia del notebook 03 con sus fuentes, manifiesto y directorios de salida aislados. El notebook activo y las tablas/figuras históricas no se sobrescriben.
4. Comparamos métricas de cuatro modelos, diagnóstico raw/Platt, costos, inventario y vocabularios. Exigimos mismas filas, columnas, orden y tipos; para números usamos `atol=rtol=1e-8`, definidos antes de ejecutar. Si no se cumple, no publicamos ni ampliamos las tolerancias retrospectivamente.
5. Cargamos únicamente los modelos recién generados por esta ejecución propia. Verificamos los scores contra las 42.047 filas de diagnóstico, la matriz de 432 columnas, las predicciones del baseline y las métricas raw/Platt de los cuatro modelos. La etiqueta nunca entra al predictor.
6. Respaldamos destinos anteriores y publicamos cuatro archivos: scores XGBoost, baseline, paquete ordinal y árboles. Si falla la publicación o su auditoría, restauramos los destinos anteriores.
7. Guardamos `results/tables/recuperacion_03_config.json` como recibo versionable. Identifica hashes originales y reconstruidos, evidencia local, parámetros seleccionados y resultado de las comprobaciones.

## Contrato para continuar

```python
from fraud_cost.artifacts import verify_reference_artifacts

registro = verify_reference_artifacts(PROJECT_ROOT)
# Solo después de verificar: cargar artefactos propios mediante joblib.
```

El verificador acepta originales con hashes idénticos o una reconstrucción completa con recibo, equivalencia e integridad verificadas. No normaliza hashes silenciosamente ni acepta un archivo solo por su nombre. Un recibo descargado sin su evidencia y sin los artefactos locales es insuficiente: hay que reconstruirlos o recuperar el conjunto completo verificable.

Los hashes originales del 03 siguen describiendo sus binarios históricos. No debemos copiarlos sobre los nuevos ni esperar que la serialización sea idéntica entre versiones de Python. El recibo añade una relación explícita entre ambas corridas y el 04 se mantiene intacto.

## Continuación metodológica

El 04 seleccionó `lightgbm_referencia`, AP media interna 0,588117, con parámetros iguales a los del LightGBM del 03. La etapa 05 podrá reutilizar su ajuste y su propio Platt después de verificar el paquete. Deberá documentar esa adopción, diagnosticar calibración y reconstruir la comparación económica con las probabilidades LightGBM; los scores genéricos `validation_scores.parquet` siguen siendo de XGBoost.

La recuperación no aprueba calibrador ni costos finales y no demuestra mayor ahorro con LightGBM. Test sigue sin puntuar y conserva la limitación de exposición histórica a etiquetas agregadas.

## Resultado observado de la recuperación local

La corrida `20261103T025750Z-524586e7` terminó con estado `verified_reconstruction` en [el recibo de recuperación](../results/tables/recuperacion_03_config.json). Utilizó Python 3.12.14 y las siete versiones principales originales. El entorno anterior quedó conservado en `.venv.backup.20261102-215342`.

| Comprobación | Resultado |
|---|---|
| Tabla de cuatro predictores y diagnóstico raw/Platt | Diferencia absoluta máxima aproximada de 1,71 × 10⁻¹³ |
| Comparación económica: 18 filas | Sin diferencias numéricas |
| Inventario: 432 posiciones | Sin diferencias numéricas |
| Vocabularios: 15.101 correspondencias | Sin diferencias numéricas |
| Inferencia de cuatro modelos serializados | Métricas verificadas en las mismas 42.047 operaciones posteriores |
| Baseline y Platt serializados | Reproducen los scores publicados |
| Seis árboles serializados | Reproducen las 18 filas de comparación económica |
| Modelo LightGBM recuperado | Sus parámetros coinciden con la configuración seleccionada en el 04 |
| Cadena de procedencia 03/04 | Verificada; registros históricos sin cambios |
| Entorno y kernel | Python compatible, kernel del proyecto y `pip check` sin incompatibilidades |
| Suite vigente | 13 pruebas aprobadas |

Los modelos y scores activos ya pertenecen a la reconstrucción ordinal verificada. Los dos archivos locales preexistentes quedaron respaldados en `data/interim/reference_recovery/<run_id>/previous_artifacts/`. El test no fue puntuado. Quedan disponibles los prerrequisitos locales para la etapa 05; esta recuperación no la implementa ni aprueba decisiones finales.

## Continuidad después de la ampliación del 04

La nueva ejecución del 04 amplió diagnósticos sin modificar el ajuste 03 ni su configuración LightGBM adoptable. `scripts/revalidate_reference_recovery.py` revalidó el vínculo y preservó el recibo anterior. No reasigna hashes a otros binarios: comprueba el mismo 03, toda su evidencia, las tablas y la inferencia antes de actualizar la referencia al 04 vigente.

Después, `scripts/reproduce_validation.py` ejecutó 05/06 y regeneró sus archivos locales; las doce tablas coincidieron dentro de atol=rtol=10⁻⁸. La preparación de la etapa 07 se documenta en [protocolo final](07_protocolo_final.md). Una vez congelado, no se reemplazan sus dependencias para continuar la evaluación; una reproducción distinta requiere una versión de experimento propia.
