# Optimización basada en costos para el bloqueo de transacciones sospechosas

Proyecto Integrador II — Ingeniería de Sistemas, Universidad de Antioquia, 2026-2.

El proyecto compara tres estrategias de decisión sobre probabilidades calibradas de fraude en IEEE-CIS: decisión convencional, Bayes Minimum Risk y un árbol sensible al costo. El objetivo principal es reducir el costo económico fuera de muestra; no maximizar aisladamente una métrica de clasificación.

## Pregunta de investigación

¿En qué medida la optimización de la regla de decisión basada en costos permite mejorar el ahorro financiero frente a un umbral fijo convencional para el bloqueo de transacciones sospechosas en el conjunto de datos IEEE-CIS, y cómo varía este beneficio ante diferentes costos administrativos de revisión?

## Estructura

```text
.
├── data/
│   ├── raw/          # Archivos originales, inmutables y no versionados
│   ├── interim/      # Datos intermedios reproducibles
│   └── processed/    # Particiones listas para modelar
├── notebooks/        # Relato experimental breve y numerado
├── src/fraud_cost/   # Lógica reutilizable y comprobable
├── docs/             # Decisiones, protocolo y preparación de sustentaciones
├── results/
│   ├── figures/      # Figuras finales exportadas
│   ├── tables/       # Métricas y tablas finales
│   └── models/       # Modelos y calibradores serializados
└── requirements.txt
```

`data/` y los artefactos de `results/` conservan su estructura mediante archivos `.gitkeep`, pero su contenido no se sube al repositorio.

## Preparación reproducible

1. Crear y activar un entorno virtual con Python 3.11.
2. Instalar dependencias y el paquete local en modo editable:

   ```bash
   python -m pip install -r requirements.txt
   python -m pip install -e .
   ```

3. Descargar desde Kaggle los archivos de la competencia IEEE-CIS Fraud Detection y ubicarlos sin modificar en `data/raw/`:

   - `train_transaction.csv`
   - `train_identity.csv`

4. No escribir credenciales en notebooks. Configurar la API de Kaggle fuera del repositorio siguiendo su documentación oficial o descargar los datos manualmente.
5. Abrir Jupyter desde la raíz del repositorio y ejecutar los notebooks en orden.


## Secuencia experimental

| Notebook | Propósito | Salida esperada |
|---|---|---|
| `01_exploracion_datos.ipynb` | Validar estructura, objetivo, monto, tiempo y ausencia | Evidencia para decisiones de preparación |
| `02_particion_temporal.ipynb` | Congelar train/validación/test y gaps | Manifiesto de particiones |

Los notebooks muestran preguntas, decisiones, llamadas principales y resultados. La carga, validaciones, métricas y reglas repetibles viven en `src/`.

## Reglas metodológicas

- La clasificación y la decisión económica son etapas distintas. El corte clasificatorio de 0.5 no se optimiza en este proyecto.
- La evaluación final utiliza una partición temporal fuera de muestra. El test no participa en selección, transformación ni calibración.
- No se remuestrea automáticamente: alterar la prevalencia puede deteriorar la interpretación probabilística.
- La calibración se evalúa y documenta antes de aplicar Bayes Minimum Risk.
- Todas las estrategias se comparan sobre las mismas observaciones y con la misma matriz de costos.
- La métrica principal es costo total/ahorro. AUC-PR, recall y tasa de legítimas bloqueadas explican el comportamiento.
- `sample_weight` es una aproximación sensible al costo, no un sustituto conceptualmente idéntico de un árbol con costos dependientes del ejemplo.

## Trazabilidad

Cada cifra del informe final debe poder rastrearse a: versión de datos, partición temporal, configuración, notebook y archivo exportado en `results/`. Las decisiones no triviales se registran en [`docs/decisiones_tecnicas.md`](docs/decisiones_tecnicas.md).
