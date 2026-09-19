# Datos del proyecto

Los datos IEEE-CIS no se versionan en GitHub por su tamaño y por las condiciones de distribución de la competencia. Las carpetas se conservan mediante `.gitkeep`.

## Estructura

```text
data/
├── raw/        # archivos originales, inmutables
├── interim/    # uniones y manifiestos de partición reproducibles
├── processed/  # particiones preparadas para modelado
└── external/   # fuentes auxiliares, si llegan a utilizarse
```

## Archivos originales esperados

```text
data/raw/
├── train_transaction.csv
└── train_identity.csv
```

El proyecto académico utiliza `train_transaction.csv` y `train_identity.csv`, porque contienen o complementan las observaciones etiquetadas. Los archivos `test_*` de Kaggle no tienen `isFraud` y **no son** el test temporal del experimento, por lo que no se almacenan en este repositorio local. Train, validación y test temporal se obtendrán de los datos etiquetados y se documentarán en un notebook posterior.

## Obtención

### Opción A — descarga manual

1. Ingresar a la competencia IEEE-CIS Fraud Detection en Kaggle.
2. Aceptar sus reglas.
3. Descargar y extraer los archivos.
4. Copiarlos en `data/raw/` sin modificar sus nombres.

### Opción B — CLI de Kaggle

Después de preparar el entorno y autenticar la CLI fuera del repositorio:

```powershell
.\scripts\download_data.ps1
```

Nunca se deben escribir tokens o contraseñas en notebooks, scripts versionados o archivos `.env` publicados.

## Reglas

- `raw/` es inmutable: ningún notebook debe sobrescribir sus archivos.
- Los derivados deben poder regenerarse desde código.
- No crear transformaciones globales antes de definir la partición temporal.
- No usar el test de Kaggle para evaluar el proyecto porque carece de etiqueta.
