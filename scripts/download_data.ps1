$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RawPath = Join-Path $ProjectRoot "data\raw"
$ExpectedFiles = @(
    "train_transaction.csv",
    "train_identity.csv"
)

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Primero ejecute scripts\setup.ps1 para crear el entorno."
}

$MissingFiles = @($ExpectedFiles | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $RawPath $_))
})

if ($MissingFiles.Count -eq 0) {
    Write-Host "Los cinco archivos esperados ya están en data\raw. No se descargará nada."
    exit 0
}

Write-Host "Faltan: $($MissingFiles -join ', ')"
Write-Host "La descarga requiere haber aceptado las reglas de la competencia y autenticado la CLI de Kaggle."
New-Item -ItemType Directory -Force -Path $RawPath | Out-Null
foreach ($FileName in $MissingFiles) {
    & $PythonPath -m kaggle competitions download ieee-fraud-detection -f $FileName -p $RawPath --unzip
}

foreach ($FileName in $ExpectedFiles) {
    $FilePath = Join-Path $RawPath $FileName
    if (-not (Test-Path -LiteralPath $FilePath)) {
        throw "La descarga terminó, pero falta $FileName."
    }
}

Write-Host "Descarga validada en data\raw."
