$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "No se encontró el lanzador 'py'. Instale Python 3.11 o 3.12 desde python.org."
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    Write-Host "Creando entorno virtual en .venv..."
    py -3.11 -m venv $VenvPath
}

Write-Host "Actualizando pip e instalando dependencias..."
& $PythonPath -m pip install --upgrade pip
& $PythonPath -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
& $PythonPath -m pip install --editable $ProjectRoot

Write-Host "Registrando kernel de Jupyter..."
& $PythonPath -m ipykernel install --user --name ieee-cis-pi2 --display-name "Python (IEEE-CIS PI2)"

Write-Host "Entorno preparado. Para abrir Jupyter ejecute:"
Write-Host ".\.venv\Scripts\python.exe -m jupyter lab"
