$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $Launcher) {
        throw "No se encontró el lanzador 'py'. Instale Python 3.11 o 3.12 y habilite el Python Launcher."
    }

    $SelectedVersion = $null
    foreach ($Version in @("3.12", "3.11")) {
        & $Launcher.Source "-$Version" -c "import sys; print(sys.version)" *> $null
        if ($LASTEXITCODE -eq 0) {
            $SelectedVersion = $Version
            break
        }
    }
    if (-not $SelectedVersion) {
        throw "No se encontró Python 3.11 ni 3.12. Instale una de esas versiones y vuelva a ejecutar el setup."
    }

    Write-Host "Creando entorno virtual en .venv con Python $SelectedVersion..."
    & $Launcher.Source "-$SelectedVersion" -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "No se pudo crear el entorno virtual." }
}

Write-Host "Actualizando pip e instalando dependencias..."
& $PythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Falló la actualización de pip." }
& $PythonPath -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Falló la instalación de requirements.txt." }
& $PythonPath -m pip install --editable $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Falló la instalación editable del paquete local." }

Write-Host "Registrando kernel de Jupyter..."
& $PythonPath -m ipykernel install --user --name ieee-cis-pi2 --display-name "Python (IEEE-CIS PI2)"
if ($LASTEXITCODE -ne 0) { throw "No se pudo registrar el kernel de Jupyter." }

Write-Host "Entorno preparado. Para abrir Jupyter ejecute:"
Write-Host ".\.venv\Scripts\python.exe -m jupyter lab"
