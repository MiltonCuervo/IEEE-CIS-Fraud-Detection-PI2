param(
    [string]$PythonExecutable,
    [string]$BrokenEnvironmentBackupDirectory
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Falló: $Executable $($Arguments -join ' ')" }
}

$HealthyEnvironment = $false
if (Test-Path -LiteralPath $PythonPath) {
    & $PythonPath -c "import sys; print(sys.version)" *> $null
    $HealthyEnvironment = $LASTEXITCODE -eq 0
}

if (-not $HealthyEnvironment) {
    if ($PythonExecutable) {
        $BasePython = (Resolve-Path -LiteralPath $PythonExecutable).Path
        Invoke-Checked $BasePython @("-c", "import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), 'Use Python 3.11 o 3.12'")
    } else {
        $Launcher = Get-Command py -ErrorAction SilentlyContinue
        if (-not $Launcher) { throw "Instale Python 3.11/3.12 o indique -PythonExecutable con su ruta." }
        $BasePython = $null
        foreach ($Version in @("3.12", "3.11")) {
            $FoundPython = & $Launcher.Source "-$Version" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0) { $BasePython = $FoundPython.Trim(); break }
        }
        if (-not $BasePython) { throw "No se encontró Python 3.11/3.12." }
    }
    if (Test-Path -LiteralPath $VenvPath) {
        # Resolvemos y comprobamos ambos destinos antes de mover el entorno.
        $ResolvedVenv = (Resolve-Path -LiteralPath $VenvPath).Path
        if ($ResolvedVenv -ne [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot '.venv'))) {
            throw "La ruta de .venv no coincide con el proyecto."
        }
        if (-not $BrokenEnvironmentBackupDirectory) {
            $BrokenEnvironmentBackupDirectory = Join-Path $ProjectRoot ('.venv.backup.' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
        }
        $BackupPath = [System.IO.Path]::GetFullPath($BrokenEnvironmentBackupDirectory)
        if ($BackupPath.StartsWith($ResolvedVenv + [System.IO.Path]::DirectorySeparatorChar) -or $BackupPath -eq $ResolvedVenv -or (Test-Path -LiteralPath $BackupPath)) {
            throw "El respaldo debe ser un destino nuevo y estar fuera del entorno original."
        }
        Move-Item -LiteralPath $ResolvedVenv -Destination $BackupPath
        Write-Host "Entorno anterior conservado en $BackupPath"
    }
    Invoke-Checked $BasePython @("-m", "venv", $VenvPath)
}

Invoke-Checked $PythonPath @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $PythonPath @("-m", "pip", "install", "-r", (Join-Path $ProjectRoot 'requirements.txt'))
Invoke-Checked $PythonPath @("-m", "pip", "install", "--editable", $ProjectRoot)
Invoke-Checked $PythonPath @("-m", "ipykernel", "install", "--user", "--name", "ieee-cis-pi2", "--display-name", "Python (IEEE-CIS PI2)")
Write-Host "Entorno preparado: .\.venv\Scripts\python.exe -m jupyter lab"
