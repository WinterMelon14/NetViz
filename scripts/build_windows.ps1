param(
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FrontendRoot = Join-Path $ProjectRoot 'frontend'
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ArtifactName = 'NetViz-windows-x64-1.0.1.zip'
$NpmCommand = Get-Command npm.cmd -ErrorAction Stop
$Node = Join-Path (Split-Path -Parent $NpmCommand.Source) 'node.exe'
$NpmCli = Join-Path (Split-Path -Parent $NpmCommand.Source) 'node_modules\npm\bin\npm-cli.js'

function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Native command failed with exit code $LASTEXITCODE."
    }
}

function Invoke-Npm {
    param([string[]]$NpmArguments)
    if ((Test-Path -LiteralPath $Node) -and (Test-Path -LiteralPath $NpmCli)) {
        Invoke-Checked { & $Node $NpmCli @NpmArguments }
    } else {
        Invoke-Checked { & $NpmCommand.Source @NpmArguments }
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    Invoke-Checked { & py -3.13 -m venv (Join-Path $ProjectRoot '.venv') }
}
if (-not $SkipInstall) {
    Invoke-Checked { & $Python -m pip install --requirement (Join-Path $ProjectRoot 'requirements-build.txt') }
}

Push-Location $FrontendRoot
try {
    if (-not $SkipInstall) { Invoke-Npm @('ci') }
    Invoke-Npm @('run', 'check:branding')
    Invoke-Npm @('run', 'typecheck')
    Invoke-Npm @('run', 'lint')
    Invoke-Npm @('test')
    Invoke-Npm @('run', 'build')
} finally {
    Pop-Location
}

Push-Location $ProjectRoot
try {
    Invoke-Checked { & $Python scripts\check_release_inputs.py }
    Invoke-Checked { & $Python -m unittest discover -s tests -p 'test_*.py' }
    Invoke-Checked { & $Python -m PyInstaller --clean --noconfirm --distpath dist --workpath build packaging\netviz.spec }
    Invoke-Checked { & $Python scripts\audit_bundle.py dist\NetViz }
    Invoke-Checked { & $Python scripts\test_frozen_release.py dist\NetViz\NetViz.exe }
    $ArtifactPath = Join-Path $ProjectRoot "dist\$ArtifactName"
    if (Test-Path -LiteralPath $ArtifactPath) { Remove-Item -LiteralPath $ArtifactPath }
    Compress-Archive -Path (Join-Path $ProjectRoot 'dist\NetViz\*') -DestinationPath $ArtifactPath -CompressionLevel Optimal
    Write-Host "Created $ArtifactPath"
} finally {
    Pop-Location
}
