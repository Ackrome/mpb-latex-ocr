param(
    [string]$InputJson = "outputs\im2latex_bundle_check\cdm_predictions.json",
    [string]$OutputDir = "outputs\im2latex_bundle_check\official_cdm_windows",
    [int]$Pools = 4,
    [string]$CdmRoot = "outputs\tools\UniMERNet\cdm",
    [string]$GhostscriptBin = "outputs\tools\conda_ghostscript\Library\bin",
    [string]$PythonExe = ".venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

function Resolve-RepoPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathValue))
}

$InputJson = Resolve-RepoPath $InputJson
$OutputDir = Resolve-RepoPath $OutputDir
$CdmRoot = Resolve-RepoPath $CdmRoot
$GhostscriptBin = Resolve-RepoPath $GhostscriptBin
$PythonExe = Resolve-RepoPath $PythonExe

if (-not (Test-Path -LiteralPath $InputJson)) {
    throw "Missing CDM input JSON: $InputJson"
}
if (-not (Test-Path -LiteralPath (Join-Path $CdmRoot "evaluation.py"))) {
    throw "Missing UniMERNet CDM checkout. Expected: $CdmRoot"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Missing Python executable: $PythonExe"
}

if (Test-Path -LiteralPath (Join-Path $GhostscriptBin "gswin64c.exe")) {
    $env:PATH = "$GhostscriptBin;$env:PATH"
}
if (-not (Get-Command gswin64c.exe -ErrorAction SilentlyContinue)) {
    throw @"
Ghostscript was not found. ImageMagick needs gswin64c.exe for official CDM PDF-to-PNG conversion.

Local conda install option:
  conda create -p outputs\tools\conda_ghostscript -c conda-forge ghostscript -y

Then rerun this script.
"@
}

& $PythonExe scripts\cdm\patch_unimernet_cdm_windows.py --cdm-root $CdmRoot

& $PythonExe -c "import skimage, cv2, matplotlib, PIL, tqdm"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
& $PythonExe (Join-Path $CdmRoot "evaluation.py") -i $InputJson -o $OutputDir -p $Pools

$InputStem = [System.IO.Path]::GetFileNameWithoutExtension($InputJson)
$MetricsPath = Join-Path $OutputDir (Join-Path $InputStem "metrics_res.json")
Write-Host "Official CDM metrics: $MetricsPath"
if (Test-Path -LiteralPath $MetricsPath) {
    Get-Content -LiteralPath $MetricsPath
}
