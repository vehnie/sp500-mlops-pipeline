<#
.SYNOPSIS
    Start the local MLflow Tracking Server and Model Registry for this project.

.DESCRIPTION
    Resolves the artifact directory to an absolute file:// URI (required on
    Windows so MLflow does not mistake the drive letter for an artifact-store
    scheme) and launches `mlflow server` with artifact proxying enabled.

    Run this before serving the model with uvicorn (README step 5) or with
    Docker (README step 7).

.PARAMETER Docker
    Bind to 0.0.0.0 and allow the host.docker.internal origin so a container
    started with MLFLOW_TRACKING_URI=http://host.docker.internal:5000 can reach
    the server. Without this switch the server binds to 127.0.0.1 (local use).

.PARAMETER Port
    Port to listen on. Defaults to 5000.

.EXAMPLE
    .\scripts\start_mlflow_server.ps1
    Start the server for local uvicorn serving.

.EXAMPLE
    .\scripts\start_mlflow_server.ps1 -Docker
    Start the server so a Docker container can reach it.
#>
[CmdletBinding()]
param(
    [switch]$Docker,
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"

# Repo root = parent of the folder containing this script, so it works no
# matter the current working directory.
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# Ensure the artifact destination exists, then build an absolute file:// URI.
$artifactDir = Join-Path $repoRoot "data\08_reporting\mlflow_artifacts"
if (-not (Test-Path $artifactDir)) {
    New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
}
$artifactUri = [System.Uri]::new((Resolve-Path $artifactDir).Path).AbsoluteUri

# Prefer the project virtual environment's mlflow; fall back to PATH.
$venvMlflow = Join-Path $repoRoot ".venv\Scripts\mlflow.exe"
$mlflow = if (Test-Path $venvMlflow) { $venvMlflow } else { "mlflow" }

if ($Docker) {
    $bindHost = "0.0.0.0"
    $allowedHosts = "localhost:*,127.0.0.1:*,host.docker.internal:$Port"
} else {
    $bindHost = "127.0.0.1"
    $allowedHosts = $null
}

Write-Host "Repo root        : $repoRoot"
Write-Host "Backend store    : sqlite:///mlflow.db"
Write-Host "Artifact root    : $artifactUri"
Write-Host "Binding          : http://$bindHost`:$Port"
Write-Host "Using mlflow     : $mlflow"
Write-Host ""

$mlflowArgs = @(
    "server",
    "--host", $bindHost,
    "--port", "$Port",
    "--backend-store-uri", "sqlite:///mlflow.db",
    "--default-artifact-root", "mlflow-artifacts:/",
    "--serve-artifacts",
    "--artifacts-destination", $artifactUri
)
if ($allowedHosts) {
    $mlflowArgs += @("--allowed-hosts", $allowedHosts)
}

& $mlflow @mlflowArgs
