<#
.SYNOPSIS
    Section 6 Empirical Ablation Pipeline — PowerShell Wrapper

.DESCRIPTION
    Convenience wrapper around section6_run.py for the ISM 2026 paper.
    Runs the evidence-first pipeline: build once, filter 7 times.

.EXAMPLE
    # Full run (all stages, ~25 hours)
    .\section6_run.ps1 -Workers 3 -Stage all

    # Evidence build only
    .\section6_run.ps1 -Workers 3 -Stage evidence

    # Re-materialise all presets (seconds)
    .\section6_run.ps1 -Stage materialise

    # Re-materialise one preset
    .\section6_run.ps1 -Stage materialise -Preset quality_focused

    # Analysis only
    .\section6_run.ps1 -Stage analyse

    # Verify data integrity
    .\section6_run.ps1 -Stage verify
#>
param(
    [ValidateSet("all", "verify", "evidence", "materialise", "analyse")]
    [string]$Stage = "all",

    [int]$Workers = 3,

    [string]$Preset = "",

    [string]$OutputRoot = "",

    [string]$DataDir = "",

    [string]$Config = "",

    [switch]$NoResume,

    [switch]$Verbose,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonScript = Join-Path $scriptDir "section6_run.py"

if (-not (Test-Path $pythonScript)) {
    Write-Error "Cannot find $pythonScript"
    exit 1
}

$pyArgs = @("--stage", $Stage, "--workers", $Workers)

if ($Preset)     { $pyArgs += @("--preset", $Preset) }
if ($OutputRoot) { $pyArgs += @("--output-root", $OutputRoot) }
if ($DataDir)    { $pyArgs += @("--data-dir", $DataDir) }
if ($Config)     { $pyArgs += @("--config", $Config) }
if ($NoResume)   { $pyArgs += "--no-resume" }
if ($Verbose)    { $pyArgs += "--verbose" }
if ($DryRun)     { $pyArgs += "--dry-run" }

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Section 6 Pipeline — $Stage" -ForegroundColor Cyan
Write-Host "  Workers: $Workers" -ForegroundColor Cyan
Write-Host "  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

python $pythonScript @pyArgs
$exitCode = $LASTEXITCODE

$stopwatch.Stop()
$elapsed = $stopwatch.Elapsed

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Elapsed: $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "  Exit code: $exitCode" -ForegroundColor $(if ($exitCode -eq 0) { "Green" } else { "Red" })
Write-Host "========================================" -ForegroundColor Cyan

exit $exitCode
