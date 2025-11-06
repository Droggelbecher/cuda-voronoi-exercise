#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Nsight Systems profiling wrapper.
.DESCRIPTION
    Supports named options for trace, sample, ctxsw, output, report.
    Everything else is treated as the command to profile.
#>

# Default options
$Trace  = "cuda"
$Sample = "none"
$Ctxsw  = "none"
$Output = "nsys_easy"
$Report = "cuda_gpu_sum"

# Parse named options manually
$CommandArgs = @()
$i = 0
while ($i -lt $args.Count) {
    switch ($args[$i]) {
        "-Trace"  { $i++; $Trace  = $args[$i] }
        "-Sample" { $i++; $Sample = $args[$i] }
        "-Ctxsw"  { $i++; $Ctxsw  = $args[$i] }
        "-Output" { $i++; $Output = $args[$i] }
        "-Report" { $i++; $Report = $args[$i] }
        default   { $CommandArgs += $args[$i] } # Everything else is command
    }
    $i++
}

if ($CommandArgs.Count -eq 0) {
    Write-Host "Usage: .\nsys_easy.ps1 [-Trace trace] [-Sample sample] [-Ctxsw ctxsw] [-Output output] [-Report report] command [args...]"
    exit 1
}

# Build nsys profile arguments
$profileArgs = @(
    "profile",
    "--trace=$Trace",
    "--sample=$Sample",
    "--cpuctxsw=$Ctxsw",
    "--force-overwrite=true",
    "-o", $Output
) + $CommandArgs

Write-Host "Running nsys profile:"
Write-Host "  Trace : $Trace"
Write-Host "  Sample: $Sample"
Write-Host "  Ctxsw : $Ctxsw"
Write-Host "  Output: $Output"
Write-Host "  Report: $Report"
Write-Host "Command: $($CommandArgs -join ' ')"
Write-Host "--------------------------------------------"

# Run nsys profile
& nsys @profileArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "nsys profile command failed."
    exit $LASTEXITCODE
}

# Run nsys stats
$statsArgs = @(
    "stats",
    "--force-export=true",
    "-r", $Report,
    "$Output.nsys-rep"
)

Write-Host "`nRunning nsys stats..."
& nsys @statsArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "nsys stats command failed."
    exit $LASTEXITCODE
}

Write-Host "`n✅ Profiling and report generation complete."
