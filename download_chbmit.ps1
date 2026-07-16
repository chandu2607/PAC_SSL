# download_chbmit_fast.ps1 — Parallel download of CHB-MIT dataset
# Downloads multiple files simultaneously using background jobs for speed
# Skips chb12, resumes already-downloaded files

$ErrorActionPreference = "Continue"
$baseUrl  = "https://physionet.org/files/chbmit/1.0.0"
$destRoot = Join-Path $PSScriptRoot "data\chb-mit"
$logFile  = Join-Path $PSScriptRoot "data\download_log.txt"
$maxParallel = 8  # concurrent downloads

$patients = @(
    "chb01","chb02","chb03","chb04","chb05","chb06","chb07","chb08",
    "chb09","chb10","chb11",
    "chb13","chb14","chb15","chb16","chb17","chb18","chb19","chb20",
    "chb21","chb22","chb23","chb24"
)

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

# Read RECORDS
$recordsFile = Join-Path $destRoot "RECORDS"
$allRecords = Get-Content $recordsFile | Where-Object { $_ -match '\.edf$' }

# Build full file list (summary.txt + .edf files) excluding chb12
$allFiles = @()
foreach ($patient in $patients) {
    $patientDir = Join-Path $destRoot $patient
    if (-not (Test-Path $patientDir)) {
        New-Item -ItemType Directory -Force -Path $patientDir | Out-Null
    }
    # Add summary file
    $allFiles += "$patient/$patient-summary.txt"
    # Add .edf files
    $allFiles += ($allRecords | Where-Object { $_ -match "^$patient/" })
}

# Filter out already-downloaded files
$toDownload = @()
$skipped = 0
foreach ($relPath in $allFiles) {
    $localPath = Join-Path $destRoot ($relPath -replace '/', '\')
    if ((Test-Path $localPath) -and ((Get-Item $localPath).Length -gt 0)) {
        $skipped++
    } else {
        $toDownload += $relPath
    }
}

Log "=== CHB-MIT Fast Parallel Download ==="
Log "Total files in manifest: $($allFiles.Count)"
Log "Already downloaded (skipped): $skipped"
Log "Files to download: $($toDownload.Count)"
Log "Parallel workers: $maxParallel"

if ($toDownload.Count -eq 0) {
    Log "All files already downloaded!"
    exit 0
}

# Download in batches using Start-Process for true parallelism
$totalCount = $toDownload.Count
$completed = 0
$failed = @()

for ($i = 0; $i -lt $totalCount; $i += $maxParallel) {
    $batch = $toDownload[$i..([math]::Min($i + $maxParallel - 1, $totalCount - 1))]
    $processes = @()

    foreach ($relPath in $batch) {
        $url = "$baseUrl/$relPath"
        $localPath = Join-Path $destRoot ($relPath -replace '/', '\')
        $localDir = Split-Path $localPath -Parent
        if (-not (Test-Path $localDir)) {
            New-Item -ItemType Directory -Force -Path $localDir | Out-Null
        }

        # Start curl.exe as a background process
        $proc = Start-Process -FilePath "curl.exe" `
            -ArgumentList "-s","-L","--retry","3","--retry-delay","5","-o",$localPath,$url `
            -NoNewWindow -PassThru
        $processes += @{ Process = $proc; Path = $relPath; LocalPath = $localPath }
    }

    # Wait for all processes in this batch to complete
    foreach ($entry in $processes) {
        $entry.Process.WaitForExit()
        if ($entry.Process.ExitCode -eq 0 -and (Test-Path $entry.LocalPath) -and ((Get-Item $entry.LocalPath).Length -gt 0)) {
            $completed++
        } else {
            $failed += $entry.Path
            Log "  FAILED: $($entry.Path)"
        }
    }

    # Progress report every batch
    $pct = [math]::Round(($completed + $skipped) / $allFiles.Count * 100, 1)
    $currentPatient = ($batch[0] -split '/')[0]
    if (($i % ($maxParallel * 5)) -eq 0 -or $i + $maxParallel -ge $totalCount) {
        Log "  Progress: $($completed + $skipped)/$($allFiles.Count) ($pct%) - currently at $currentPatient"
    }
}

# Retry failed files once
if ($failed.Count -gt 0) {
    Log "Retrying $($failed.Count) failed files..."
    $stillFailed = @()
    foreach ($relPath in $failed) {
        $url = "$baseUrl/$relPath"
        $localPath = Join-Path $destRoot ($relPath -replace '/', '\')
        & curl.exe -s -L --retry 5 --retry-delay 10 -o $localPath $url 2>$null
        if ($LASTEXITCODE -eq 0 -and (Test-Path $localPath) -and ((Get-Item $localPath).Length -gt 0)) {
            $completed++
        } else {
            $stillFailed += $relPath
            Log "  STILL FAILED: $relPath"
        }
    }
    $failed = $stillFailed
}

# Final summary with per-patient breakdown
Log "=== Download Complete ==="
Log "Successfully downloaded: $completed"
Log "Previously existed (skipped): $skipped"
Log "Failed: $($failed.Count)"
if ($failed.Count -gt 0) {
    foreach ($f in $failed) { Log "  MISSING: $f" }
}

# Per-patient size breakdown
Log ""
Log "--- Per-Patient Summary ---"
foreach ($patient in $patients) {
    $patientDir = Join-Path $destRoot $patient
    if (Test-Path $patientDir) {
        $files = Get-ChildItem $patientDir -File -Recurse
        $edfCount = ($files | Where-Object { $_.Extension -eq ".edf" }).Count
        $sizeGB = [math]::Round(($files | Measure-Object -Property Length -Sum).Sum / 1GB, 2)
        Log "  $patient : $edfCount .edf files, ${sizeGB} GB"
    }
}

$totalSize = (Get-ChildItem $destRoot -File -Recurse | Measure-Object -Property Length -Sum).Sum
Log ""
Log "Total dataset size: $([math]::Round($totalSize / 1GB, 2)) GB"
Log "========================="
