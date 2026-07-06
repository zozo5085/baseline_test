# run_experiment_queue.ps1 - sequential experiment queue runner for D:\ReCLIPP_Test
# PowerShell 5.1 compatible. ASCII only in this file.
#
# Behavior (per docs/EXPERIMENT_QUEUE.md):
#  - NEVER deletes checkpoints or logs. All logs get timestamped filenames.
#  - Waits for any currently running ML python process (train.py/test.py) to finish
#    before starting the queue, polling every 5 minutes.
#  - Runs experiments strictly in order. After each: parse last "the mIOU:" value.
#  - STOPS the whole queue and records the reason in docs/EXPERIMENT_STATUS.md when:
#      * Traceback / CUDA OOM / RuntimeError found in the experiment log
#      * non-zero exit code
#      * a test produces no parsable mIoU
#      * a test mIoU < $StopBelow (clearly below baseline 0.8451)
#      * a train target SAVE_DIR already contains best_weight.pth (overwrite protection)
#
# Usage (from any terminal):
#   powershell -ExecutionPolicy Bypass -File D:\ReCLIPP_Test\tools\run_experiment_queue.ps1

$ErrorActionPreference = 'Continue'
$Root = 'D:\ReCLIPP_Test'
$Py = 'C:\Users\NUTC2507\miniconda3\envs\reclip5090\python.exe'
$StatusFile = Join-Path $Root 'docs\EXPERIMENT_STATUS.md'
$LogDir = Join-Path $Root 'experiments\queue_logs'
$BaselineMiou = 0.8451
$StopBelow = 0.80
$PollSeconds = 300

# ---- queue definition (keep in sync with docs/EXPERIMENT_QUEUE.md) ----
$Experiments = @(
    @{ Id = 'E01'; Kind = 'test';
       Cfg = 'config\voc_test_l9l12_selective_v2_cfg.yaml';
       ModelModule = 'model.model_feature_fusion';
       Requires = 'experiments\voc_l9l12_selective_v2\best_weight.pth';
       LogPrefix = 'E01_test_l9l12_v2' },
    @{ Id = 'E02'; Kind = 'train';
       Cfg = 'config\voc_train_l6l12_selective_v2_cfg.yaml';
       ModelModule = 'model.model_feature_fusion';
       ProtectDir = 'experiments\voc_l6l12_selective_v2';
       LogPrefix = 'E02_train_l6l12_v2' },
    @{ Id = 'E03'; Kind = 'test';
       Cfg = 'config\voc_test_l6l12_selective_v2_cfg.yaml';
       ModelModule = 'model.model_feature_fusion';
       Requires = 'experiments\voc_l6l12_selective_v2\best_weight.pth';
       LogPrefix = 'E03_test_l6l12_v2' }
)
# -----------------------------------------------------------------------

function Write-Status([string]$line) {
    Add-Content -Path $StatusFile -Value $line -Encoding UTF8
    Write-Host $line
}

function Get-RunningMLProc {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
        $_.CommandLine -match 'train\.py|test\.py'
    }
}

function Get-LastMiou([string]$logPath) {
    if (-not (Test-Path $logPath)) { return $null }
    $m = Select-String -Path $logPath -Pattern 'the mIOU:([0-9.]+)' | Select-Object -Last 1
    if ($null -eq $m) { return $null }
    return [double]$m.Matches[0].Groups[1].Value
}

function Test-LogHasError([string[]]$logPaths) {
    foreach ($p in $logPaths) {
        if (-not (Test-Path $p)) { continue }
        $hit = Select-String -Path $p -Pattern 'Traceback|CUDA out of memory|RuntimeError' | Select-Object -First 1
        if ($null -ne $hit) { return $hit.Line }
    }
    return $null
}

Set-Location $Root
New-Item -ItemType Directory -Force $LogDir | Out-Null

Write-Status ''
Write-Status ('## Queue run started {0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
$gpuMem = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader) -join ' '
Write-Status ('- GPU memory at start: {0}' -f $gpuMem)

# ---- wait for any currently running experiment ----
$announced = $false
while ($true) {
    $procs = @(Get-RunningMLProc)
    if ($procs.Count -eq 0) { break }
    if (-not $announced) {
        Write-Status ('- waiting for running ML process(es) to finish: pid {0}' -f ($procs.ProcessId -join ','))
        $announced = $true
    }
    Write-Host ('[{0}] still waiting ({1} process(es))...' -f (Get-Date -Format 'HH:mm:ss'), $procs.Count)
    Start-Sleep -Seconds $PollSeconds
}
Write-Status ('- {0}: no running ML process, queue begins' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))

# ---- run the queue ----
$stopped = $false
foreach ($e in $Experiments) {
    if ($stopped) { break }
    $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
    $log = Join-Path $LogDir ('{0}_{1}.log' -f $e.LogPrefix, $ts)
    $errLog = Join-Path $LogDir ('{0}_{1}.err.log' -f $e.LogPrefix, $ts)

    # precondition: required input exists
    if ($e.Requires) {
        $req = Join-Path $Root $e.Requires
        if (-not (Test-Path $req)) {
            Write-Status ('- **{0} SKIPPED / QUEUE STOPPED**: required file missing: {1}' -f $e.Id, $e.Requires)
            $stopped = $true; break
        }
    }
    # precondition: never overwrite an existing checkpoint
    if ($e.ProtectDir) {
        $bw = Join-Path $Root (Join-Path $e.ProtectDir 'best_weight.pth')
        if (Test-Path $bw) {
            Write-Status ('- **{0} SKIPPED / QUEUE STOPPED**: {1} already exists (checkpoints are never overwritten)' -f $e.Id, $bw)
            $stopped = $true; break
        }
    }

    $scriptRel = 'tools\test.py'
    if ($e.Kind -eq 'train') { $scriptRel = 'tools\train.py' }
    Write-Status ('- {0} started {1}; log: {2}' -f $e.Id, (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $log)

    $procArgs = @($scriptRel, '--cfg', $e.Cfg, '--model', 'RECLIPPP', '--model_module', $e.ModelModule)
    $p = Start-Process -FilePath $Py -ArgumentList $procArgs -WorkingDirectory $Root `
        -RedirectStandardOutput $log -RedirectStandardError $errLog -NoNewWindow -Wait -PassThru
    $exit = $p.ExitCode

    $errLine = Test-LogHasError @($log, $errLog)
    if ($null -ne $errLine) {
        Write-Status ('- **{0} FAILED / QUEUE STOPPED**: error in log: {1}' -f $e.Id, $errLine)
        $stopped = $true; break
    }
    if ($exit -ne 0) {
        Write-Status ('- **{0} FAILED / QUEUE STOPPED**: exit code {1}' -f $e.Id, $exit)
        $stopped = $true; break
    }

    $miou = Get-LastMiou $log
    if ($e.Kind -eq 'test') {
        if ($null -eq $miou) {
            Write-Status ('- **{0} FAILED / QUEUE STOPPED**: no mIoU parsed from {1}' -f $e.Id, $log)
            $stopped = $true; break
        }
        $delta = [math]::Round($miou - $BaselineMiou, 4)
        Write-Status ('- {0} DONE {1}: mIoU {2} (baseline {3}, delta {4})' -f $e.Id, (Get-Date -Format 'HH:mm:ss'), $miou, $BaselineMiou, $delta)
        if ($miou -lt $StopBelow) {
            Write-Status ('- **QUEUE STOPPED after {0}**: mIoU {1} is clearly below baseline (< {2}); diagnose before continuing' -f $e.Id, $miou, $StopBelow)
            $stopped = $true; break
        }
    }
    else {
        # train: verify checkpoint was produced
        $bw = Join-Path $Root (Join-Path $e.ProtectDir 'best_weight.pth')
        if (-not (Test-Path $bw)) {
            Write-Status ('- **{0} FAILED / QUEUE STOPPED**: training ended but {1} was not produced' -f $e.Id, $bw)
            $stopped = $true; break
        }
        if ($null -ne $miou) {
            Write-Status ('- {0} DONE {1}: best_weight.pth produced; last in-training eval mIoU {2} (informal number)' -f $e.Id, (Get-Date -Format 'HH:mm:ss'), $miou)
        } else {
            Write-Status ('- {0} DONE {1}: best_weight.pth produced (no in-training eval found in console log)' -f $e.Id, (Get-Date -Format 'HH:mm:ss'))
        }
    }
}

Write-Status ('## Queue run ended {0} (stopped early: {1})' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $stopped)
