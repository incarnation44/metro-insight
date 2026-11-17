#Requires -Version 5.1
<#
.SYNOPSIS
  이관 ZIP을 임시 폴더에 풀어 '복원 훈련'을 돌린다.

.DESCRIPTION
  새 기기에서 압축을 풀었을 때 바로 동작하는지 이 기기에서 미리 검사한다.
  1) ZIP 해제 → 2) git status 청결 → 3) 필수 비추적 파일 존재
  4) pytest / ruff / verify_manifest / 오프스크린 기동
  각 항목의 PASS·FAIL과 실제 출력 마지막 줄을 요약해 보여준다.

.PARAMETER ZipPath
  검사할 ZIP 경로.

.PARAMETER WorkDir
  압축을 풀 임시 폴더. 기본값은 `%TEMP%\metro-insight_restore`.

.PARAMETER PythonExe
  검사에 쓸 python 실행 파일. 기본값은 conda `metro-insight` 환경을 자동 탐색.

.PARAMETER Keep
  검사 후에도 푼 폴더를 남긴다(기본은 삭제).

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_bundle.ps1 -ZipPath "C:\Users\me\Desktop\metroinsight 자료\metro-insight_전체프로젝트_2026-11-16.zip"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ZipPath,
    [string]$WorkDir = (Join-Path $env:TEMP 'metro-insight_restore'),
    [string]$PythonExe,
    [switch]$Keep,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

if (-not (Test-Path -LiteralPath $ZipPath)) { throw "ZIP이 없다: $ZipPath" }
$ZipPath = (Resolve-Path -LiteralPath $ZipPath).Path

# --- python 탐색 ---
if (-not $PythonExe) {
    $probe = @(
        (Join-Path $env:USERPROFILE 'anaconda3\envs\metro-insight\python.exe'),
        (Join-Path $env:USERPROFILE 'miniconda3\envs\metro-insight\python.exe'),
        'C:\ProgramData\anaconda3\envs\metro-insight\python.exe'
    )
    foreach ($p in $probe) {
        if (Test-Path -LiteralPath $p) { $PythonExe = $p; break }
    }
    if (-not $PythonExe) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) { $PythonExe = $cmd.Source }
    }
}
if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) {
    throw "python을 찾지 못했다. -PythonExe로 지정한다."
}
Write-Host "python: $PythonExe"

$results = New-Object System.Collections.Generic.List[object]
function Add-Result {
    param([string]$Name, [bool]$Pass, [string]$Detail)
    $results.Add([pscustomobject]@{ 항목 = $Name; 결과 = $(if ($Pass) { 'PASS' } else { 'FAIL' }); 근거 = $Detail })
}

# --- 1) 해제 ---
Write-Host "[1/5] ZIP 해제 → $WorkDir"
if (Test-Path -LiteralPath $WorkDir) { [IO.Directory]::Delete($WorkDir, $true) }
New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null
[IO.Compression.ZipFile]::ExtractToDirectory($ZipPath, $WorkDir)

$repoRoot = Join-Path $WorkDir 'metro-insight'
if (-not (Test-Path -LiteralPath $repoRoot)) {
    $inner = Get-ChildItem -LiteralPath $WorkDir -Directory | Select-Object -First 1
    if ($inner) { $repoRoot = $inner.FullName } else { throw "ZIP 안에 폴더가 없다: $ZipPath" }
}
$fileCount = (Get-ChildItem -LiteralPath $repoRoot -Recurse -File -Force).Count
Add-Result '압축 해제' $true ("$fileCount 개 파일")

# --- 2) git 상태 ---
Write-Host '[2/5] git status 검사'
$gitStatus = & git -C $repoRoot status --porcelain 2>&1
Add-Result 'git status 청결' ([string]::IsNullOrWhiteSpace(($gitStatus -join ''))) (($gitStatus | Select-Object -First 5) -join ' / ')
$head = (& git -C $repoRoot rev-parse --short HEAD 2>&1) | Select-Object -First 1
Write-Host "      HEAD = $head"

# --- 3) 필수 비추적 파일 ---
Write-Host '[3/5] 필수 산출물 존재 검사'
$required = @(
    'data/raw/subway/subway_2024_sample.csv',
    'data/processed/train.csv.gz',
    'data/processed/valid.csv.gz',
    'data/processed/test.csv.gz',
    'models/cal_A.joblib',
    'models/cal_B.joblib',
    'models/cal_C.joblib',
    'models/sequence_D.pt',
    'models/sequence_E.pt',
    'models/cal_profiles.npz',
    'models/cal_profiles_eval.npz',
    'models/boarding_history.npz'
)
$missing = @()
foreach ($rel in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $rel))) { $missing += $rel }
}
$rawCsv = @(Get-ChildItem -LiteralPath (Join-Path $repoRoot 'data/raw/subway') -Filter '*.csv' -File -Force | Where-Object { $_.Name -notlike '*_sample.csv' })
if ($rawCsv.Count -lt 3) { $missing += "data/raw/subway 원본 CSV ($($rawCsv.Count)/3)" }
$expDirs = @(Get-ChildItem -LiteralPath (Join-Path $repoRoot 'models/_exp') -Directory -Force -ErrorAction SilentlyContinue)
if ($expDirs.Count -lt 9) { $missing += "models/_exp 실험 디렉터리 ($($expDirs.Count)/9)" }
Add-Result '필수 산출물' ($missing.Count -eq 0) $(if ($missing.Count -eq 0) { "12개 + 원본 CSV $($rawCsv.Count)개 + 실험 $($expDirs.Count)개" } else { $missing -join ', ' })

# --- 4) 검증 명령 ---
if (-not $SkipTests) {
    $env:PYTHONPATH = Join-Path $repoRoot 'src'
    $env:QT_QPA_PLATFORM = 'offscreen'

    function Invoke-Check {
        param([string]$Name, [string[]]$Args, [string]$ExpectPattern)
        Write-Host "      $Name ..."
        # PowerShell 5.1은 네이티브 프로그램의 2>&1 출력을 ErrorRecord로 승격한다.
        # ErrorActionPreference=Stop 이면 그 순간 죽으므로 호출 구간만 Continue로 내린다.
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $out = & $PythonExe @Args 2>&1
        } finally {
            $ErrorActionPreference = $prev
        }
        $tail = ($out | Where-Object { $_ -ne '' } | Select-Object -Last 3) -join ' | '
        $ok = ($out -join "`n") -match $ExpectPattern
        Add-Result $Name $ok $tail
    }

    Write-Host '[4/5] pytest / ruff / verify_manifest / 기동'
    Push-Location -LiteralPath $repoRoot
    try {
        Invoke-Check 'pytest' @('-m', 'pytest', '-q') 'passed'
        Invoke-Check 'ruff' @('-m', 'ruff', 'check', 'src', 'tests') 'All checks passed'
        Invoke-Check 'verify_manifest' @('notebooks/verify_manifest.py') '전부 통과'
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $smoke = & $PythonExe -c "from PySide6.QtWidgets import QApplication; from metro_insight.app.main_window import MainWindow; app = QApplication([]); w = MainWindow(); print('TITLE:' + w.windowTitle())" 2>&1
        } finally {
            $ErrorActionPreference = $prev
        }
        Add-Result '오프스크린 기동' (($smoke -join "`n") -match 'TITLE:') (($smoke | Select-Object -Last 2) -join ' | ')
    } finally {
        Pop-Location
    }
} else {
    Write-Host '[4/5] 검증 명령 생략(-SkipTests)'
}

# --- 5) 요약 ---
Write-Host '[5/5] 요약'
$lines = $results | ForEach-Object { '{0,-18} {1,-5} {2}' -f $_.항목, $_.결과, $_.근거 }
$lines | ForEach-Object { Write-Host "  $_" }
$failed = @($results | Where-Object 결과 -eq 'FAIL')
if ($failed.Count -eq 0) {
    Write-Host '전체 PASS — 복원 훈련 통과'
} else {
    Write-Host ("FAIL {0}건: {1}" -f $failed.Count, ($failed.항목 -join ', '))
}

if (-not $Keep) {
    Write-Host '임시 폴더 삭제'
    [IO.Directory]::Delete($WorkDir, $true)
} else {
    Write-Host "임시 폴더 유지: $WorkDir"
}

exit $(if ($failed.Count -eq 0) { 0 } else { 1 })