#Requires -Version 5.1
<#
.SYNOPSIS
  metro-insight 이관(ZIP) 패키지를 만든다.

.DESCRIPTION
  Git 저장소(`.git` 포함) · 커밋된 파일 · `.gitignore` 대상이지만 반드시 필요한 산출물을
  하나의 ZIP으로 묶는다. 새 기기에서 압축만 풀면 `git status`가 깨끗한 상태로 바로 실행된다.

  포함 : 소스·테스트·문서, data/raw/subway 원본 CSV 3개, data/raw/weather,
         data/processed/{train,valid,test}.csv.gz, models/*.joblib·*.pt·*.npz, models/_exp/ 9개 실험
  제외 : .env.production(기상청 인증키), .env, 캐시(__pycache__ 등), .vscode, reports, var, *.egg-info

.PARAMETER RepoRoot
  저장소 루트. 기본값은 이 스크립트의 부모 폴더.

.PARAMETER OutDir
  ZIP을 만들 위치. 기본값은 `~/Desktop/metroinsight 자료`.

.PARAMETER Stamp
  파일명에 붙는 날짜. 기본값은 오늘(yyyy-MM-dd).

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_bundle.ps1

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_bundle.ps1 -OutDir D:\배포 -Stamp 2026-11-16
#>
[CmdletBinding()]
param(
    [string]$RepoRoot,
    [string]$OutDir = (Join-Path $env:USERPROFILE 'Desktop\metroinsight 자료'),
    [string]$Stamp = (Get-Date -Format 'yyyy-MM-dd'),
    [switch]$NoGit
)

$ErrorActionPreference = 'Stop'
if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $PSScriptRoot }
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\', '/')
if (-not (Test-Path -LiteralPath $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

# --- 제외 규칙 ---
# 폴더 이름 단위로 통째로 빼는 것들
$skipDirs = @('__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.vscode', '.idea', '.venv', 'venv', 'reports', 'var', 'node_modules', '.ipynb_checkpoints')
# 파일 이름 단위로 빼는 것들 (비밀·캐시)
$skipFiles = @('.env', '.env.production', '.env.local', 'Thumbs.db', '.DS_Store')
# 확장자 단위로 빼는 것들
$skipExt = @('.pyc', '.pyo', '.log', '.tmp', '.swp')

# 이미 압축된 형식은 다시 압축하지 않는다(속도)
$storeExt = @('.gz', '.zip', '.png', '.jpg', '.jpeg', '.pt', '.joblib', '.npz', '.npy', '.7z')

$zipName = "metro-insight_전체프로젝트_$Stamp.zip"
$zipPath = Join-Path $OutDir $zipName
if (Test-Path -LiteralPath $zipPath) {
    [IO.File]::Delete($zipPath)
}

function Test-SkipPath {
    param([string]$RelativePath)
    $segments = $RelativePath -split '[\\/]'
    for ($i = 0; $i -lt $segments.Count - 1; $i++) {
        $seg = $segments[$i]
        if ($skipDirs -contains $seg) { return $true }
        if ($seg -like '*.egg-info') { return $true }
    }
    $leaf = $segments[-1]
    if ($skipFiles -contains $leaf) { return $true }
    if ($skipExt -contains ([IO.Path]::GetExtension($leaf).ToLowerInvariant())) { return $true }
    return $false
}

Write-Host "[1/4] 파일 수집 중: $RepoRoot"
$candidates = Get-ChildItem -LiteralPath $RepoRoot -Recurse -File -Force
$selected = New-Object System.Collections.Generic.List[string]
$skipped = 0
$rootPrefix = $RepoRoot + [IO.Path]::DirectorySeparatorChar

foreach ($file in $candidates) {
    $relative = $file.FullName.Substring($rootPrefix.Length)
    if (-not $NoGit -and $relative -like '.git\*') {
        # .git 은 통째로 포함하되 캐시성 파일만 뺀다
        if ($relative -match '\.git[\\/](lfs[\\/]|gc\.log$|COMMIT_EDITMSG$)') { $skipped++; continue }
    }
    if (Test-SkipPath -RelativePath $relative) { $skipped++; continue }
    $selected.Add($file.FullName)
}

Write-Host ("      포함 {0:N0}개 / 제외 {1:N0}개" -f $selected.Count, $skipped)

Write-Host "[2/4] ZIP 생성: $zipPath"
$archive = [IO.Compression.ZipFile]::Open($zipPath, [IO.Compression.ZipArchiveMode]::Create)
try {
    $done = 0
    foreach ($fullPath in $selected) {
        $relative = $fullPath.Substring($rootPrefix.Length)
        $entryName = 'metro-insight/' + ($relative -replace '\\', '/')
        $ext = [IO.Path]::GetExtension($fullPath).ToLowerInvariant()
        $level = if ($storeExt -contains $ext) { [IO.Compression.CompressionLevel]::NoCompression } else { [IO.Compression.CompressionLevel]::Optimal }
        [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $fullPath, $entryName, $level)
        $done++
        if ($done % 100 -eq 0) { Write-Host ("      ... {0:N0}/{1:N0}" -f $done, $selected.Count) }
    }
} finally {
    $archive.Dispose()
}

$zipInfo = Get-Item -LiteralPath $zipPath
Write-Host ("      완료: {0:N0} MB" -f ($zipInfo.Length / 1MB))

Write-Host "[3/4] 압축 무결성 검사"
$check = [IO.Compression.ZipFile]::OpenRead($zipPath)
try { $entryCount = $check.Entries.Count } finally { $check.Dispose() }
Write-Host ("      엔트리 {0:N0}개 정상" -f $entryCount)

Write-Host "[4/4] SHA-256 기록"
$hash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash
$shaFile = Join-Path $OutDir "metro-insight_전체프로젝트_$Stamp.sha256.txt"
"$hash  $zipName" | Set-Content -LiteralPath $shaFile -Encoding utf8
Write-Host "      $hash"

Write-Host ''
Write-Host '완료.'
Write-Host "  ZIP    : $zipPath"
Write-Host "  SHA256 : $shaFile"