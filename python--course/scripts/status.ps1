<#
.SYNOPSIS
  查看所有服务的运行状态，并跑一遍连通性自检。
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
if (-not (Test-Path $Py)) { $Py = (Get-Command python -ErrorAction SilentlyContinue).Source }

function Test-Port([int]$Port) {
    (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object LocalPort -eq $Port) -ne $null
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  服务状态" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

$svc = @(
    @{ N = "Qdrant";       P = 6333; U = "http://127.0.0.1:6333/dashboard"; Desc = "向量数据库" },
    @{ N = "Neo4j HTTP";   P = 7474; U = "http://127.0.0.1:7474/browser/"; Desc = "图数据库管理台" },
    @{ N = "Neo4j Bolt";   P = 7687; U = "bolt://127.0.0.1:7687";         Desc = "图数据库驱动" },
    @{ N = "FastAPI";      P = 8000; U = "http://127.0.0.1:8000/docs";    Desc = "后端" },
    @{ N = "Vite 前端";    P = 5173; U = "http://127.0.0.1:5173";         Desc = "前端" },
    @{ N = "假 LLM(演示)"; P = 8890; U = "-";                             Desc = "离线模式才需要" }
)

foreach ($s in $svc) {
    $up = Test-Port $s.P
    $mark = if ($up) { "[UP]" } else { "[--]" }
    $color = if ($up) { "Green" } else { "DarkGray" }
    Write-Host ("  {0} {1,-14} :{2,-6} {3}" -f $mark, $s.N, $s.P, $s.U) -ForegroundColor $color
}

Write-Host ""
Write-Host "------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  依赖连通性自检（用 app 自己的客户端连真服务）" -ForegroundColor DarkGray
Write-Host "------------------------------------------------------" -ForegroundColor DarkGray

if ($Py -and (Test-Path (Join-Path $ProjectRoot "_verify_native_services.py"))) {
    $env:PYTHONIOENCODING = "utf-8"
    Push-Location $ProjectRoot
    try {
        & $Py "_verify_native_services.py" 2>&1 |
            Where-Object { $_ -notmatch "notification|GqlStatus|WARN" } |
            ForEach-Object { Write-Host "  $_" }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "  未找到 _verify_native_services.py 或 python" -ForegroundColor Yellow
}

Write-Host ""
