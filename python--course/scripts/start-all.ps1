<#
.SYNOPSIS
  知识图谱入库系统 — 一键启动所有依赖服务与前后端。

.DESCRIPTION
  启动顺序（带端口就绪等待）:
    1. Qdrant   原生 Windows 版，C:\qdrant\qdrant.exe  -> 127.0.0.1:6333
    2. Neo4j    原生 Windows 版，C:\neo4j              -> 127.0.0.1:7687
    3. 离线演示 LLM（可选，仅在未配置真实 key 时启用）-> 127.0.0.1:8890
    4. FastAPI 后端                                    -> 127.0.0.1:8000
    5. Vite 前端                                      -> 127.0.0.1:5173

.PARAMETER Demo
  强制启用离线演示模式（用 _fake_llm_server.py 当 LLM/Embedding），
  即使 .env 里配了真实 key 也不用它。适合没有 API key 时先看效果。

.PARAMETER SkipFrontend
  只起后端与依赖服务，不起前端。

.PARAMETER SkipServices
  不碰 qdrant / neo4j（已经起好了的时候用）。

.EXAMPLE
  .\scripts\start-all.ps1
.EXAMPLE
  .\scripts\start-all.ps1 -Demo
#>
[CmdletBinding()]
param(
    [switch]$Demo,
    [switch]$SkipFrontend,
    [switch]$SkipServices
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
if (-not (Test-Path $Py)) { $Py = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $Py) { Write-Host "找不到 python.exe，请先安装 Python 3.12" -ForegroundColor Red; exit 1 }

$Jdk = "C:\Program Files\Microsoft\jdk-17.0.20.101-hotspot"
if (-not (Test-Path "$Jdk\bin\java.exe")) {
    $j = Get-Command java -ErrorAction SilentlyContinue
    if ($j) { $Jdk = Split-Path (Split-Path $j.Source -Parent) -Parent }
}

# ---------------- 端口工具 ----------------
function Test-Port([int]$Port) {
    (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object LocalPort -eq $Port) -ne $null
}

function Wait-Port([int]$Port, [int]$TimeoutSec, [string]$Name) {
    for ($i = 0; $i -lt $TimeoutSec; $i++) {
        if (Test-Port $Port) { Write-Host "  OK   $Name 已就绪 (:$Port)" -ForegroundColor Green; return $true }
        Start-Sleep -Seconds 1
    }
    Write-Host "  FAIL $Name 等待超时 (:$Port)" -ForegroundColor Red
    return $false
}

function Start-Detached([string]$File, [string[]]$ArgumentList, [string]$Dir, [string]$LogName) {
    $log = Join-Path $Dir "$LogName.log"
    $err = Join-Path $Dir "$LogName.err"
    # 注意：参数名不能用 $Args，它是 PowerShell 的自动变量，会被静默吃掉导致参数丢失
    # -ArgumentList 也不接受空集合
    if ($ArgumentList -and $ArgumentList.Count -gt 0) {
        $p = Start-Process -FilePath $File -ArgumentList $ArgumentList -WorkingDirectory $Dir `
            -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
    } else {
        $p = Start-Process -FilePath $File -WorkingDirectory $Dir `
            -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
    }
    return $p
}

# 停掉某个端口上的陈旧进程（保证新进程用的是最新配置）
function Restart-Stale([int]$Port, [string]$Name) {
    $conns = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -eq $Port }
    if (-not $conns) { return $false }
    Write-Host ("  ..   {0} 已有进程占用 :{1}，先停掉它以保证使用最新配置" -f $Name, $Port) -ForegroundColor Yellow
    $conns | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
        Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
    return $true
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  知识图谱入库系统 — 一键启动" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "项目目录: $ProjectRoot"
Write-Host "Python  : $Py"
Write-Host ""

# ---------------- 0. 配置 ----------------
$envFile = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $ProjectRoot ".example.env") $envFile
    Write-Host "已从 .example.env 生成 .env" -ForegroundColor Yellow
}

# 读一下 .env，判断要不要走离线演示模式
$cfg = @{}
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([A-Z0-9_]+)\s*=\s*(.*)$') { $cfg[$Matches[1]] = $Matches[2].Trim() }
}

# 占位 key 判定：空、或含一串 x（模板里的 sk-xxxx...）、或以 sk-xxx 开头
function Test-PlaceholderKey([string]$K) {
    if ([string]::IsNullOrWhiteSpace($K)) { return $true }
    if ($K -match 'x{4,}') { return $true }
    if ($K -eq 'sk-xxx' -or $K -eq 'xxx') { return $true }
    return $false
}

$needKey = Test-PlaceholderKey $cfg['LLM_API_KEY']
$embKey = Test-PlaceholderKey $cfg['EMBEDDING_API_KEY']
$DemoMode = $Demo -or $needKey -or $embKey

if ($DemoMode) {
    Write-Host ">> 模式: 离线演示（未检测到真实 API key，用本地假 LLM，可完整跑通全流程）" -ForegroundColor Magenta
    Write-Host "   想用真实模型：编辑 .env 填入 LLM_API_KEY / EMBEDDING_API_KEY 后重跑" -ForegroundColor DarkGray
} else {
    Write-Host ">> 模式: 真实 LLM（使用 .env 中配置的 SiliconFlow key）" -ForegroundColor Magenta
}

# ---------------- 1. Qdrant ----------------
if (-not $SkipServices) {
    Write-Host "`n[1/5] Qdrant 向量数据库" -ForegroundColor Cyan
    if (Test-Port 6333) {
        Write-Host "  -    已在运行 (127.0.0.1:6333)" -ForegroundColor DarkGray
    } elseif (Test-Path "C:\qdrant\qdrant.exe") {
        $env:QDRANT__SERVICE__HOST = "127.0.0.1"
        $env:QDRANT__SERVICE__HTTP_PORT = "6333"
        $env:QDRANT__SERVICE__GRPC_PORT = "6334"
        $env:QDRANT__STORAGE__STORAGE_PATH = "C:\qdrant\storage"
        $env:QDRANT__TELEMETRY_DISABLED = "true"
        $null = Start-Detached "C:\qdrant\qdrant.exe" @() "C:\qdrant" "qdrant"
        Wait-Port 6333 30 "Qdrant" | Out-Null
    } else {
        Write-Host "  FAIL 未找到 C:\qdrant\qdrant.exe" -ForegroundColor Red
    }

    # ---------------- 2. Neo4j ----------------
    Write-Host "`n[2/5] Neo4j 图数据库" -ForegroundColor Cyan
    if (Test-Port 7687) {
        Write-Host "  -    已在运行 (127.0.0.1:7687)" -ForegroundColor DarkGray
    } elseif (Test-Path "C:\neo4j\lib") {
        if (-not (Test-Path "$Jdk\bin\java.exe")) {
            Write-Host "  FAIL 未找到 java（JDK17 未安装）" -ForegroundColor Red
        } else {
            $env:JAVA_HOME = $Jdk
            $env:NEO4J_HOME = "C:\neo4j"
            Remove-Item "C:\neo4j\logs\neo4j-console.log","C:\neo4j\logs\neo4j-console.err" -Force -ErrorAction SilentlyContinue
            $null = Start-Detached "$Jdk\bin\java.exe" @(
                "-cp", "C:\neo4j\lib\*",
                "-Dbasedir=C:\neo4j",
                "-Dneo4j.home=C:\neo4j",
                "org.neo4j.server.startup.Neo4jCommand", "console"
            ) "C:\neo4j" "neo4j-console"
            if (-not (Wait-Port 7687 90 "Neo4j")) {
                Write-Host "  ---- neo4j 启动日志 ----" -ForegroundColor Red
                Get-Content "C:\neo4j\logs\neo4j-console.err" -Tail 15 -ErrorAction SilentlyContinue |
                    ForEach-Object { Write-Host "  $_" }
            }
        }
    } else {
        Write-Host "  FAIL 未找到 C:\neo4j\lib" -ForegroundColor Red
    }
} else {
    Write-Host "`n[1-2/5] 已跳过依赖服务（-SkipServices）" -ForegroundColor DarkGray
}

# ---------------- 3. 离线演示 LLM ----------------
$fakePid = $null
if ($DemoMode) {
    Write-Host "`n[3/5] 离线演示 LLM / Embedding" -ForegroundColor Cyan
    if (Test-Port 8890) {
        Write-Host "  -    已在运行 (127.0.0.1:8890)" -ForegroundColor DarkGray
    } else {
        # 上次可能是坏的 REPL 进程占着 8890，先清掉
        Restart-Stale 8890 "假 LLM" | Out-Null
        $fakePid = Start-Detached $Py @("_fake_llm_server.py", "8890") $ProjectRoot "fake-llm"
        Wait-Port 8890 20 "假 LLM" | Out-Null
    }
} else {
    Write-Host "`n[3/5] 已跳过离线演示 LLM（使用真实 key）" -ForegroundColor DarkGray
}

# ---------------- 4. 后端 ----------------
Write-Host "`n[4/5] FastAPI 后端" -ForegroundColor Cyan
$env:APP_HOST = "127.0.0.1"
$env:APP_PORT = "8000"
$env:PYTHONIOENCODING = "utf-8"
# 让 python 能 import 到项目根目录
$env:PYTHONPATH = $ProjectRoot
if ($DemoMode) {
    $env:LLM_BASE_URL = "http://127.0.0.1:8890/v1"
    $env:EMBEDDING_BASE_URL = "http://127.0.0.1:8890/v1"
    $env:LLM_API_KEY = "sk-demo"
    $env:EMBEDDING_API_KEY = "sk-demo"
    # 假 LLM 产出 8 维向量；不改会和 qdrant 里已建的 8 维集合冲突
    $env:EMBEDDING_DIM = "8"
    $env:LLM_MODEL = "demo-llm"
    $env:EMBEDDING_MODEL = "demo-embed"
}
if (Test-Port 8000) {
    # 已有后端：它的配置可能是旧的，重启以保证用的是本次 .env
    Restart-Stale 8000 "后端"
}
$null = Start-Detached $Py @("run.py") $ProjectRoot "backend"
Wait-Port 8000 30 "后端" | Out-Null

# ---------------- 5. 前端 ----------------
$feUrl = $null
if (-not $SkipFrontend) {
    Write-Host "`n[5/5] 前端 (Vite)" -ForegroundColor Cyan
    if (Test-Port 5173) {
        Write-Host "  -    已在运行 (127.0.0.1:5173)" -ForegroundColor DarkGray
    } elseif (Test-Path (Join-Path $ProjectRoot "frontend\node_modules")) {
        $npm = "npm.cmd"
        $feLog = Join-Path $ProjectRoot "frontend\vite.log"
        $p = Start-Process -FilePath $npm -ArgumentList @("run", "dev") `
            -WorkingDirectory (Join-Path $ProjectRoot "frontend") `
            -WindowStyle Hidden -RedirectStandardOutput $feLog `
            -RedirectStandardError (Join-Path $ProjectRoot "frontend\vite.err") -PassThru
        Wait-Port 5173 40 "前端" | Out-Null
    } else {
        Write-Host "  WARN 未安装前端依赖，先执行: cd frontend; npm install" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n[5/5] 已跳过前端（-SkipFrontend）" -ForegroundColor DarkGray
}

# ---------------- 汇总 ----------------
Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  启动完成" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

$svc = @(
    @{ N = "Qdrant";  P = 6333; U = "http://127.0.0.1:6333/dashboard" },
    @{ N = "Neo4j";   P = 7687; U = "http://127.0.0.1:7474/browser/" },
    @{ N = "后端 API"; P = 8000; U = "http://127.0.0.1:8000/docs" }
)
if ($DemoMode) { $svc += @{ N = "假 LLM"; P = 8890; U = "-" } }
if (-not $SkipFrontend) { $svc += @{ N = "前端"; P = 5173; U = "http://127.0.0.1:5173" } }

foreach ($s in $svc) {
    $up = Test-Port $s.P
    $mark = if ($up) { "[UP]" } else { "[--]" }
    $color = if ($up) { "Green" } else { "DarkGray" }
    Write-Host ("  {0} {1,-10} :{2,-5} {3}" -f $mark, $s.N, $s.P, $s.U) -ForegroundColor $color
}

# 探一下后端健康
if (Test-Port 8000) {
    try {
        $h = Invoke-RestMethod "http://127.0.0.1:8000/api/health" -TimeoutSec 8
        if ($h.ok) { Write-Host "`n  后端健康检查通过 (kg-ingest)" -ForegroundColor Green }
    } catch { Write-Host "`n  后端健康检查未通过：$($_.Exception.Message)" -ForegroundColor Yellow }
}

Write-Host ""
Write-Host "  下一步：浏览器打开 http://127.0.0.1:5173" -ForegroundColor Yellow
Write-Host "  → 「✨ 一句话入库」输入一句话，即可看到全链路过程"
Write-Host "  → 「🪐 实体星球」查看实体与关系引力图"
Write-Host "  → 「🧬 多路混合检索」体验向量 + FTS5 + RRF"
Write-Host ""
Write-Host "  停止全部：.\scripts\stop-all.ps1" -ForegroundColor DarkGray
Write-Host ""
