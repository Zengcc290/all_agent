<#
.SYNOPSIS
  一键停止所有服务（Qdrant / Neo4j / 假LLM / FastAPI 后端 / Vite 前端）。
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Stop-ByPort([int]$Port, [string]$Name) {
    $conns = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -eq $Port }
    if (-not $conns) {
        Write-Host ("  {0,-10} 未在运行" -f $Name) -ForegroundColor DarkGray
        return
    }
    $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $pids) {
        try {
            $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
            if ($p) {
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                Write-Host ("  {0,-10} 已停止 (PID {1} {2})" -f $Name, $procId, $p.ProcessName) -ForegroundColor Green
            }
        } catch {
            Write-Host ("  {0,-10} 停止失败: {1}" -f $Name, $_.Exception.Message) -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  停止所有服务" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# 先停前端和后端（它们可能持有 qdrant/neo4j 连接）
Stop-ByPort 5173 "前端"
Stop-ByPort 8000 "后端"
Stop-ByPort 8890 "假LLM"
Start-Sleep -Milliseconds 500
Stop-ByPort 7687 "Neo4j"
Stop-ByPort 6333 "Qdrant"

# node 有时不会随端口一起退出
Get-Process node -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "*$ProjectRoot*" -or $_.MainWindowTitle -like "*vite*" } |
    ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }

Write-Host ""
Write-Host "  全部已停止" -ForegroundColor Green
Write-Host ""
