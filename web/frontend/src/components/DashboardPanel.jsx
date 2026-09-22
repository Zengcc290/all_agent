import React, { useCallback, useState } from 'react'
import { api, isLockMismatch, lockMismatchText } from '../api.js'
import { errMessage } from '../util.js'

/** 监控台：运行健康度 / 库规模 / 三库对账与修复 / 重建向量 / 重新播种。 */
export default function DashboardPanel({ refreshKey, onDataChanged }) {
  const [health, setHealth] = useState(null)
  const [stats, setStats] = useState(null)
  const [reconcile, setReconcile] = useState(null)
  const [msg, setMsg] = useState('')
  const [busy, setBusy] = useState(false)

  const loadAll = useCallback(async () => {
    setMsg('')
    const tasks = [
      api.health().then(setHealth).catch((e) => setMsg(errMessage(e))),
      api.stats().then(setStats).catch((e) => setMsg(errMessage(e))),
      api.reconcile().then(setReconcile).catch(() => setReconcile(null)),
    ]
    await Promise.all(tasks)
  }, [])

  React.useEffect(() => { loadAll() }, [loadAll, refreshKey])

  const seedNow = useCallback(async () => {
    setBusy(true); setMsg('')
    try {
      const r = await api.seed()
      setMsg(`播种完成：${JSON.stringify(r)}`)
      onDataChanged && onDataChanged()
    } catch (e) {
      setMsg(errMessage(e))
    }
    setBusy(false)
  }, [onDataChanged])

  const repair = useCallback(async (kinds) => {
    setBusy(true); setMsg('')
    try {
      const r = await api.reconcileRepair(kinds)
      setMsg(`修复完成：${JSON.stringify(r)}`)
      loadAll()
      onDataChanged && onDataChanged()
    } catch (e) {
      setMsg(errMessage(e))
    }
    setBusy(false)
  }, [loadAll, onDataChanged])

  const rebuild = useCallback(async () => {
    if (!window.confirm('将按当前 embedding 配置重建整个向量投影（全量重灌），确认继续？')) return
    setBusy(true); setMsg('')
    try {
      const r = await api.rebuildEmbedding()
      setMsg(`重建完成：${JSON.stringify(r)}`)
      loadAll()
    } catch (e) {
      setMsg(isLockMismatch(e) ? lockMismatchText(e) : errMessage(e))
    }
    setBusy(false)
  }, [loadAll])

  const s = stats || {}
  const degraded = (health && health.degraded) || {}

  return (
    <>
      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div>
            <h2>运行健康度</h2>
            <p className="hint" style={{ margin: 0 }}>来自 /api/health：嵌入模式、聊天可用性、三库实现与降级原因。</p>
          </div>
          <div className="row tight">
            <button className="btn sm" onClick={loadAll} disabled={busy}>刷新</button>
            <button className="btn sm warn" onClick={rebuild} disabled={busy}>重建向量投影</button>
            <button className="btn sm" onClick={seedNow} disabled={busy}>重新播种</button>
          </div>
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <span className={`pill ${health && health.ok ? 'ok' : 'err'}`}>服务 {health && health.ok ? '✓' : '✗'}</span>
          <span className={`pill ${health && health.chat_ready ? 'ok' : 'warn'}`}>
            聊天 {health && health.chat_ready ? '✓ 可用' : '✗ 未配置模型'}
          </span>
          <span className={`pill ${degraded.keyword_fallback ? 'warn' : 'ok'}`}>
            嵌入 {health && health.embedding_mode}
            {degraded.keyword_fallback ? ' · 退化为关键词' : ''}
          </span>
          <span className={`pill ${health && health.search_available ? 'ok' : 'warn'}`}>
            联网搜索 {health && health.search_available ? '✓' : '✗'}
          </span>
          {health && health.store_modes && Object.entries(health.store_modes).map(([k, v]) => (
            <span className="pill accent2" key={k}>{k}: {v}</span>
          ))}
        </div>
        {degraded.embedding_hint && <pre className="out" style={{ marginTop: 10 }}>{degraded.embedding_hint}</pre>}
        {health && (
          <div className="row" style={{ marginTop: 10 }}>
            <span className="pill">锁定模型 {health.embedding_current && health.embedding_current.model}</span>
            <span className="pill">维度 {health.qdrant_dimension}</span>
            <span className={`pill ${health.embedding_mismatch ? 'err' : 'ok'}`}>
              {health.embedding_mismatch ? '维度不一致（需重建）' : '一致'}
            </span>
            <span className="pill">本体抽取器 {health.knowledge_extractor}</span>
          </div>
        )}
      </div>

      <div className="card">
        <h2>库规模（knowledge.stats）</h2>
        <div className="stat-grid" style={{ marginTop: 12 }}>
          <div className="stat"><div className="k">文档数</div><div className="v b">{s.documents ?? 0}</div></div>
          <div className="stat"><div className="k">分块总数</div><div className="v">{s.chunks ?? 0}</div></div>
          <div className="stat"><div className="k">已索引分块</div><div className="v a">{s.chunks_indexed ?? 0}</div></div>
          <div className="stat"><div className="k">事实条数</div><div className="v o">{s.facts ?? 0}</div></div>
          <div className="stat"><div className="k">记忆条目</div><div className="v p">{s.memories_total ?? 0}</div></div>
        </div>
      </div>

      <div className="card">
        <h2>三库对账（真值源 ↔ 向量投影 ↔ 图投影）</h2>
        <p className="hint" style={{ margin: 0 }}>只看不改；发现漂移可用下面的按钮按类型修复。</p>
        {reconcile ? <pre className="out" style={{ marginTop: 10, maxHeight: 300 }}>{JSON.stringify(reconcile, null, 2)}</pre>
          : <div className="empty">对账数据不可用</div>}
        {(reconcile && reconcile.drift && reconcile.drift.length > 0) ? (
          <div className="row" style={{ marginTop: 10 }}>
            {reconcile.drift.map((d) => (
              <button className="btn sm warn" key={d.kind} disabled={busy} onClick={() => repair([d.kind])}>
                修复 {d.kind}（{d.count ?? 0}）
              </button>
            ))}
            <button className="btn sm primary" disabled={busy} onClick={() => repair(reconcile.drift.map((d) => d.kind))}>
              全部修复
            </button>
          </div>
        ) : (
          <div className="row" style={{ marginTop: 10 }}>
            <span className="pill ok">没有漂移 ✓</span>
          </div>
        )}
      </div>

      {msg && <div className="card"><pre className="out">{msg}</pre></div>}
    </>
  )
}