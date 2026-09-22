import React, { useEffect, useState, useCallback } from 'react'
import { api } from '../api.js'

/**
 * 待入库队列：把 sqlite 中未入库成功的 chunk 渲染成可折叠列表。
 * 每条后面有「重新入库」按钮：点击 -> 立即禁用并显示「入库中」-> 真实发起入库请求。
 */
export default function PendingQueue({ onChanged, refreshKey }) {
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [open, setOpen] = useState(null)
  const [busy, setBusy] = useState({})   // chunk_id -> 'loading' | 'done' | 'error'
  const [results, setResults] = useState({}) // chunk_id -> 结果信息
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState('')

  const load = useCallback(async () => {
    setLoading(true); setErr('')
    try {
      const r = await api.pendingChunks(50, 10)
      setItems(r.result?.items || [])
      setTotal(r.result?.total_pending ?? 0)
    } catch (e) { setErr(e.message) }
    setLoading(false)
  }, [])

  useEffect(() => { load() }, [load, refreshKey])

  const reingest = async (chunkId) => {
    if (busy[chunkId]) return
    setBusy((b) => ({ ...b, [chunkId]: 'loading' }))
    setResults((r) => ({ ...r, [chunkId]: null }))
    try {
      const r = await api.reingest(chunkId)
      const res = r.result || r
      const ok = res.promoted === true
      setBusy((b) => ({ ...b, [chunkId]: ok ? 'done' : 'error' }))
      setResults((x) => ({ ...x, [chunkId]: res }))
      if (ok) setItems((list) => list.filter((i) => i.chunk_id !== chunkId))
      if (onChanged) onChanged()
    } catch (e) {
      setBusy((b) => ({ ...b, [chunkId]: 'error' }))
      setResults((x) => ({ ...x, [chunkId]: { error: e.message } }))
    }
  }

  const statusPill = (it) => {
    const q = it.qdrant_status === 'success'
    const n = it.neo4j_status === 'success'
    const cls = q && n ? 'ok' : (q || n ? 'warn' : 'err')
    const txt = q && n ? '已全部入库' : (q ? '仅 qdrant 成功' : (n ? '仅 neo4j 成功' : '两条线均未成功'))
    return <span className={`pill ${cls}`}>{txt}</span>
  }

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div>
          <h2>待入库队列 · sqlite ingest_queue</h2>
          <p className="hint" style={{ margin: 0 }}>
            只列出 <b>qdrant 或 neo4j 任一线尚未成功</b> 的 chunk；两条线都成功后该 chunk 会自动转正进 <code>chunks</code> 表。
          </p>
        </div>
        <div className="row">
          <span className="pill accent">待入库 {total} 条</span>
          <button className="btn sm" onClick={load} disabled={loading}>{loading ? '加载中…' : '刷新'}</button>
        </div>
      </div>

      {err && <p className="hint" style={{ color: 'var(--err)' }}>加载失败：{err}</p>}
      {items.length === 0 && !loading && !err && (
        <div className="empty">队列已清空 —— 所有 chunk 都已通过 qdrant 与 neo4j 两条入库线 🎉</div>
      )}

      <div className="chunk-list" style={{ marginTop: 12 }}>
        {items.map((it) => {
          const st = busy[it.chunk_id]
          const isOpen = open === it.chunk_id
          const res = results[it.chunk_id]
          return (
            <div className={`chunk-item ${st === 'done' ? 'ok' : st === 'error' ? 'err' : ''}`} key={it.chunk_id}>
              <div
                className={`chunk-head ${isOpen ? 'open' : ''}`}
                onClick={() => setOpen(isOpen ? null : it.chunk_id)}
              >
                <span className="caret">▶</span>
                <span className="prev">{it.preview || '（空）'}</span>
                {statusPill(it)}
                <span className="id" style={{ marginLeft: 'auto' }}>{it.chunk_id}</span>
                <button
                  className={`btn sm ${st === 'loading' ? '' : 'primary'}`}
                  disabled={st === 'loading'}
                  onClick={(e) => { e.stopPropagation(); reingest(it.chunk_id) }}
                >
                  {st === 'loading'
                    ? <><span className="spinner" />入库中…</>
                    : (st === 'done' ? '重新入库' : '重新入库')}
                </button>
              </div>
              {isOpen && (
                <div className="chunk-body">
                  <div className="full">
                    {it.content_full || it.preview}
                    <br /><span style={{ opacity: 0.55 }}>（ preview_chars=10，以上为截取的前 10 个字符；全文字符数 {it.char_len}）</span>
                  </div>
                  <div className="row" style={{ fontSize: 11.5, color: 'var(--txt-dim)' }}>
                    <span>document_id: <code>{it.document_id}</code></span>
                    <span>seq: {it.seq}</span>
                    <span>updated: {it.updated_at}</span>
                  </div>
                  {it.error && <p className="hint" style={{ color: 'var(--err)', margin: '6px 0 0' }}>上次错误：{it.error}</p>}
                  {res && res.error && <p className="hint" style={{ color: 'var(--err)', margin: '6px 0 0' }}>入库失败：{res.error}</p>}
                  {res && res.details && (
                    <div style={{ marginTop: 8, display: 'grid', gap: 7 }}>
                      {Object.entries(res.details).map(([line, d]) => (
                        <div key={line} className="hit" style={{ marginBottom: 0 }}>
                          <div className="top">
                            <span>{line === 'qdrant' ? 'Qdrant 向量线' : 'Neo4j 图谱线'}</span>
                            <span className={`pill ${d.ok ? 'ok' : 'err'}`}>{d.ok ? '成功' : '失败'}</span>
                          </div>
                          <div className="body" style={{ fontSize: 11.5 }}>
                            {d.ok
                              ? (line === 'qdrant'
                                ? `已写入向量点，维度 ${d.dim ?? '-'}，耗时 ${d.elapsed_ms ?? '-'}ms`
                                : `实体 ${d.entities ?? 0} 个（新增 ${d.new_entities ?? 0} / 复用 ${d.reused_entities ?? 0}），关系 ${d.relations ?? 0} 条，映射实体 ${d.linked_entities ?? 0}`)
                              : d.error}
                          </div>
                        </div>
                      ))}
                      <div>
                        <span className={`pill ${res.promoted ? 'ok' : 'warn'}`}>
                          {res.promoted ? '已转正进 chunks 表' : '仍在队列中，等待重试'}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
