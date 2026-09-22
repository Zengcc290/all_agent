import React, { useCallback, useState } from 'react'
import { api } from '../api.js'
import { fmtScore, clip, errMessage } from '../util.js'

/** 图谱检索面板：POST /api/graph-rag（纯本地，不依赖聊天模型）。 */
export default function QueryPanel({ onHighlight, onJumpToGraph }) {
  const [query, setQuery] = useState('')
  const [limit, setLimit] = useState(5)
  const [hops, setHops] = useState(1)
  const [at, setAt] = useState('')
  const [busy, setBusy] = useState(false)
  const [res, setRes] = useState(null)
  const [err, setErr] = useState('')

  const run = useCallback(async () => {
    if (!query.trim() || busy) return
    setBusy(true); setErr(''); setRes(null)
    try {
      const r = await api.graphRag({
        query: query.trim(),
        limit: Number(limit) || 5,
        hops: Number(hops) || 0,
        at: at || undefined,
      })
      setRes(r)
      // 命中的实体/路径反哺星云图高亮
      const names = new Set(r.entities || [])
      onHighlight && onHighlight({ entityTitles: names, paths: r.paths || [] })
    } catch (e) {
      setErr(errMessage(e))
      onHighlight && onHighlight(null)
    }
    setBusy(false)
  }, [query, limit, hops, at, busy, onHighlight])

  return (
    <>
      <div className="card">
        <h2>图谱检索 · GraphRAG</h2>
        <p className="hint" style={{ margin: 0 }}>
          先向量/关键词召回证据，再沿知识图谱做 0~N 跳关系扩展，输出证据、关系路径与可直接投喂的上下文。
          未配置聊天模型也能用。
        </p>
        <label className="fld" style={{ marginTop: 10 }}>
          <span>问题 / 关键词<span className="req">*</span></span>
          <textarea value={query} onChange={(e) => setQuery(e.target.value)} rows={3}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run() } }} />
        </label>
        <div className="row" style={{ alignItems: 'flex-end' }}>
          <label className="fld" style={{ marginBottom: 0, width: 110 }}>
            <span>证据条数 limit</span>
            <input type="number" min="1" max="50" value={limit} onChange={(e) => setLimit(e.target.value)} />
          </label>
          <label className="fld" style={{ marginBottom: 0, width: 110 }}>
            <span>跳数 hops</span>
            <input type="number" min="0" max="5" value={hops} onChange={(e) => setHops(e.target.value)} />
          </label>
          <label className="fld" style={{ marginBottom: 0, flex: 1 }}>
            <span>时间点 at（历史视图，可空）</span>
            <input value={at} onChange={(e) => setAt(e.target.value)} placeholder="2024-03-01T00:00:00" />
          </label>
          <button className="btn primary" disabled={!query.trim() || busy} onClick={run}>
            {busy ? <><span className="spinner" />检索中</> : '检索'}
          </button>
        </div>
      </div>

      {err && <div className="card" style={{ borderColor: 'rgba(248,113,113,0.45)' }}><pre className="out">{err}</pre></div>}

      {res && (
        <>
          <div className="card">
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <h2>检索结果</h2>
              <div className="row tight">
                <span className="pill accent">实体 {(res.entities || []).length}</span>
                <span className="pill accent2">证据 {(res.evidence || []).length}</span>
                <span className="pill">路径 {(res.paths || []).length}</span>
                {onJumpToGraph && <button className="btn sm" onClick={onJumpToGraph}>在星云图中查看</button>}
              </div>
            </div>
            {(res.entities || []).length > 0 && (
              <div className="row tight" style={{ marginTop: 8 }}>
                {(res.entities || []).map((e, i) => <span className="pill accent" key={i}>{e}</span>)}
              </div>
            )}
          </div>

          {(res.evidence || []).length > 0 && (
            <div className="card">
              <h2>向量证据</h2>
              {(res.evidence || []).map((e, i) => (
                <div className="hit" key={i}>
                  <div className="top">
                    <span>{(e.metadata && (e.metadata.filename || e.metadata.source)) || e.id || '记忆库'}</span>
                    <span className="pill accent">相似度 {fmtScore(e.score)}</span>
                  </div>
                  <div className="body">{clip(e.content, 500)}</div>
                  {e.metadata && (e.metadata.subject || e.metadata.predicate || e.metadata.object) && (
                    <div className="tags">
                      <span className="pill accent2">
                        {[e.metadata.subject, e.metadata.predicate, e.metadata.object].filter(Boolean).join(' · ')}
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {(res.paths || []).length > 0 && (
            <div className="card">
              <h2>图关系路径</h2>
              {(res.paths || []).map((p, i) => (
                <div className="path-item" key={i}>
                  <div className="nodes">
                    {(p.entities || []).map((x, j) => <span className="nd" key={j}>{x}</span>)}
                  </div>
                  <div className="row tight" style={{ marginTop: 6 }}>
                    <span className="pill">{(p.relations || []).join(' → ') || '直达'}</span>
                    <span className="pill accent2">置信度 {fmtScore(p.confidence)}</span>
                  </div>
                  {(p.evidence || []).map((ev, j) => (
                    <div className="hit" key={j} style={{ marginTop: 8 }}>
                      <div className="body">{clip(ev.evidence || ev.source || JSON.stringify(ev), 300)}</div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}

          {res.context && (
            <div className="card">
              <h2>组装的上下文（build_context）</h2>
              <p className="hint" style={{ margin: 0 }}>这就是聊天/抽取时实际喂给模型的那段文本。</p>
              <pre className="out" style={{ marginTop: 8, maxHeight: 420 }}>{res.context}</pre>
            </div>
          )}
        </>
      )}
    </>
  )
}