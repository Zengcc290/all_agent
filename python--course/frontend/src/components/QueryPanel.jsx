import React, { useState } from 'react'
import { api } from '../api.js'
import { colorOfType } from './ForceGraph.jsx'

export default function QueryPanel() {
  const [entity, setEntity] = useState('')
  const [hops, setHops] = useState(2)
  const [direction, setDirection] = useState('both')
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [thres, setThres] = useState(0)
  const [gRes, setGRes] = useState(null)
  const [vRes, setVRes] = useState(null)
  const [gErr, setGErr] = useState('')
  const [vErr, setVErr] = useState('')
  const [busy, setBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)

  const runGraph = async () => {
    setBusy(true); setGErr(''); setGRes(null)
    try {
      const r = await api.callTool('query_graph', { entity, hops, limit: 50, direction })
      setGRes(r.result)
    } catch (e) { setGErr(e.message) }
    setBusy(false)
  }

  const runVector = async () => {
    setBusy(true); setVErr(''); setVRes(null)
    try {
      const r = await api.callTool('search_similar_chunks', { query, top_k: topK, score_threshold: thres })
      setVRes(r.result)
    } catch (e) { setVErr(e.message) }
    setBusy(false)
  }

  const runBoth = async () => {
    if (!entity.trim() || !query.trim()) return
    setBusy(true); setGErr(''); setVErr(''); setGRes(null); setVRes(null)
    const t0 = performance.now()
    try {
      const r = await api.callMany([
        { tool: 'query_graph', args: { entity, hops, limit: 50, direction } },
        { tool: 'search_similar_chunks', args: { query, top_k: topK, score_threshold: thres } },
      ])
      const [g, v] = r.results
      if (g.ok) setGRes(g.result); else setGErr(g.error)
      if (v.ok) setVRes(v.result); else setVErr(v.error)
    } catch (e) { setGErr(e.message); setVErr(e.message) }
    setElapsed(Math.round(performance.now() - t0))
    setBusy(false)
  }

  const n = (gRes && gRes.nodes) || []
  const byName = new Map(n.map((x) => [x.key, x]))

  return (
    <>
      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div>
            <h2>双模查询 · Neo4j 多跳 ∥ Qdrant 向量相似度</h2>
            <p className="hint" style={{ margin: 0 }}>
              左侧使用 <code>query_graph</code>（沿 <code>-[:REL*1..N]-</code> 做可变长度路径遍历）；
              右侧使用 <code>search_similar_chunks</code>（embedding 编码后在 qdrant 里做最近邻检索）。
              点「并行查询」两个工具会通过 <code>asyncio.gather</code> 同时执行。
            </p>
          </div>
        </div>
        <div className="grid2" style={{ marginTop: 14 }}>
          <div>
            <h2 style={{ fontSize: 13, color: 'var(--accent)' }}>Neo4j · 多跳查询</h2>
            <label className="fld">
              <span>起始实体<span className="en">entity</span></span>
              <input value={entity} onChange={(e) => setEntity(e.target.value)} placeholder="如：糖尿病 / 张三 / nanpuyuanqu" />
            </label>
            <div className="row">
              <label className="fld" style={{ width: 120 }}>
                <span>跳数</span>
                <input type="number" min={1} max={5} value={hops} onChange={(e) => setHops(+e.target.value)} />
              </label>
              <label className="fld" style={{ width: 150 }}>
                <span>方向</span>
                <select value={direction} onChange={(e) => setDirection(e.target.value)}>
                  <option value="both">both（无向双向）</option>
                  <option value="out">out（只沿有向）</option>
                </select>
              </label>
              <button className="btn" onClick={runGraph} disabled={busy || !entity.trim()}>执行多跳查询</button>
            </div>
          </div>
          <div>
            <h2 style={{ fontSize: 13, color: 'var(--accent)' }}>Qdrant · 向量相似度</h2>
            <label className="fld">
              <span>查询文本<span className="en">query</span></span>
              <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="输入要检索的文本" />
            </label>
            <div className="row">
              <label className="fld" style={{ width: 110 }}>
                <span>top_k</span>
                <input type="number" min={1} max={50} value={topK} onChange={(e) => setTopK(+e.target.value)} />
              </label>
              <label className="fld" style={{ width: 130 }}>
                <span>分数下限</span>
                <input type="number" step={0.05} min={0} max={1} value={thres} onChange={(e) => setThres(+e.target.value)} />
              </label>
              <button className="btn" onClick={runVector} disabled={busy || !query.trim()}>执行向量检索</button>
            </div>
          </div>
        </div>
        <div className="row" style={{ marginTop: 4 }}>
          <button className="btn primary" onClick={runBoth} disabled={busy || !entity.trim() || !query.trim()}>
            {busy ? <><span className="spinner" />执行中…</> : '⚡ 并行执行两个查询'}
          </button>
          {elapsed > 0 && <span className="pill accent">并行总耗时 {elapsed}ms</span>}
        </div>
      </div>

      <div className="grid2">
        <div className="card">
          <h2>多跳路径结果</h2>
          {gErr && <p className="hint" style={{ color: 'var(--err)' }}>{gErr}</p>}
          {gRes && (
            <>
              <p className="hint" style={{ margin: '6px 0' }}>
                命中 <b style={{ color: 'var(--accent)' }}>{gRes.path_count}</b> 条路径
                {gRes.direction === 'both' ? '（无向遍历，双向可达）' : '（沿有向关系前进）'}
              </p>
              <div style={{ maxHeight: 460, overflow: 'auto' }}>
                {(gRes.paths || []).map((p, i) => (
                  <div className="path-item" key={i} style={{ marginBottom: 8 }}>
                    <div className="nodes">
                      {(p.nodes || []).map((nd, j) => (
                        <React.Fragment key={j}>
                          {j > 0 && (
                            <span className={`arw ${(p.rels || [])[j - 1]?.directed === false ? 'u' : ''}`}>
                              {(p.rels || [])[j - 1]?.predicate}
                              {(p.rels || [])[j - 1]?.time ? ` · ${p.rels[j - 1].time}` : ''}
                            </span>
                          )}
                          <span className="nd" style={{ borderColor: colorOfType(nd.type) + '66', color: colorOfType(nd.type) }}>
                            {nd.name}{nd.time ? ` ⏱${nd.time}` : ''}
                          </span>
                        </React.Fragment>
                      ))}
                    </div>
                    <div style={{ fontSize: 10.5, color: 'var(--txt-dim)', marginTop: 5 }}>
                      {p.rels?.length} 跳 · {p.nodes?.length} 个节点
                    </div>
                  </div>
                ))}
                {(gRes.paths || []).length === 0 && <div className="empty">没有命中路径</div>}
              </div>
            </>
          )}
        </div>

        <div className="card">
          <h2>向量相似度结果</h2>
          {vErr && <p className="hint" style={{ color: 'var(--err)' }}>{vErr}</p>}
          {vRes && (
            <>
              <p className="hint" style={{ margin: '6px 0' }}>
                命中 <b style={{ color: 'var(--accent)' }}>{vRes.count}</b> 条 · 向量维度 {vRes.dim}
              </p>
              <div style={{ maxHeight: 460, overflow: 'auto' }}>
                {(vRes.hits || []).map((h, i) => (
                  <div className="hit" key={i} style={{ marginBottom: 8 }}>
                    <div className="top">
                      <span>#{i + 1} · score {h.score}</span>
                      <span>{h.chunk_id}</span>
                    </div>
                    <div className="body">{(h.content || '').slice(0, 220)}</div>
                    {h.entities && h.entities.length > 0 && (
                      <div className="tags">
                        {h.entities.map((e, j) => <span className="pill" key={j}>{e}</span>)}
                      </div>
                    )}
                  </div>
                ))}
                {(vRes.hits || []).length === 0 && <div className="empty">没有命中的 chunk</div>}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  )
}

export { colorOfType }
