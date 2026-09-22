import React, { useState } from 'react'
import { api } from '../api.js'
import { colorOfType } from './ForceGraph.jsx'

const SAMPLES = [
  '糖尿病患者能不能用胰岛素，剂量怎么定？',
  '北京协和医院治疗糖尿病视网膜病变的方法',
  '二甲双胍和胰岛素的副作用有什么不同？',
]

export default function HybridPanel() {
  const [q, setQ] = useState('')
  const [topK, setTopK] = useState(8)
  const [rrfK, setRrfK] = useState(60)
  const [split, setSplit] = useState(true)
  const [routes, setRoutes] = useState(['vector', 'fts'])
  const [res, setRes] = useState(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const [showDetail, setShowDetail] = useState('')

  const run = async () => {
    const text = q.trim()
    if (!text || busy) return
    setBusy(true); setErr(''); setRes(null); setShowDetail('')
    try {
      const r = await api.callTool('hybrid_search', {
        question: text, top_k: topK, rrf_k: rrfK,
        use_llm_split: split, routes: routes,
      }, { timeout: 300 })
      setRes(r.result)
    } catch (e) { setErr(e.message) }
    setBusy(false)
  }

  const toggleRoute = (r) =>
    setRoutes((rs) => (rs.includes(r) ? rs.filter(x => x !== r) : [...rs, r]))

  const RouteBadge = ({ hit }) => (
    <span className="pill" style={{
      fontSize: 10,
      color: hit === 'vector' ? 'var(--accent)' : 'var(--accent2)',
      borderColor: hit === 'vector' ? 'rgba(94,234,212,.4)' : 'rgba(129,140,248,.4)',
    }}>
      {hit === 'vector' ? '向量' : 'FTS5'} #{''}
    </span>
  )

  return (
    <>
      <div className="card">
        <h2>多路混合检索 · 向量 + FTS5 + RRF</h2>
        <p className="hint">
          ① LLM 把复合问题拆成多个可独立检索的子问题 →
          ② 每个子问题<b>并行</b>跑两条检索路（<b style={{ color: 'var(--accent)' }}>向量</b> = qdrant 语义召回，
          <b style={{ color: 'var(--accent2)' }}> FTS5</b> = sqlite 关键词精准召回）→
          ③ 用 <b>RRF</b>（倒数排序融合）把所有「子问题 × 检索路」的排名合并成一份最终排序。
          RRF 不看绝对分值、只看排名，天然免疫不同引擎打分量纲不可比的问题。
        </p>
        <textarea rows={2} value={q} onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.ctrlKey && e.key === 'Enter') run() }}
          placeholder="输入一个复合问题，例如「糖尿病患者能不能用胰岛素，剂量怎么定？」（Ctrl+Enter）" />

        <div className="row" style={{ marginTop: 10 }}>
          <button className="btn primary" onClick={run} disabled={!q.trim() || busy}>
            {busy ? <><span className="spinner" />检索中…</> : '🔍 多路检索'}
          </button>
          <label className="fld" style={{ width: 96, marginBottom: 0 }}>
            <span>top_k</span>
            <input type="number" min={1} max={50} value={topK} onChange={(e) => setTopK(+e.target.value)} />
          </label>
          <label className="fld" style={{ width: 96, marginBottom: 0 }}>
            <span>RRF k</span>
            <input type="number" min={1} max={500} value={rrfK} onChange={(e) => setRrfK(+e.target.value)} />
          </label>
          <label style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 6, color: 'var(--txt-dim)' }}>
            <input type="checkbox" style={{ width: 14 }} checked={split}
              onChange={(e) => setSplit(e.target.checked)} />
            LLM 拆分子问题
          </label>
          <div className="row" style={{ gap: 6 }}>
            {['vector', 'fts'].map((r) => (
              <button key={r} className={`btn sm ${routes.includes(r) ? 'primary' : ''}`}
                onClick={() => toggleRoute(r)}>{r === 'vector' ? '向量路' : 'FTS5 路'}</button>
            ))}
          </div>
          {SAMPLES.map((s) => (
            <button key={s} className="btn sm ghost" onClick={() => setQ(s)} disabled={busy}>
              {s.slice(0, 11)}…
            </button>
          ))}
        </div>
      </div>

      {err && <div className="card"><p className="hint" style={{ color: 'var(--err)' }}>{err}</p></div>}

      {res && (
        <>
          <div className="card">
            <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
              <div>
                <h2>① 子问题拆解</h2>
                <p className="hint" style={{ margin: 0 }}>
                  LLM 把这一条提问拆成了 <b style={{ color: 'var(--accent)' }}>{res.sub_query_count}</b> 个可独立检索的子问题。
                </p>
              </div>
              <span className="pill accent">耗时 {res.elapsed_ms}ms</span>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {(res.sub_queries || []).map((s, i) => (
                <span className="pill" key={i} style={{ fontSize: 11.5 }}>Q{i + 1}. {s}</span>
              ))}
            </div>
            <div className="row" style={{ marginTop: 12, gap: 8 }}>
              {Object.entries(res.agreement?.per_route || {}).map(([k, v]) => (
                <span className={`pill ${k === 'vector' ? 'accent' : ''}`} key={k}>
                  {k === 'vector' ? '向量路' : 'FTS5 路'}命中 {v}
                </span>
              ))}
              <span className="pill">
                多路重合 {res.agreement?.multi_route}/{res.agreement?.total_unique}
                （overlap {res.agreement?.overlap_ratio}）
              </span>
              <span className="pill">RRF k = {res.rrf_k}</span>
            </div>
          </div>

          <div className="grid2">
            <div className="card">
              <h2>② RRF 融合结果</h2>
              <p className="hint" style={{ margin: 0 }}>
                按 <code>1/(k+rank)</code> 跨路累加排序 —— <b>同时被多路命中的 chunk 会更靠前</b>。
              </p>
              <div style={{ marginTop: 12, maxHeight: 520, overflow: 'auto' }}>
                {(res.fused || []).map((h, i) => (
                  <div className="hit" key={i} style={{ marginBottom: 8 }}>
                    <div className="top">
                      <span style={{ color: 'var(--accent)', fontWeight: 700 }}>
                        #{i + 1} · rrf {h.rrf_score}
                      </span>
                      <span>命中 {h.route_count} 路</span>
                      <span style={{ opacity: 0.7 }}>{h.chunk_id}</span>
                    </div>
                    <div className="row" style={{ gap: 5, margin: '5px 0' }}>
                      {(h.route_hits || []).map((r) => (
                        <span className="pill" key={r} style={{
                          fontSize: 10,
                          color: r === 'vector' ? 'var(--accent)' : 'var(--accent2)',
                        }}>
                          {r === 'vector' ? '向量' : 'FTS5'}·排名 {h.routes?.[r]?.rank}
                          {h.routes?.[r]?.score != null ? ` · 分 ${Number(h.routes[r].score).toFixed(4)}` : ''}
                        </span>
                      ))}
                    </div>
                    <div className="body">{(h.content || '').slice(0, 200)}</div>
                    {(h.sub_queries_hit || []).length > 0 && (
                      <div className="tags">
                        {h.sub_queries_hit.map((s, j) => <span className="pill accent" key={j} style={{ fontSize: 10 }}>{s}</span>)}
                      </div>
                    )}
                    {(h.entities || []).length > 0 && (
                      <div className="tags">
                        {h.entities.map((e, j) => <span className="pill" key={j} style={{ fontSize: 10 }}>{e}</span>)}
                      </div>
                    )}
                  </div>
                ))}
                {(res.fused || []).length === 0 && <div className="empty">没有命中的 chunk（先入库一些句子）</div>}
              </div>
            </div>

            <div className="card">
              <h2>③ 各路原始结果</h2>
              <p className="hint" style={{ margin: 0 }}>展开任一路可看它自己召回的顺序与分数。</p>
              <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
                {Object.entries(res.per_query || {}).map(([sub, byRoute]) => (
                  <div className="hit" key={sub} style={{ marginBottom: 0 }}>
                    <div className="top" style={{ cursor: 'pointer' }}
                      onClick={() => setShowDetail(showDetail === sub ? '' : sub)}>
                      <span style={{ color: 'var(--accent2)', fontWeight: 600 }}>{sub}</span>
                      <span>{showDetail === sub ? '收起 ▴' : '展开 ▾'}</span>
                    </div>
                    {showDetail === sub && Object.entries(byRoute).map(([route, hits]) => (
                      <div key={route} style={{ marginTop: 8 }}>
                        <div style={{
                          fontSize: 10.5, marginBottom: 5, letterSpacing: 0.6,
                          color: route === 'vector' ? 'var(--accent)' : 'var(--accent2)',
                        }}>
                          {route.toUpperCase()} · {hits.length} 条
                        </div>
                        {hits.map((h, j) => (
                          <div key={j} style={{
                            fontSize: 11.5, color: 'var(--txt-dim)', padding: '4px 0',
                            borderBottom: '1px dashed rgba(120,150,210,.12)',
                          }}>
                            <span style={{ color: 'var(--txt)' }}>#{j + 1}</span>
                            {h.score != null && <span> score={Number(h.score).toFixed(4)}</span>}
                            {h.bm25 != null && <span> bm25={h.bm25}</span>}
                            {'  '}{(h.content || h.error || '').slice(0, 70)}
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </>
  )
}
