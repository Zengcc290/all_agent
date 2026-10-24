import React, { useCallback, useEffect, useState } from 'react'
import { api } from './api.js'
import ForceGraph, { colorOfType } from './components/ForceGraph.jsx'
import PendingQueue from './components/PendingQueue.jsx'
import ToolConsole from './components/ToolConsole.jsx'
import IngestPanel from './components/IngestPanel.jsx'
import QueryPanel from './components/QueryPanel.jsx'
import HybridPanel from './components/HybridPanel.jsx'

const TABS = [
  { k: 'planet', t: '🪐 实体星球' },
  { k: 'ingest', t: '✨ 一句话入库' },
  { k: 'queue', t: '📥 待入库队列' },
  { k: 'hybrid', t: '🧬 多路混合检索' },
  { k: 'query', t: '🔭 多跳 / 向量查询' },
  { k: 'tools', t: '🧰 工具台' },
  { k: 'dash', t: '📊 监控台' },
]

export default function App() {
  const [tab, setTab] = useState('planet')
  const [nodes, setNodes] = useState([])
  const [links, setLinks] = useState([])
  const [stats, setStats] = useState(null)
  const [tools, setTools] = useState([])
  const [pendingCount, setPendingCount] = useState(0)
  const [graphLoading, setGraphLoading] = useState(false)
  const [showLabels, setShowLabels] = useState(true)
  const [toasts, setToasts] = useState([])
  const [refreshKey, setRefreshKey] = useState(0)
  const [llmTest, setLlmTest] = useState(null)

  const toast = useCallback((msg, kind = '') => {
    const id = Math.random().toString(36).slice(2)
    setToasts((t) => [...t, { id, msg, kind }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3800)
  }, [])

  const loadGraph = useCallback(async () => {
    setGraphLoading(true)
    try {
      const r = await api.graph()
      setNodes(r.nodes || [])
      setLinks(r.links || [])
    } catch (e) {
      toast('图谱加载失败：' + e.message, 'err')
    }
    setGraphLoading(false)
  }, [toast])

  const loadStats = useCallback(async () => {
    try {
      const r = await api.stats()
      setStats(r.result || r)
      setPendingCount((r.result || r).sqlite?.pending ?? 0)
    } catch (e) { /* 静默 */ }
  }, [])

  const loadTools = useCallback(async () => {
    try {
      const r = await api.tools()
      setTools(r.tools || [])
    } catch (e) { /* 静默 */ }
  }, [])

  useEffect(() => {
    loadGraph(); loadStats(); loadTools()
    const t = setInterval(loadStats, 8000)
    return () => clearInterval(t)
  }, [loadGraph, loadStats, loadTools])

  const onChanged = useCallback(() => {
    loadStats(); loadGraph()
    setRefreshKey((k) => k + 1)
  }, [loadGraph, loadStats])

  const runLlmTest = async () => {
    setLlmTest(null)
    try {
      const r = await api.llmTest()
      setLlmTest(r.result || r)
      toast('LLM / Embedding 探测完成', (r.result || r).llm?.ok ? 'ok' : 'err')
    } catch (e) { toast(e.message, 'err') }
  }

  const directed = links.filter((l) => l.directed).length
  const undirected = links.length - directed
  const s = stats || {}

  return (
    <div className="app">
      <div className="topbar">
        <div className="logo" />
        <div>
          <h1>知识图谱入库系统</h1>
          <div className="sub">FastAPI · React · Qdrant · Neo4j · SQLite · OpenAI 流式</div>
        </div>
        <div className="tabs">
          {TABS.map((x) => (
            <button key={x.k} className={`tab ${tab === x.k ? 'active' : ''}`} onClick={() => setTab(x.k)}>
              {x.t}
              {x.k === 'queue' && pendingCount > 0 && <span className="badge">{pendingCount}</span>}
              {x.k === 'tools' && tools.length > 0 && <span className="badge">{tools.length}</span>}
            </button>
          ))}
        </div>
      </div>

      <div className="main">
        {/* ---------------- 实体星球 ---------------- */}
        {tab === 'planet' && (
          <>
            <div className="card">
              <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
                <div>
                  <h2>实体星球 · 实体引力关系图</h2>
                  <p className="hint" style={{ margin: 0 }}>
                    节点 = Neo4j 实体（引力/排斥力布局，可拖拽、滚轮缩放）；连线 = 实体关系。
                    <b style={{ color: 'var(--accent)' }}> 实线箭头 = 有向关系</b>，
                    <b style={{ color: 'var(--txt-dim)' }}> 虚线 = 无向关系</b>，线上标注谓词与时间。
                  </p>
                </div>
                <div className="row">
                  <span className="pill accent">{nodes.length} 实体</span>
                  <span className="pill">{directed} 有向 →</span>
                  <span className="pill">{undirected} 无向 —</span>
                  <label style={{ fontSize: 11.5, color: 'var(--txt-dim)', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <input type="checkbox" style={{ width: 14 }} checked={showLabels}
                           onChange={(e) => setShowLabels(e.target.checked)} />显示标注
                  </label>
                  <button className="btn sm" onClick={loadGraph} disabled={graphLoading}>
                    {graphLoading ? '加载中…' : '刷新图谱'}
                  </button>
                </div>
              </div>
              {nodes.length === 0 ? (
                <div className="empty" style={{ padding: '60px 0' }}>
                  图库里还没有任何实体 —— 去「一句话入库」tab 输入一句话试试 ✨
                </div>
              ) : (
                <ForceGraph nodes={nodes} links={links} height={620} showLabels={showLabels} />
              )}
            </div>

            {nodes.length > 0 && (
              <div className="card">
                <h2>实体清单</h2>
                <p className="hint" style={{ margin: 0 }}>点击节点可拖拽查看详情；列表展示全部实体的类型与时间标记。</p>
                <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 7, maxHeight: 260, overflow: 'auto' }}>
                  {nodes.map((n) => (
                    <span key={n.id} className="pill" style={{ borderColor: colorOfType(n.type) + '55' }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: colorOfType(n.type), display: 'inline-block' }} />
                      <b style={{ color: colorOfType(n.type) }}>{n.name}</b>
                      <span style={{ opacity: 0.7 }}>{n.type}</span>
                      {n.time && <span style={{ opacity: 0.7 }}>· {n.time}</span>}
                    </span>
                  ))}
                </div>
                <h2 style={{ marginTop: 18 }}>关系清单</h2>
                <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 7, maxHeight: 220, overflow: 'auto' }}>
                  {links.map((l) => (
                    <span className="pill" key={l.id} style={{ fontSize: 10.5 }}>
                      <b>{l.source}</b>
                      <span style={{ color: 'var(--accent)' }}>{l.directed ? ' →' : ' —'}</span>
                      <span style={{ opacity: 0.9 }}>{l.predicate}</span>
                      <b>{l.target}</b>
                      {l.time && <span style={{ opacity: 0.65 }}>· {l.time}</span>}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* ---------------- 一句话入库 ---------------- */}
        {tab === 'ingest' && (
          <>
            <IngestPanel onChanged={onChanged} />
          </>
        )}

        {/* ---------------- 待入库队列 ---------------- */}
        {tab === 'queue' && (
          <>
            <PendingQueue onChanged={onChanged} refreshKey={refreshKey} />
            <ChunksCard onChanged={onChanged} />
          </>
        )}

        {/* ---------------- 多路混合检索 ---------------- */}
        {tab === 'hybrid' && <HybridPanel />}

        {/* ---------------- 查询 ---------------- */}
        {tab === 'query' && <QueryPanel />}

        {/* ---------------- 工具台 ---------------- */}
        {tab === 'tools' && (
          <ToolConsole tools={tools} onRefreshTools={() => { loadTools(); toast('已重新发现工具', 'ok') }} />
        )}

        {/* ---------------- 监控台 ---------------- */}
        {tab === 'dash' && (
          <>
            <div className="card">
              <div className="row" style={{ justifyContent: 'space-between' }}>
                <div>
                  <h2>三库监控</h2>
                  <p className="hint" style={{ margin: 0 }}>sqlite 原始文档 / 入库状态 / chunk，qdrant 向量集合，neo4j 图谱。</p>
                </div>
                <div className="row">
                  <button className="btn sm" onClick={loadStats} disabled={!stats}>刷新</button>
                  <button className="btn sm" onClick={runLlmTest}>
                    {llmTest ? '重新探测 LLM' : '探测 LLM / Embedding'}
                  </button>
                </div>
              </div>
              <div className="stat-grid" style={{ marginTop: 14 }}>
                <div className="stat"><div className="k">原始文档</div><div className="v b">{s.sqlite?.documents ?? 0}</div></div>
                <div className="stat"><div className="k">已转正 chunk</div><div className="v a">{s.sqlite?.chunks ?? 0}</div></div>
                <div className="stat"><div className="k">待入库</div><div className="v o">{s.sqlite?.pending ?? 0}</div></div>
                <div className="stat"><div className="k">队列总数</div><div className="v">{s.sqlite?.queue ?? 0}</div></div>
                <div className="stat"><div className="k">chunk↔实体映射</div><div className="v p">{s.sqlite?.entity_links ?? 0}</div></div>
                <div className="stat"><div className="k">Qdrant 向量点</div><div className="v a">{s.qdrant?.points ?? 0}</div></div>
                <div className="stat"><div className="k">Qdrant 维度</div><div className="v">{s.qdrant?.dim ?? '-'}</div></div>
                <div className="stat"><div className="k">Neo4j 实体</div><div className="v b">{s.neo4j?.entities ?? 0}</div></div>
                <div className="stat"><div className="k">Neo4j 关系</div><div className="v o">{s.neo4j?.relations ?? 0}</div></div>
                <div className="stat"><div className="k">FTS5 索引行</div><div className="v p">{s.sqlite?.fts_rows ?? 0}</div></div>
                <div className="stat"><div className="k">Neo4j chunk</div><div className="v">{s.neo4j?.chunks ?? 0}</div></div>
              </div>
              <div className="row" style={{ marginTop: 12, gap: 8 }}>
                <span className={`pill ${s.sqlite ? 'ok' : 'err'}`}>sqlite</span>
                <span className={`pill ${s.qdrant?.ok ? 'ok' : 'err'}`}>
                  qdrant ({s.qdrant?.mode || 'server'}) {s.qdrant?.ok ? `· ${s.qdrant.collection}` : '· ' + (s.qdrant?.error || '未连接')}
                </span>
                <span className={`pill ${s.neo4j?.ok ? 'ok' : 'err'}`}>
                  neo4j {s.neo4j?.ok ? '· ' + s.neo4j.uri : '· ' + (s.neo4j?.error || '未连接')}
                </span>
              </div>
              {llmTest && (
                <div style={{ marginTop: 12 }}>
                  <div className="row" style={{ gap: 8 }}>
                    <span className={`pill ${llmTest.llm?.ok ? 'ok' : 'err'}`}>
                      LLM {s.config?.llm?.model} {llmTest.llm?.ok ? '✓ 可用' : '✗ ' + (llmTest.llm?.error || '')}
                    </span>
                    <span className={`pill ${llmTest.embedding?.ok ? 'ok' : 'err'}`}>
                      Embedding {s.config?.embedding?.model} {llmTest.embedding?.ok ? `✓ dim=${llmTest.embedding.dim}` : '✗ ' + (llmTest.embedding?.error || '')}
                    </span>
                  </div>
                  {llmTest.llm?.reply && (
                    <pre className="out" style={{ marginTop: 9, maxHeight: 120 }}>LLM 回复：{llmTest.llm.reply}</pre>
                  )}
                </div>
              )}
            </div>

            <div className="card">
              <h2>当前配置</h2>
              <pre className="out" style={{ marginTop: 10 }}>
                {JSON.stringify(s.config || {}, null, 2)}
                {'\n\nsqlite 路径: '}{s.sqlite?.db_path}
              </pre>
            </div>

            <div className="card">
              <h2>数据库表结构</h2>
              <p className="hint" style={{ margin: 0 }}>
                <code>documents</code> 原始文档 · <code>ingest_queue</code> 入库状态（chunk 级 qdrant/neo4j 双状态）·
                <code>chunks</code> 两条线都成功才转正 ·<code>chunk_entities</code> chunk↔实体多对多映射。
              </p>
              <pre className="out" style={{ marginTop: 10 }}>{`documents(id, content, source, meta, created_at)
ingest_queue(chunk_id PK, document_id, seq, content,
             qdrant_status, neo4j_status, error, created_at, updated_at)
chunks(chunk_id PK, document_id, content, char_len, ingested_at)
chunk_entities(chunk_id, entity_key, entity_type, created_at)   -- 多对多：一个 chunk -> 多个实体，一个实体 <- 多个 chunk

Neo4j:
(:Entity {key, name, type, time, aliases})-[:REL {predicate, directed, time, chunk_id, evidence}]->(:Entity)
(:Chunk {id, content})-[:MENTIONS]->(:Entity)

SQLite FTS5 (trigram，中文可用):
chunks_fts(chunk_id UNINDEXED, content)   -- chunk 转正时自动同步`}</pre>
            </div>
          </>
        )}
      </div>

      <div className="toast">
        {toasts.map((t) => <div key={t.id} className={`t ${t.kind}`}>{t.msg}</div>)}
      </div>
    </div>
  )
}

function ChunksCard({ onChanged }) {
  const [rows, setRows] = useState(null)
  const [open, setOpen] = useState(null)
  const [busy, setBusy] = useState({})   // chunk_id -> 'loading' | 'done' | 'error'
  const [results, setResults] = useState({}) // chunk_id -> 结果信息

  const load = useCallback(() => {
    api.chunks().then((r) => setRows(r.items || [])).catch(() => setRows([]))
  }, [])

  useEffect(() => { load() }, [load])

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
      if (onChanged) onChanged()
      load()
    } catch (e) {
      setBusy((b) => ({ ...b, [chunkId]: 'error' }))
      setResults((x) => ({ ...x, [chunkId]: { error: e.message } }))
    }
  }

  return (
    <div className="card">
      <h2>已入库 chunk · sqlite chunks 表</h2>
      <p className="hint" style={{ margin: 0 }}>只有 qdrant 与 neo4j 双双成功的 chunk 才会出现在这里。</p>
      {rows === null && <div className="empty">加载中…</div>}
      {rows && rows.length === 0 && <div className="empty">还没有已转正的 chunk</div>}
      <div className="chunk-list" style={{ marginTop: 12 }}>
        {(rows || []).map((c) => (
          <div className="chunk-item ok" key={c.chunk_id}>
            <div className={`chunk-head ${open === c.chunk_id ? 'open' : ''}`} onClick={() => setOpen(open === c.chunk_id ? null : c.chunk_id)}>
              <span className="caret">▶</span>
              <span className="prev">{(c.content || '').slice(0, 10)}</span>
              <span className="pill ok">qdrant + neo4j ✓</span>
              <span className="id" style={{ marginLeft: 'auto' }}>{c.chunk_id}</span>
              <span className="pill">{c.char_len} 字符</span>
              <button
                className={`btn sm ${busy[c.chunk_id] === 'loading' ? '' : 'primary'}`}
                disabled={busy[c.chunk_id] === 'loading'}
                onClick={(e) => { e.stopPropagation(); reingest(c.chunk_id) }}
              >
                {busy[c.chunk_id] === 'loading' ? <><span className="spinner" />入库中…</> : '重新入库'}
              </button>
            </div>
            {open === c.chunk_id && (
              <div className="chunk-body">
                <div className="full">{c.content}</div>
                <div className="row" style={{ fontSize: 11, color: 'var(--txt-dim)' }}>
                  <span>ingested_at: {c.ingested_at}</span>
                  <span>document_id: <code>{c.document_id}</code></span>
                </div>
                {(c.entities || []).length > 0 && (
                  <div className="tags" style={{ marginTop: 8, display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                    {(c.entities || []).map((e, i) => <span className="pill accent" key={i}>{e}</span>)}
                  </div>
                )}
                {results[c.chunk_id] && (
                  <div style={{ marginTop: 8, display: 'grid', gap: 7 }}>
                    {results[c.chunk_id].error ? (
                      <p className="hint" style={{ color: 'var(--err)', margin: 0 }}>重新入库失败：{results[c.chunk_id].error}</p>
                    ) : (
                      <>
                        {(results[c.chunk_id].details || []).length > 0 && (
                          <div style={{ display: 'grid', gap: 7 }}>
                            {Object.entries(results[c.chunk_id].details).map(([line, d]) => (
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
                          </div>
                        )}
                        <span className={`pill ${results[c.chunk_id].promoted ? 'ok' : 'warn'}`}>
                          {results[c.chunk_id].promoted ? '重新入库完成 ✓' : '重新入库未转正'}
                        </span>
                      </>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
