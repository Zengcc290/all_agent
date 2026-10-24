import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api.js'
import NebulaGraph from './components/NebulaGraph.jsx'
import ChatPanel from './components/ChatPanel.jsx'
import IngestPanel from './components/IngestPanel.jsx'
import QueryPanel from './components/QueryPanel.jsx'
import DocumentsPanel from './components/DocumentsPanel.jsx'
import JobsPanel from './components/JobsPanel.jsx'
import DashboardPanel from './components/DashboardPanel.jsx'
import { domainColor, kindColor, kindLabel, clip, errMessage } from './util.js'

const TABS = [
  { k: 'nebula', t: '🪐 星云图' },
  { k: 'chat', t: '💬 知识管家' },
  { k: 'ingest', t: '✨ 入库' },
  { k: 'query', t: '🔭 图谱检索' },
  { k: 'docs', t: '📚 文档中心' },
  { k: 'jobs', t: '📥 入库任务' },
  { k: 'monitor', t: '📊 监控台' },
]

const POLL_MS = 8000

/** 校验 hash 里的 tab 名，非法值回落到默认 tab。 */
function tabFromHash() {
  const raw = String(window.location.hash || '').replace(/^#/, '')
  return TABS.some((x) => x.k === raw) ? raw : 'nebula'
}

export default function App() {
  const [tab, setTab] = useState(tabFromHash)
  const [payload, setPayload] = useState({ nodes: [], edges: [], stats: {}, revision: 0, unchanged: true })
  const [graphLoading, setGraphLoading] = useState(false)
  const [showLabels, setShowLabels] = useState(true)
  const [selected, setSelected] = useState(null)
  const [highlight, setHighlight] = useState(null)
  const [domainFilter, setDomainFilter] = useState('')
  const [kindFilter, setKindFilter] = useState('')
  const [asOf, setAsOf] = useState('')
  const [pendingJobs, setPendingJobs] = useState(0)
  const [refreshKey, setRefreshKey] = useState(0)
  const [toasts, setToasts] = useState([])
  const revisionRef = useRef(0)

  // tab 与 URL hash 双向同步：支持 #chat 这类深链，刷新后停在同一页
  useEffect(() => {
    const onHash = () => setTab(tabFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const goto = useCallback((key) => {
    setTab(key)
    if (window.location.hash !== '#' + key) window.location.hash = key
  }, [])

  const toast = useCallback((msg, kind = '') => {
    const id = Math.random().toString(36).slice(2)
    setToasts((t) => [...t, { id, msg, kind }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3800)
  }, [])

  /** 全量拉图；since>=0 时做增量轮询（后端 unchanged 时只回 stats）。 */
  const loadGraph = useCallback(async (delta = false) => {
    setGraphLoading(true)
    try {
      const r = await api.graph(delta ? revisionRef.current : -1, asOf)
      if (r.unchanged) {
        revisionRef.current = r.revision || revisionRef.current
      } else {
        revisionRef.current = r.revision || 0
        setPayload({ nodes: r.nodes || [], edges: r.edges || [], stats: r.stats || {}, revision: r.revision, unchanged: false })
      }
    } catch (e) {
      toast('图谱加载失败：' + errMessage(e), 'err')
    }
    setGraphLoading(false)
  }, [asOf, toast])

  const refresh = useCallback(() => {
    revisionRef.current = -1
    loadGraph().then(() => setRefreshKey((k) => k + 1))
  }, [loadGraph])

  // 队列角标：有排队/进行中的任务时显示
  const loadJobs = useCallback(async () => {
    try {
      const r = await api.jobs('', 50)
      const rows = r.items || []
      setPendingJobs(rows.filter((j) => j.status === 'pending' || j.status === 'running').length)
    } catch { /* 静默：内存库下队列不可用 */ }
  }, [])

  useEffect(() => {
    loadGraph()
    loadJobs()
    const t = setInterval(() => { loadGraph(true); loadJobs() }, POLL_MS)
    return () => clearInterval(t)
  }, [loadGraph, loadJobs])

  // 领域 / 类型筛选（纯前端）
  const domains = useMemo(
    () => [...new Set(payload.nodes.map((n) => n.domain).filter(Boolean))].sort(),
    [payload.nodes],
  )
  const visible = useMemo(() => {
    const keep = payload.nodes.filter(
      (n) => (!domainFilter || n.domain === domainFilter) && (!kindFilter || n.kind === kindFilter),
    )
    const keepIds = new Set(keep.map((n) => n.id))
    // 领域筛选后，卫星节点没有连线，靠恒星的层级引力仍然聚成一团
    return { nodes: keep, links: payload.edges.filter((e) => keepIds.has(e.source) && keepIds.has(e.target)) }
  }, [payload, domainFilter, kindFilter])

  /** 检索结果 → 星云图高亮（节点按标题匹配，边按路径相邻关系匹配）。 */
  const applyHighlight = useCallback((spec) => {
    if (!spec) { setHighlight(null); return }
    const byTitle = new Map()
    for (const n of payload.nodes) {
      const key = String(n.title || '')
      if (!byTitle.has(key)) byTitle.set(key, new Set())
      byTitle.get(key).add(n.id)
    }
    const nodeIds = new Set()
    for (const name of spec.entityTitles || []) {
      for (const id of (byTitle.get(name) || [])) nodeIds.add(id)
    }
    const titleToId = (t) => (byTitle.has(t) ? [...byTitle.get(t)][0] : null)
    const edgeKeys = new Set()
    for (const p of spec.paths || []) {
      const ents = p.entities || []
      for (let i = 0; i < ents.length - 1; i += 1) {
        const a = titleToId(ents[i])
        const b = titleToId(ents[i + 1])
        if (a && b) { edgeKeys.add(`${a}->${b}`); edgeKeys.add(`${b}->${a}`) }
      }
    }
    setHighlight({ nodes: nodeIds, edges: edgeKeys })
  }, [payload.nodes])

  const selectedEdges = useMemo(() => {
    if (!selected) return []
    return payload.edges.filter((e) => e.source === selected.id || e.target === selected.id)
  }, [payload.edges, selected])

  const s = payload.stats || {}
  const visibleCount = visible.nodes.length

  return (
    <div className="app">
      <div className="topbar">
        <div className="logo" />
        <div>
          <h1>知识星云 · Aetheria</h1>
          <div className="sub">FastAPI · React · Qdrant · Neo4j · SQLite · GraphRAG</div>
        </div>
        <div className="tabs">
          {TABS.map((x) => (
            <button key={x.k} className={`tab ${tab === x.k ? 'active' : ''}`} onClick={() => goto(x.k)}>
              {x.t}
              {x.k === 'jobs' && pendingJobs > 0 && <span className="badge">{pendingJobs}</span>}
            </button>
          ))}
        </div>
      </div>

      <div className="main">
        {tab === 'nebula' && (
          <>
            <div className="card">
              <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
                <div>
                  <h2>实体星球 · 知识星云图</h2>
                  <p className="hint" style={{ margin: 0 }}>
                    <b style={{ color: 'var(--warn)' }}>恒星=领域</b>，
                    <b style={{ color: kindColor('entity') }}>行星=实体</b>，
                    <b style={{ color: kindColor('fact') }}>卫星=事实/知识块</b>；
                    拖拽节点可钉住，滚轮缩放，点击查看详情。每 {POLL_MS / 1000}s 按 revision 增量刷新，无变化不重排。
                  </p>
                </div>
                <div className="row tight">
                  <span className="pill accent">{visibleCount} / {payload.nodes.length} 节点</span>
                  <span className="pill">{s.relations ?? 0} 关系</span>
                  <span className="pill">{payload.edges.length} 边</span>
                  <label className="row tight" style={{ fontSize: 11.5, color: 'var(--txt-dim)' }}>
                    <input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} />显示标注
                  </label>
                  <button className="btn sm" onClick={() => loadGraph()} disabled={graphLoading}>
                    {graphLoading ? <><span className="spinner" />加载中</> : '刷新图谱'}
                  </button>
                </div>
              </div>
              <div className="row" style={{ marginBottom: 12 }}>
                <select style={{ width: 170 }} value={domainFilter} onChange={(e) => setDomainFilter(e.target.value)}>
                  <option value="">全部领域</option>
                  {domains.map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
                <select style={{ width: 150 }} value={kindFilter} onChange={(e) => setKindFilter(e.target.value)}>
                  <option value="">全部类型</option>
                  <option value="domain">领域</option>
                  <option value="entity">实体</option>
                  <option value="fact">事实</option>
                  <option value="chunk">知识块</option>
                  <option value="note">备注</option>
                  <option value="event">事件</option>
                </select>
                <input
                  style={{ width: 230 }} placeholder="历史视图 at（ISO-8601，如 2024-03-01T00:00:00）"
                  value={asOf}
                  onChange={(e) => setAsOf(e.target.value)}
                  onBlur={() => { revisionRef.current = -1; loadGraph() }}
                  onKeyDown={(e) => { if (e.key === 'Enter') { revisionRef.current = -1; loadGraph() } }}
                />
                {(domainFilter || kindFilter || asOf) && (
                  <button className="btn sm ghost" onClick={() => { setDomainFilter(''); setKindFilter(''); setAsOf('') }}>清空筛选</button>
                )}
              </div>
              {visible.nodes.length === 0 ? (
                <div className="empty" style={{ padding: '60px 0' }}>
                  图库里还没有节点 —— 去「入库」添加一句话，或点右上「重新播种」
                </div>
              ) : (
                <NebulaGraph
                  nodes={visible.nodes}
                  links={visible.links}
                  height={620}
                  showLabels={showLabels}
                  highlight={highlight}
                  selectedId={selected ? selected.id : ''}
                  onSelect={setSelected}
                />
              )}
            </div>

            {selected && (
              <div className="card">
                <div className="row" style={{ justifyContent: 'space-between' }}>
                  <h2>{selected.title}</h2>
                  <div className="row tight">
                    <span className="pill accent2">{kindLabel(selected.kind)}</span>
                    <span className="pill" style={{ borderColor: domainColor(selected.domain) }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: domainColor(selected.domain), display: 'inline-block' }} />
                      {selected.domain}
                    </span>
                    {selected.date && <span className="pill">{selected.date}</span>}
                    <span className="pill">重要度 {selected.importance}</span>
                    <button className="btn sm ghost" onClick={() => setSelected(null)}>关闭</button>
                  </div>
                </div>
                {selected.meta && (selected.meta.aliases || []).length > 0 && (
                  <div className="row tight" style={{ marginTop: 8 }}>
                    <span className="hint" style={{ margin: 0 }}>别名</span>
                    {selected.meta.aliases.map((a, i) => <span className="pill accent" key={i}>{a}</span>)}
                  </div>
                )}
                {selected.content && <div className="body" style={{ marginTop: 10 }}>{selected.content}</div>}
                {selectedEdges.length > 0 && (
                  <>
                    <h3>关联关系（{selectedEdges.length}）</h3>
                    <div className="row tight">
                      {selectedEdges.map((e) => {
                        const other = e.source === selected.id ? e.target : e.source
                        const otherNode = payload.nodes.find((n) => n.id === other)
                        return (
                          <span className="pill" key={e.id}>
                            <b>{e.relation}</b> · {otherNode ? otherNode.title : other}
                            {e.event_at ? ` · ${String(e.event_at).slice(0, 10)}` : ''}
                          </span>
                        )
                      })}
                    </div>
                  </>
                )}
              </div>
            )}

            {payload.nodes.length > 0 && (
              <div className="card">
                <h2>图谱统计（/api/graph stats）</h2>
                <div className="stat-grid" style={{ marginTop: 12 }}>
                  <div className="stat"><div className="k">领域</div><div className="v b">{s.domains ?? 0}</div></div>
                  <div className="stat"><div className="k">实体</div><div className="v a">{s.entities ?? 0}</div></div>
                  <div className="stat"><div className="k">关系</div><div className="v o">{s.relations ?? 0}</div></div>
                  <div className="stat"><div className="k">事实</div><div className="v p">{s.facts ?? 0}</div></div>
                  <div className="stat"><div className="k">知识块</div><div className="v">{s.chunks ?? 0}</div></div>
                  <div className="stat"><div className="k">备注</div><div className="v">{s.notes ?? 0}</div></div>
                  <div className="stat"><div className="k">事件</div><div className="v">{s.events ?? 0}</div></div>
                  <div className="stat"><div className="k">历史事实</div><div className="v">{s.historical_facts ?? 0}</div></div>
                  <div className="stat"><div className="k">孤儿实体</div><div className="v">{s.orphan_entities ?? 0}</div></div>
                  <div className="stat"><div className="k">节点合计</div><div className="v b">{s.total ?? payload.nodes.length}</div></div>
                </div>
              </div>
            )}
          </>
        )}

        {tab === 'chat' && <ChatPanel onDataChanged={refresh} />}
        {tab === 'ingest' && <IngestPanel onDataChanged={refresh} />}
        {tab === 'query' && (
          <QueryPanel
            onHighlight={applyHighlight}
            onJumpToGraph={() => setTab('nebula')}
          />
        )}
        {tab === 'docs' && (
          <DocumentsPanel
            refreshKey={refreshKey}
            onDataChanged={refresh}
            onPickChunk={(chunk) => { if (chunk) toast(`已定位分块 ${chunk.chunk_id}（${chunk.char_start}–${chunk.char_end}）`) }}
          />
        )}
        {tab === 'jobs' && <JobsPanel refreshKey={refreshKey} onDataChanged={refresh} />}
        {tab === 'monitor' && <DashboardPanel refreshKey={refreshKey} onDataChanged={refresh} />}
      </div>

      <div className="toast" aria-live="polite">
        {toasts.map((t) => <div key={t.id} className={`t ${t.kind}`}>{t.msg}</div>)}
      </div>
    </div>
  )
}