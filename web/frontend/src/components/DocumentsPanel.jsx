import React, { useCallback, useEffect, useState } from 'react'
import { api, isLockMismatch, lockMismatchText } from '../api.js'
import { errMessage, fmtBytes, clip } from '../util.js'

/** 文档中心：文档列表 / 详情（按真值源 char_start/char_end 高亮分块）/ 重嵌入 / 导出导入。 */
export default function DocumentsPanel({ refreshKey, onDataChanged, onPickChunk }) {
  const [page, setPage] = useState(1)
  const [status, setStatus] = useState('')
  const [tag, setTag] = useState('')
  const [data, setData] = useState(null)
  const [doc, setDoc] = useState(null)
  const [chunkId, setChunkId] = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')

  const load = useCallback(async () => {
    try {
      setData(await api.documents({ page, page_size: 20, status, tag }))
    } catch (e) {
      setMsg(errMessage(e))
    }
  }, [page, status, tag])

  useEffect(() => { load() }, [load, refreshKey])

  const open = useCallback(async (documentId) => {
    setBusy(true); setMsg(''); setChunkId('')
    try {
      setDoc(await api.document(documentId))
    } catch (e) {
      setMsg(errMessage(e))
    }
    setBusy(false)
  }, [])

  const revectorize = useCallback(async (documentId) => {
    setBusy(true); setMsg('')
    try {
      const r = await api.revectorize(documentId)
      setMsg(`重嵌入完成：${JSON.stringify(r)}`)
      open(documentId)
      onDataChanged && onDataChanged()
    } catch (e) {
      setMsg(isLockMismatch(e) ? lockMismatchText(e) : errMessage(e))
    }
    setBusy(false)
  }, [open, onDataChanged])

  const items = (data && data.items) || []
  const chunks = (doc && doc.chunks) || []
  const picked = chunks.find((c) => c.chunk_id === chunkId) || null
  const rawText = (doc && doc.raw_text) || ''

  return (
    <>
      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div>
            <h2>文档中心 · 真值源（SQLite documents / chunks）</h2>
            <p className="hint" style={{ margin: 0 }}>
              共 {data ? data.total : 0} 篇。状态流转：parsed → vectorized → extracted；failed 可重嵌入。
            </p>
          </div>
          <div className="row tight">
            <a className="btn sm" href={api.exportUrl()} target="_blank" rel="noreferrer">导出 JSON</a>
            <ImportButton onDone={onDataChanged} />
            <button className="btn sm" onClick={load} disabled={busy}>刷新</button>
          </div>
        </div>

        <div className="row" style={{ marginTop: 10 }}>
          <input style={{ width: 180 }} placeholder="按 tag 过滤" value={tag} onChange={(e) => { setTag(e.target.value); setPage(1) }} />
          <select style={{ width: 160 }} value={status} onChange={(e) => { setStatus(e.target.value); setPage(1) }}>
            <option value="">全部状态</option>
            <option value="parsed">parsed</option>
            <option value="vectorized">vectorized</option>
            <option value="extracted">extracted</option>
            <option value="failed">failed</option>
          </select>
          <span className="spacer" />
          <button className="btn sm" disabled={page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>上一页</button>
          <span className="pill">{data ? `第 ${data.page} 页 / ${Math.max(1, Math.ceil(data.total / (data.page_size || 20)))} 页` : '…'}</span>
          <button className="btn sm" disabled={!data || data.page * data.page_size >= data.total} onClick={() => setPage((p) => p + 1)}>下一页</button>
        </div>

        {items.length === 0 && <div className="empty">还没有文档，去「入库」上传一篇试试</div>}
        <div className="list" style={{ marginTop: 12 }}>
          {items.map((d) => (
            <div className={`item ${doc && doc.document_id === d.document_id ? 'sel' : ''}`} key={d.document_id}>
              <div className="head">
                <span className="t">{d.title || d.source || d.document_id}</span>
                <span className={`pill ${d.status === 'failed' ? 'err' : d.status === 'extracted' ? 'ok' : 'accent2'}`}>{d.status}</span>
                <span className="pill">{d.chunk_count} 块</span>
                {(d.tags || []).map((t) => <span className="pill" key={t}>{t}</span>)}
                <span className="id">{d.document_id}</span>
                <button className="btn sm" onClick={() => open(d.document_id)} disabled={busy}>查看</button>
                <button className={`btn sm ${d.status === 'failed' ? 'primary' : ''}`} onClick={() => revectorize(d.document_id)} disabled={busy}>
                  {busy ? <span className="spinner" /> : '重嵌入'}
                </button>
              </div>
              <div className="hint" style={{ margin: '6px 0 0' }}>{d.source || '来源未知'} · {d.created_at || ''}</div>
            </div>
          ))}
        </div>
        {msg && <pre className="out" style={{ marginTop: 10 }}>{msg}</pre>}
      </div>

      {doc && (
        <div className="card">
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <h2>{doc.title || doc.document_id}</h2>
            <div className="row tight">
              <span className={`pill ${doc.status === 'failed' ? 'err' : 'ok'}`}>{doc.status}</span>
              <span className="pill">{chunks.length} 块</span>
              {onPickChunk && (
                <button className="btn sm" onClick={() => onPickChunk(picked)} disabled={!picked}>在星云图中定位</button>
              )}
              <button className="btn sm ghost" onClick={() => setDoc(null)}>收起</button>
            </div>
          </div>
          {doc.error && (
            <pre className="out" style={{ marginTop: 8, borderColor: 'rgba(248,113,113,0.45)' }}>{doc.error}</pre>
          )}

          <h3>分块（向量状态）</h3>
          <div className="chunk-list">
            {chunks.map((c) => (
              <div
                className={`chunk-item ${chunkId === c.chunk_id ? 'sel' : ''}`}
                key={c.chunk_id}
                style={{ cursor: 'pointer', borderColor: chunkId === c.chunk_id ? 'var(--accent)' : undefined }}
                onClick={() => setChunkId(chunkId === c.chunk_id ? '' : c.chunk_id)}
              >
                <div className="meta">
                  #{c.chunk_index} · {c.chunk_id} · 字符 {c.char_start}–{c.char_end} · 向量 {c.vector_status}
                </div>
                <div className="txt">{clip(c.text, 240)}</div>
              </div>
            ))}
          </div>

          <h3>原文（raw_text，{fmtBytes(rawText.length)} 字符；选中分块按真值源偏移高亮）</h3>
          <pre className="out" style={{ maxHeight: 320 }}>{renderHighlighted(rawText, picked)}</pre>
        </div>
      )}
    </>
  )
}

/** 原文切片渲染：选中分块用 <mark>，其余原文原样保留（不重切分）。 */
function renderHighlighted(rawText, chunk) {
  if (!chunk || chunk.char_start == null || chunk.char_end == null) return rawText
  const start = Math.max(0, Math.min(rawText.length, chunk.char_start))
  const end = Math.max(start, Math.min(rawText.length, chunk.char_end))
  return [
    <span key="head">{rawText.slice(0, start)}</span>,
    <mark key="mid">{rawText.slice(start, end)}</mark>,
    <span key="tail">{rawText.slice(end)}</span>,
  ]
}

function ImportButton({ onDone }) {
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  const onChange = useCallback(async (e) => {
    const file = e.target.files && e.target.files[0]
    if (!file) return
    setBusy(true); setMsg('')
    try {
      const r = await api.importFile(file)
      setMsg(`导入完成：${JSON.stringify(r)}`)
      onDone && onDone()
    } catch (err) {
      setMsg(errMessage(err))
    }
    setBusy(false)
    e.target.value = ''
  }, [onDone])

  return (
    <>
      <label className="btn sm" style={{ cursor: 'pointer' }}>
        {busy ? '导入中…' : '导入 JSON'}
        <input type="file" accept="application/json,.json" onChange={onChange} style={{ display: 'none' }} />
      </label>
      {msg && <span className="hint" style={{ marginLeft: 6 }}>{msg}</span>}
    </>
  )
}