import React, { useCallback, useEffect, useRef, useState } from 'react'
import { api, lockMismatchText, isLockMismatch } from '../api.js'
import { fmtScore, clip, errMessage } from '../util.js'

/** 知识管家的前端入口。

 * 对应后端 POST /api/chat：ReAct 智能体，可带检索依据（sources/paths/
 * retrieval）。危险写操作（memory.manage）第一轮只给确认提案，这里把
 * confirmations 渲染成按钮；用户确认后带 confirmation 重发同一条消息。
 */
export default function ChatPanel({ onDataChanged }) {
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [mode, setMode] = useState('offline')
  const [busy, setBusy] = useState(false)
  const [chatReady, setChatReady] = useState(null)
  const [searchOn, setSearchOn] = useState(false)
  const boxRef = useRef(null)

  useEffect(() => {
    api.health().then((h) => {
      setChatReady(Boolean(h.chat_ready))
      setSearchOn(Boolean(h.search_available))
    }).catch(() => setChatReady(false))
  }, [])

  useEffect(() => {
    const el = boxRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  const push = useCallback((m) => setMessages((prev) => [...prev, { id: Math.random().toString(36).slice(2), ...m }]), [])

  const patch = useCallback((id, m) => setMessages((prev) => prev.map((x) => (x.id === id ? { ...x, ...m } : x))), [])

  const send = useCallback(async (text, confirmation) => {
    const content = (text ?? draft).trim()
    if (!content || busy) return
    if (!confirmation) push({ role: 'user', text: content })
    setDraft('')
    setBusy(true)
    const pendingId = Math.random().toString(36).slice(2)
    push({ id: pendingId, role: 'assistant', pending: true, text: '' })
    try {
      const r = await api.chat({ message: content, mode, confirmation })
      patch(pendingId, {
        pending: false,
        text: r.answer,
        effectiveMode: r.mode,
        sources: r.sources || [],
        paths: r.paths || [],
        retrieval: r.retrieval || {},
        confirmations: r.confirmations || [],
        askedText: content,
      })
      if ((r.confirmations || []).length === 0) onDataChanged && onDataChanged()
    } catch (e) {
      if (isLockMismatch(e)) {
        patch(pendingId, { pending: false, text: lockMismatchText(e) })
      } else {
        patch(pendingId, { pending: false, text: '', error: errMessage(e) })
      }
    }
    setBusy(false)
  }, [busy, draft, mode, onDataChanged, patch, push])

  const confirm = useCallback(async (msg, call, accepted) => {
    setMessages((prev) => prev.map((x) => (
      x.id === msg.id ? { ...x, confirmations: [], confirmDone: accepted ? '已确认执行' : '已拒绝' } : x
    )))
    if (!accepted) return
    setBusy(true)
    const pendingId = Math.random().toString(36).slice(2)
    push({ id: pendingId, role: 'assistant', pending: true, text: '' })
    try {
      const r = await api.chat({
        message: msg.askedText,
        mode: msg.effectiveMode || mode,
        confirmation: call,
      })
      patch(pendingId, {
        pending: false,
        text: r.answer,
        effectiveMode: r.mode,
        sources: r.sources || [],
        paths: r.paths || [],
        retrieval: r.retrieval || {},
        confirmations: r.confirmations || [],
        askedText: msg.askedText,
      })
      onDataChanged && onDataChanged()
    } catch (e) {
      patch(pendingId, { pending: false, text: '', error: errMessage(e) })
    }
    setBusy(false)
  }, [mode, onDataChanged, patch, push])

  return (
    <div className="card">
      <h2>知识管家 · ReAct 对话</h2>
      <p className="hint" style={{ margin: 0 }}>
        {chatReady === null ? '正在探测聊天模型…'
          : chatReady ? '已接入云端聊天模型；回答会带上检索依据（向量证据 + 图关系路径）。'
            : '未配置聊天模型（ChatBody 返回 503）：请填写 config/provider.toml 后重启，或用「图谱检索」页做纯本地检索。'}
      </p>
      <div className="row" style={{ margin: '10px 0 12px' }}>
        <label className="row tight" style={{ fontSize: 11.5, color: 'var(--txt-dim)' }}>
          <input type="radio" checked={mode === 'offline'} onChange={() => setMode('offline')} /> 本地记忆
        </label>
        <label className="row tight" style={{ fontSize: 11.5, color: searchOn ? 'var(--txt-dim)' : 'var(--err)' }}>
          <input type="radio" checked={mode === 'online'} onChange={() => setMode('online')} disabled={!searchOn} />
          联网搜索{searchOn ? '' : '（未配置 AnySearch）'}
        </label>
      </div>

      <div className="chat-box" ref={boxRef} aria-live="polite">
        {messages.length === 0 && <div className="empty">问点什么吧，例如「我知道哪些关于 Aetheria 的事实？」</div>}
        {messages.map((m) => (
          <div key={m.id} className={`msg ${m.role} ${m.error ? 'err' : ''}`}>
            <div className="who">
              {m.role === 'user' ? '我' : (m.effectiveMode === 'online' ? '知识管家 · 联网' : '知识管家')}
            </div>
            {m.pending && <span className="row tight"><span className="spinner" />思考中…</span>}
            {!m.pending && m.text && <div className="broken">{m.text}</div>}
            {m.error && <div className="broken" style={{ color: 'var(--err)' }}>失败：{m.error}</div>}
            {m.confirmDone && <span className="pill warn" style={{ marginTop: 6 }}>{m.confirmDone}</span>}
            {m.confirmations && m.confirmations.length > 0 && (
              <div className="confirm-box" style={{ marginTop: 8 }}>
                <b>等待确认的写操作</b>
                {m.confirmations.map((c, i) => (
                  <div key={i}>
                    <div className="row" style={{ marginTop: 6 }}>
                      <span className="pill warn">{c.tool_name}</span>
                      <button className="btn sm primary" disabled={busy} onClick={() => confirm(m, c, true)}>同意执行</button>
                      <button className="btn sm danger" disabled={busy} onClick={() => confirm(m, c, false)}>拒绝</button>
                    </div>
                    <pre className="out" style={{ maxHeight: 140 }}>{JSON.stringify(c.arguments, null, 2)}</pre>
                  </div>
                ))}
              </div>
            )}
            {m.retrieval && m.retrieval.note && <p className="hint" style={{ margin: '8px 0 0' }}>{m.retrieval.note}</p>}
            {(m.sources && m.sources.length > 0 || (m.paths && m.paths.length > 0)) && (
              <details>
                <summary>检索依据（{m.sources ? m.sources.length : 0} 条证据 / {m.paths ? m.paths.length : 0} 条路径）</summary>
                <div style={{ marginTop: 8 }}>
                  {(m.sources || []).map((s, i) => (
                    <div className="hit" key={`s${i}`}>
                      <div className="top">
                        <span>{s.source || '记忆库'}</span>
                        <span className="pill accent">相似度 {fmtScore(s.score)}</span>
                      </div>
                      <div className="body">{clip(s.context || s.snippet || '', 400)}</div>
                    </div>
                  ))}
                  {(m.paths || []).map((p, i) => (
                    <div className="path-item" key={`p${i}`}>
                      <div className="nodes">
                        {(p.entities || []).map((x, j) => <span className="nd" key={j}>{x}</span>)}
                      </div>
                      {(p.relations || []).length > 0 && (
                        <div className="hint" style={{ margin: '6px 0 0' }}>关系链：{(p.relations || []).join(' → ')}</div>
                      )}
                    </div>
                  ))}
                </div>
              </details>
            )}
            {m.retrieval && (m.retrieval.hits || []).length > 0 && (
              <details>
                <summary>逐条打分（向量 / 关键词 / RRF）</summary>
                <div style={{ marginTop: 8 }}>
                  {m.retrieval.hits.map((hit, i) => (
                    <div className="hit" key={i}>
                      <div className="top">
                        <span>{hit.source || hit.chunk_id || '记忆库'}</span>
                        <span className="pill">RRF {fmtScore(hit.rrf_score)} · 向量 {fmtScore(hit.vector_score)} · 关键词 {fmtScore(hit.keyword_score)}</span>
                      </div>
                      <div className="body">{clip(hit.snippet, 400)}</div>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
        ))}
      </div>

      <div className="row" style={{ alignItems: 'flex-end' }}>
        <label className="fld" style={{ flex: 1, marginBottom: 0 }}>
          <span>提问</span>
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
            placeholder="Enter 发送，Shift+Enter 换行"
          />
        </label>
        <button className="btn primary" disabled={busy || !draft.trim()} onClick={() => send()}>
          {busy ? <><span className="spinner" />发送中</> : '发送'}
        </button>
      </div>
    </div>
  )
}