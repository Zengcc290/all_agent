import React, { useCallback, useState } from 'react'
import { api, isLockMismatch, lockMismatchText } from '../api.js'
import { errMessage, fmtBytes } from '../util.js'

/** 入库面板：文档上传 / 一句话入库 / 图片入库 / 手工三元组。 */
export default function IngestPanel({ onDataChanged }) {
  const [tab, setTab] = useState('doc')

  return (
    <>
      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div>
            <h2>入库 · 让星云长出新星星</h2>
            <p className="hint" style={{ margin: 0 }}>
              文档走 RAG 切块；一句话与图片走 LLM 实体/关系抽取；三元组直接写语义记忆。
            </p>
          </div>
          <div className="row tight">
            {[
              { k: 'doc', t: '📄 文档' },
              { k: 'text', t: '✨ 一句话' },
              { k: 'image', t: '🖼 图片' },
              { k: 'fact', t: '📌 三元组' },
            ].map((x) => (
              <button key={x.k} className={`btn sm ${tab === x.k ? 'primary' : ''}`} onClick={() => setTab(x.k)}>{x.t}</button>
            ))}
          </div>
        </div>
      </div>

      {tab === 'doc' && <DocForm onDone={onDataChanged} />}
      {tab === 'text' && <TextForm onDone={onDataChanged} />}
      {tab === 'image' && <ImageForm onDone={onDataChanged} />}
      {tab === 'fact' && <FactForm onDone={onDataChanged} />}
    </>
  )
}

function Outcome({ out }) {
  if (!out) return null
  if (out.error) {
    return (
      <div className="card" style={{ borderColor: 'rgba(248,113,113,0.45)' }}>
        <h3 style={{ marginTop: 0, color: 'var(--err)' }}>入库失败</h3>
        <pre className="out">{out.error}</pre>
      </div>
    )
  }
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>入库完成</h3>
      <pre className="out">{JSON.stringify(out.data, null, 2)}</pre>
    </div>
  )
}

function DocForm({ onDone }) {
  const [file, setFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [out, setOut] = useState(null)

  const submit = useCallback(async () => {
    if (!file || busy) return
    setBusy(true); setOut(null)
    try {
      const r = await api.ingestFile(file)
      setOut({ data: { filename: file.name, size: fmtBytes(file.size), ...r } })
      onDone && onDone()
    } catch (e) {
      setOut({ error: isLockMismatch(e) ? lockMismatchText(e) : errMessage(e) })
    }
    setBusy(false)
  }, [file, busy, onDone])

  return (
    <>
      <div className="card">
        <h2>上传文档 → RAG 切块入库</h2>
        <p className="hint" style={{ margin: 0 }}>
          后端按 800 字符切块（重叠 120），逐块做 LLM 领域分类与实体/关系抽取，再写入 Qdrant 与 Neo4j。
        </p>
        <div className="row" style={{ marginTop: 10 }}>
          <input type="file" onChange={(e) => setFile(e.target.files && e.target.files[0])} style={{ flex: 1 }} />
          <button className="btn primary" disabled={!file || busy} onClick={submit}>
            {busy ? <><span className="spinner" />上传中</> : '上传并入库'}
          </button>
        </div>
        {file && <p className="hint" style={{ marginTop: 8 }}>{file.name} · {fmtBytes(file.size)}</p>}
      </div>
      <Outcome out={out} />
    </>
  )
}

function TextForm({ onDone }) {
  const [text, setText] = useState('')
  const [eventAt, setEventAt] = useState('')
  const [wait, setWait] = useState(true)
  const [busy, setBusy] = useState(false)
  const [out, setOut] = useState(null)

  const submit = useCallback(async () => {
    if (!text.trim() || busy) return
    setBusy(true); setOut(null)
    try {
      const r = await api.addKnowledge({ text: text.trim(), event_at: eventAt, wait })
      setOut({ data: r })
      if (wait) onDone && onDone()
    } catch (e) {
      setOut({ error: isLockMismatch(e) ? lockMismatchText(e) : errMessage(e) })
    }
    setBusy(false)
  }, [text, eventAt, wait, busy, onDone])

  return (
    <>
      <div className="card">
        <h2>一句话入库</h2>
        <p className="hint" style={{ margin: 0 }}>
          原文向量化 + LLM 抽取实体、关系与事件时间，直接长进图里。同步等待可立即看到抽取报告。
        </p>
        <label className="fld" style={{ marginTop: 10 }}>
          <span>原文<span className="req">*</span></span>
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} placeholder="例如：张三于 2024 年 3 月在杭州加入了阿里云团队。" />
        </label>
        <div className="row" style={{ alignItems: 'flex-end' }}>
          <label className="fld" style={{ marginBottom: 0, flex: 1 }}>
            <span>事件时间 <span className="en">event_at，可留空</span></span>
            <input value={eventAt} onChange={(e) => setEventAt(e.target.value)} placeholder="2024-03-01 或 ISO-8601" />
          </label>
          <label className="row tight" style={{ fontSize: 11.5, color: 'var(--txt-dim)', marginBottom: 8 }}>
            <input type="checkbox" checked={wait} onChange={(e) => setWait(e.target.checked)} />
            同步等待结果（关闭则提交后台队列）
          </label>
          <button className="btn primary" disabled={!text.trim() || busy} onClick={submit}>
            {busy ? <><span className="spinner" />入库中</> : '入库'}
          </button>
        </div>
      </div>
      <Outcome out={out} />
    </>
  )
}

function ImageForm({ onDone }) {
  const [file, setFile] = useState(null)
  const [text, setText] = useState('')
  const [capturedAt, setCapturedAt] = useState('')
  const [busy, setBusy] = useState(false)
  const [out, setOut] = useState(null)

  const submit = useCallback(async () => {
    if (!file || busy) return
    setBusy(true); setOut(null)
    try {
      const r = await api.addImageKnowledge({ file, text, captured_at: capturedAt })
      setOut({ data: { filename: file.name, ...r } })
      onDone && onDone()
    } catch (e) {
      setOut({ error: isLockMismatch(e) ? lockMismatchText(e) : errMessage(e) })
    }
    setBusy(false)
  }, [file, text, capturedAt, busy, onDone])

  return (
    <>
      <div className="card">
        <h2>图片 / 相机观测入库</h2>
        <p className="hint" style={{ margin: 0 }}>
          VL 嵌入 + 视觉模型抽取主体、谓词、宾语与参与者，生成多元关系写入图库。
        </p>
        <div className="row" style={{ marginTop: 10 }}>
          <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files && e.target.files[0])} style={{ flex: 1 }} />
        </div>
        <label className="fld" style={{ marginTop: 10 }}>
          <span>图片说明（可选）</span>
          <input value={text} onChange={(e) => setText(e.target.value)} placeholder="一句话说明画面内容" />
        </label>
        <label className="fld">
          <span>拍摄时间 captured_at（可选）</span>
          <input value={capturedAt} onChange={(e) => setCapturedAt(e.target.value)} placeholder="2024-03-01T10:00:00+08:00" />
        </label>
        <button className="btn primary" disabled={!file || busy} onClick={submit}>
          {busy ? <><span className="spinner" />入库中</> : '入库'}
        </button>
      </div>
      <Outcome out={out} />
    </>
  )
}

function FactForm({ onDone }) {
  const [form, setForm] = useState({ subject: '', predicate: '', object: '', domain: '', note: '', confidence: 1.0 })
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }))

  const submit = useCallback(async () => {
    if (!form.subject.trim() || !form.predicate.trim() || !form.object.trim() || busy) return
    setBusy(true); setMsg('')
    try {
      await api.addFact({
        subject: form.subject.trim(),
        predicate: form.predicate.trim(),
        object: form.object.trim(),
        domain: form.domain.trim() || undefined,
        note: form.note.trim() || undefined,
        confidence: Number(form.confidence) || 1.0,
      })
      setMsg('ok')
      setForm({ subject: '', predicate: '', object: '', domain: '', note: '', confidence: 1.0 })
      onDone && onDataChanged()
    } catch (e) {
      setMsg(errMessage(e))
    }
    setBusy(false)
  }, [form, busy, onDone])

  return (
    <div className="card">
      <h2>手工添加三元组</h2>
      <p className="hint" style={{ margin: 0 }}>写进语义记忆（category=facts），随后出现在星云图的关系边里。</p>
      <div className="grid2" style={{ marginTop: 10 }}>
        <label className="fld"><span>主语 subject<span className="req">*</span></span><input value={form.subject} onChange={set('subject')} /></label>
        <label className="fld"><span>谓语 predicate<span className="req">*</span></span><input value={form.predicate} onChange={set('predicate')} /></label>
        <label className="fld"><span>宾语 object<span className="req">*</span></span><input value={form.object} onChange={set('object')} /></label>
        <label className="fld"><span>领域 domain</span><input value={form.domain} onChange={set('domain')} placeholder="留空则自动分类" /></label>
        <label className="fld"><span>备注 note</span><input value={form.note} onChange={set('note')} /></label>
        <label className="fld"><span>置信度 confidence</span><input type="number" min="0" max="1" step="0.05" value={form.confidence} onChange={set('confidence')} /></label>
      </div>
      <div className="row">
        <button className="btn primary" disabled={busy} onClick={submit}>{busy ? <><span className="spinner" />写入中</> : '写入'}</button>
        {msg && <span className={`pill ${msg === 'ok' ? 'ok' : 'err'}`}>{msg === 'ok' ? '已写入 ✓' : msg}</span>}
      </div>
    </div>
  )
}